"""
Data models for the nanook-curator system.

This module defines the core data structures used throughout the curation workflow,
including video metadata and the state object that flows through the LangGraph nodes.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import re


class VideoData(BaseModel):
    """
    Represents a YouTube video with all metadata required for curation.
    
    This model stores both basic video information and processed data
    including quality scores and extracted topics.
    """
    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    channel: str = Field(..., description="Channel name")
    view_count: int = Field(..., ge=0, description="Number of views")
    like_count: int = Field(..., ge=0, description="Number of likes")
    comment_count: int = Field(..., ge=0, description="Number of comments")
    upload_date: str = Field(..., description="Upload date in ISO format")
    transcript: Optional[str] = Field(None, description="Video transcript text")
    quality_score: Optional[float] = Field(None, ge=0, le=100, description="Quality score (0-100)")
    key_topics: List[str] = Field(default_factory=list, description="Extracted key topics")
    
    @validator('video_id')
    def validate_video_id(cls, v):
        """Validate YouTube video ID format."""
        if not v or len(v) != 11:
            raise ValueError('Video ID must be 11 characters long')
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Video ID contains invalid characters')
        return v
    
    @validator('upload_date')
    def validate_upload_date(cls, v):
        """Validate upload date format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError('Upload date must be in ISO format')
        return v
    
    @validator('title', 'channel')
    def validate_non_empty_strings(cls, v):
        """Ensure title and channel are not empty."""
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()
    
    def get_engagement_rate(self) -> float:
        """Calculate engagement rate as (likes + comments) / views."""
        if self.view_count == 0:
            return 0.0
        return (self.like_count + self.comment_count) / self.view_count
    
    def get_like_ratio(self) -> float:
        """Calculate like ratio as likes / (likes + estimated dislikes)."""
        # Since dislikes are not available, estimate based on engagement patterns
        # Assume healthy videos have 95%+ like ratio
        if self.like_count == 0:
            return 0.0
        # Simple heuristic: if engagement is high, assume good like ratio
        engagement = self.get_engagement_rate()
        return min(0.95 + (engagement * 0.05), 1.0)
    
    def is_recent(self, days_back: int = 7) -> bool:
        """Check if video was uploaded within specified days."""
        try:
            upload_dt = datetime.fromisoformat(self.upload_date.replace('Z', '+00:00'))
            days_old = (datetime.now().replace(tzinfo=upload_dt.tzinfo) - upload_dt).days
            return days_old <= days_back
        except ValueError:
            return False


class CuratorState(BaseModel):
    """
    State object that flows through the LangGraph workflow.
    
    This model maintains all data and configuration needed for the curation process,
    including search parameters, processing state, and results.
    """
    # Input parameters
    search_keywords: List[str] = Field(..., description="Keywords for video discovery")
    max_videos: int = Field(default=10, ge=1, le=50, description="Maximum videos to discover")
    days_back: int = Field(default=7, ge=1, le=30, description="Days back to search for videos")
    
    # Processing state
    discovered_videos: List[VideoData] = Field(default_factory=list, description="Videos found during discovery")
    processed_videos: List[VideoData] = Field(default_factory=list, description="Videos after quality evaluation")
    ranked_videos: List[VideoData] = Field(default_factory=list, description="Top-ranked videos for script generation")
    
    # Iterative refinement state
    search_attempt: int = Field(default=0, ge=0, description="Current search attempt number")
    max_search_attempts: int = Field(default=3, ge=1, le=10, description="Maximum search refinement attempts")
    current_search_terms: List[str] = Field(default_factory=list, description="Current search terms being used")
    quality_threshold: float = Field(default=70.0, ge=0, le=100, description="Minimum quality score threshold")
    min_quality_videos: int = Field(default=3, ge=1, description="Minimum number of quality videos required")
    
    # Output
    podcast_script: Optional[str] = Field(None, description="Generated podcast script")
    generation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about script generation")
    
    # Error handling
    errors: List[str] = Field(default_factory=list, description="List of errors encountered during processing")
    
    @validator('search_keywords')
    def validate_search_keywords(cls, v):
        """Ensure search keywords are not empty."""
        if not v:
            raise ValueError('At least one search keyword is required')
        # Remove empty strings and strip whitespace
        keywords = [kw.strip() for kw in v if kw.strip()]
        if not keywords:
            raise ValueError('Search keywords cannot be empty')
        return keywords
    
    @validator('current_search_terms', pre=True, always=True)
    def set_current_search_terms(cls, v, values):
        """Initialize current_search_terms with search_keywords if empty."""
        if not v and 'search_keywords' in values:
            return values['search_keywords'].copy()
        return v
    
    def add_error(self, error: str, node_name: str = None) -> None:
        """Add an error to the error list with optional node context."""
        timestamp = datetime.now().isoformat()
        if node_name:
            error_msg = f"[{timestamp}] {node_name}: {error}"
        else:
            error_msg = f"[{timestamp}] {error}"
        self.errors.append(error_msg)
    
    def has_sufficient_quality_videos(self) -> bool:
        """Check if we have enough videos meeting the quality threshold."""
        quality_videos = [v for v in self.ranked_videos if v.quality_score and v.quality_score >= self.quality_threshold]
        return len(quality_videos) >= self.min_quality_videos
    
    def can_refine_search(self) -> bool:
        """Check if we can still refine the search."""
        return self.search_attempt < self.max_search_attempts
    
    def get_top_videos(self, count: int = 5) -> List[VideoData]:
        """Get the top N videos by quality score."""
        sorted_videos = sorted(
            [v for v in self.ranked_videos if v.quality_score is not None],
            key=lambda x: x.quality_score,
            reverse=True
        )
        return sorted_videos[:count]
    
    def update_generation_metadata(self, **kwargs) -> None:
        """Update generation metadata with new key-value pairs."""
        self.generation_metadata.update(kwargs)
        self.generation_metadata['last_updated'] = datetime.now().isoformat()
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of the current processing state."""
        return {
            'search_attempt': self.search_attempt,
            'discovered_count': len(self.discovered_videos),
            'processed_count': len(self.processed_videos),
            'ranked_count': len(self.ranked_videos),
            'quality_videos_count': len([v for v in self.ranked_videos if v.quality_score and v.quality_score >= self.quality_threshold]),
            'has_script': self.podcast_script is not None,
            'error_count': len(self.errors),
            'can_refine': self.can_refine_search(),
            'meets_quality_threshold': self.has_sufficient_quality_videos()
        }