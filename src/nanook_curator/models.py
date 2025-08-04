"""
Data models for the nanook-curator system.

This module defines the core data structures used throughout the curation workflow,
including video metadata and the state object that flows through the LangGraph nodes.
"""

from typing import List, Dict, Optional, Any, Annotated
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import re
import operator


def _replace_value(left, right):
    """
    Reducer that safely replaces values while preserving state object integrity.
    
    This handles LangGraph's state management during transitions and ensures
    we always maintain CuratorState objects, not dictionaries.
    """
    # If right is None, keep the left value
    if right is None:
        return left
    
    # If both are the same, no change needed
    if left == right:
        return left
    
    # For scalar values, just return the new value
    if not isinstance(right, (dict, list)):
        return right
    
    # If right is a dictionary but left isn't, this might be a partial update
    # In this case, preserve the left value to maintain state integrity
    if isinstance(right, dict) and not isinstance(left, dict):
        return left
        
    # Otherwise, return the right value
    return right


class VideoData(BaseModel):
    """
    Represents a YouTube video with all metadata required for curation.
    
    This model stores both basic video information and processed data
    including quality scores and extracted topics.
    """
    model_config = {"extra": "allow"}  # Allow extra attributes for enhanced metrics
    
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
    
    @field_validator('video_id')
    @classmethod
    def validate_video_id(cls, v):
        """Validate YouTube video ID format."""
        if not v or len(v) != 11:
            raise ValueError('Video ID must be 11 characters long')
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Video ID contains invalid characters')
        return v
    
    @field_validator('upload_date')
    @classmethod
    def validate_upload_date(cls, v):
        """Validate upload date format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError('Upload date must be in ISO format')
        return v
    
    @field_validator('title', 'channel')
    @classmethod
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
        # Check if enhanced engagement metrics are available
        if hasattr(self, 'engagement_metrics') and 'likeRatio' in self.engagement_metrics:
            return self.engagement_metrics['likeRatio']
        
        # Fallback to simple estimation if enhanced metrics not available
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
    
    def get_view_to_subscriber_ratio(self) -> float:
        """Get view-to-subscriber ratio from enhanced engagement metrics."""
        if hasattr(self, 'engagement_metrics') and 'viewToSubscriberRatio' in self.engagement_metrics:
            return self.engagement_metrics['viewToSubscriberRatio']
        return 0.0
    
    def get_engagement_score(self) -> float:
        """Get overall engagement score (0-100) from enhanced metrics."""
        if hasattr(self, 'engagement_metrics') and 'engagementScore' in self.engagement_metrics:
            return self.engagement_metrics['engagementScore']
        # Fallback calculation
        return min(self.get_engagement_rate() * 1000, 100.0)
    
    def get_like_to_view_ratio(self) -> float:
        """Get like-to-view ratio from enhanced engagement metrics."""
        if hasattr(self, 'engagement_metrics') and 'likeToViewRatio' in self.engagement_metrics:
            return self.engagement_metrics['likeToViewRatio']
        # Fallback calculation
        return self.like_count / self.view_count if self.view_count > 0 else 0.0
    
    def get_comment_to_view_ratio(self) -> float:
        """Get comment-to-view ratio from enhanced engagement metrics."""
        if hasattr(self, 'engagement_metrics') and 'commentToViewRatio' in self.engagement_metrics:
            return self.engagement_metrics['commentToViewRatio']
        # Fallback calculation
        return self.comment_count / self.view_count if self.view_count > 0 else 0.0
    
    def has_enhanced_metrics(self) -> bool:
        """Check if this video has enhanced engagement metrics from detailed fetching."""
        return hasattr(self, 'engagement_metrics') and bool(self.engagement_metrics)
    
    def get_channel_subscriber_count(self) -> int:
        """Get the subscriber count of the video's channel."""
        if hasattr(self, 'channel_subscriber_count'):
            return self.channel_subscriber_count
        return 0
    
    def is_available_for_processing(self) -> bool:
        """Check if video is available for processing based on status information."""
        if not hasattr(self, 'video_status'):
            return True  # Assume available if no status info
        
        status = self.video_status
        privacy_status = status.get('privacyStatus', '')
        upload_status = status.get('uploadStatus', '')
        
        # Check if video is public and processed
        if privacy_status in ['private', 'privacyStatusUnspecified']:
            return False
        
        if upload_status and upload_status not in ['processed', '']:
            return False
        
        return True


class CuratorState(BaseModel):
    """
    State object that flows through the LangGraph workflow.
    
    This model maintains all data and configuration needed for the curation process,
    including search parameters, processing state, and results.
    """
    # Input parameters
    search_keywords: Annotated[List[str], operator.add] = Field(..., description="Keywords for video discovery")
    max_videos: Annotated[int, _replace_value] = Field(default=10, ge=1, le=50, description="Maximum videos to discover")
    days_back: Annotated[int, _replace_value] = Field(default=7, ge=1, le=30, description="Days back to search for videos")
    
    # Processing state
    discovered_videos: Annotated[List[VideoData], operator.add] = Field(default_factory=list, description="Videos found during discovery")
    processed_videos: Annotated[List[VideoData], operator.add] = Field(default_factory=list, description="Videos after quality evaluation")
    ranked_videos: Annotated[List[VideoData], operator.add] = Field(default_factory=list, description="Top-ranked videos for script generation")
    
    # Iterative refinement state
    search_attempt: Annotated[int, _replace_value] = Field(default=0, ge=0, description="Current search attempt number")
    max_search_attempts: Annotated[int, _replace_value] = Field(default=3, ge=1, le=10, description="Maximum search refinement attempts")
    current_search_terms: Annotated[List[str], operator.add] = Field(default_factory=list, description="Current search terms being used")
    quality_threshold: Annotated[float, _replace_value] = Field(default=70.0, ge=0, le=100, description="Minimum quality score threshold")
    min_quality_videos: Annotated[int, _replace_value] = Field(default=3, ge=1, description="Minimum number of quality videos required")
    
    # Output
    podcast_script: Annotated[Optional[str], _replace_value] = Field(None, description="Generated podcast script")
    generation_metadata: Annotated[Dict[str, Any], _replace_value] = Field(default_factory=dict, description="Metadata about script generation")
    
    # Error handling
    errors: Annotated[List[str], operator.add] = Field(default_factory=list, description="List of errors encountered during processing")
    
    @field_validator('search_keywords')
    @classmethod
    def validate_search_keywords(cls, v):
        """Ensure search keywords are not empty."""
        if not v:
            raise ValueError('At least one search keyword is required')
        # Remove empty strings and strip whitespace
        keywords = [kw.strip() for kw in v if kw.strip()]
        if not keywords:
            raise ValueError('Search keywords cannot be empty')
        return keywords
    
    @field_validator('current_search_terms', mode='before')
    @classmethod
    def set_current_search_terms(cls, v, info):
        """Initialize current_search_terms with search_keywords if empty."""
        if not v and info.data and 'search_keywords' in info.data:
            return info.data['search_keywords'].copy()
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