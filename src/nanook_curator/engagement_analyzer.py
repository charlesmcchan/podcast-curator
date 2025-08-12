"""
Engagement metrics analyzer for the nanook-curator system.

This module provides comprehensive engagement analysis for YouTube videos including:
- Like-to-dislike ratio calculation with configurable thresholds
- Comment sentiment analysis 
- View-to-subscriber ratio evaluation
- Overall engagement score calculation
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .models import VideoData
from .config import get_config, Configuration

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EngagementMetrics:
    """Container for comprehensive engagement metrics."""
    like_ratio: float
    like_to_dislike_ratio: float
    view_to_subscriber_ratio: float
    comment_sentiment_score: float
    engagement_rate: float
    overall_engagement_score: float
    meets_threshold: bool
    detailed_metrics: Dict[str, Any]


class EngagementAnalyzer:
    """
    Comprehensive engagement metrics analyzer for YouTube videos.
    
    Provides analysis capabilities for various engagement metrics including
    like ratios, sentiment analysis, and combined scoring.
    """
    
    def __init__(self, config: Optional[Configuration] = None):
        """
        Initialize the engagement analyzer.
        
        Args:
            config: Optional configuration instance. If not provided, uses global config.
        """
        self.config = config or get_config()
        self.like_ratio_threshold = 0.8  # 80% minimum threshold as per requirements
        
        # Sentiment analysis keywords (simple approach without external dependencies)
        self.positive_keywords = {
            'excellent', 'amazing', 'great', 'fantastic', 'awesome', 'wonderful',
            'perfect', 'love', 'brilliant', 'outstanding', 'incredible', 'superb',
            'good', 'nice', 'helpful', 'useful', 'informative', 'clear', 'thanks',
            'thank you', 'appreciate', 'learned', 'understand', 'explained'
        }
        
        self.negative_keywords = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'disgusting',
            'stupid', 'boring', 'useless', 'waste', 'disappointed', 'confused',
            'unclear', 'wrong', 'incorrect', 'misleading', 'clickbait', 'fake'
        }
    
    def calculate_like_to_dislike_ratio(self, video: VideoData) -> Tuple[float, float]:
        """
        Calculate like-to-dislike ratio with 80% minimum threshold.
        
        Since YouTube removed public dislike counts, this uses estimation based on
        engagement patterns and like ratios from available data.
        
        Args:
            video: VideoData instance with engagement metrics
            
        Returns:
            Tuple of (like_ratio, estimated_like_to_dislike_ratio)
        """
        if video.like_count == 0:
            return 0.0, 0.0
        
        # Check if enhanced metrics are available with actual like ratio
        if hasattr(video, 'engagement_metrics') and 'likeRatio' in video.engagement_metrics:
            like_ratio = video.engagement_metrics['likeRatio']
        else:
            # Estimate like ratio based on engagement patterns
            engagement_rate = video.get_engagement_rate()
            view_count = video.view_count
            
            # Higher engagement typically correlates with better like ratios
            # Adjust based on video maturity and engagement patterns
            base_ratio = 0.85  # Conservative baseline
            
            # Bonus for high engagement relative to view count
            if view_count > 0:
                engagement_bonus = min(engagement_rate * 10, 0.1)  # Up to 10% bonus
                like_ratio = min(base_ratio + engagement_bonus, 0.98)
            else:
                like_ratio = base_ratio
        
        # Calculate like-to-dislike ratio
        # like_ratio = likes / (likes + dislikes)
        # Solving for like_to_dislike_ratio = likes / dislikes
        if like_ratio >= 0.99:
            like_to_dislike_ratio = 99.0  # Very high ratio for near-perfect scores
        elif like_ratio <= 0.01:
            like_to_dislike_ratio = 0.01  # Very low ratio
        else:
            like_to_dislike_ratio = like_ratio / (1 - like_ratio)
        
        return like_ratio, like_to_dislike_ratio
    
    def analyze_comment_sentiment(self, comments: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment of video comments using keyword-based approach.
        
        Args:
            comments: List of comment texts to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not comments:
            return {
                'sentiment_score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_analyzed': 0,
                'sentiment_distribution': {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            }
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for comment in comments:
            if not comment:
                continue
                
            # Clean and normalize comment text
            clean_comment = re.sub(r'[^\w\s]', ' ', comment.lower())
            words = set(clean_comment.split())
            
            # Count sentiment indicators
            positive_matches = len(words.intersection(self.positive_keywords))
            negative_matches = len(words.intersection(self.negative_keywords))
            
            if positive_matches > negative_matches:
                positive_count += 1
            elif negative_matches > positive_matches:
                negative_count += 1
            else:
                neutral_count += 1
        
        total_analyzed = len(comments)
        
        # Calculate sentiment score (-1 to 1, where 1 is most positive)
        if total_analyzed > 0:
            sentiment_score = (positive_count - negative_count) / total_analyzed
        else:
            sentiment_score = 0.0
        
        # Calculate distribution
        if total_analyzed > 0:
            pos_pct = positive_count / total_analyzed
            neg_pct = negative_count / total_analyzed
            neu_pct = neutral_count / total_analyzed
        else:
            pos_pct = neg_pct = neu_pct = 0.0
        
        return {
            'sentiment_score': sentiment_score,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_analyzed': total_analyzed,
            'sentiment_distribution': {
                'positive': pos_pct,
                'negative': neg_pct,
                'neutral': neu_pct
            }
        }
    
    def calculate_view_to_subscriber_ratio(self, video: VideoData) -> float:
        """
        Calculate view-to-subscriber ratio for engagement evaluation.
        
        Args:
            video: VideoData instance with view and subscriber information
            
        Returns:
            View-to-subscriber ratio (views per subscriber)
        """
        # Check if enhanced metrics include this ratio
        if hasattr(video, 'engagement_metrics') and 'viewToSubscriberRatio' in video.engagement_metrics:
            return video.engagement_metrics['viewToSubscriberRatio']
        
        # Get subscriber count from video data
        subscriber_count = video.get_channel_subscriber_count()
        
        if subscriber_count == 0:
            return 0.0
        
        return video.view_count / subscriber_count
    
    def calculate_overall_engagement_score(self, video: VideoData, 
                                         comment_sentiment: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate overall engagement score combining multiple metrics.
        
        Args:
            video: VideoData instance
            comment_sentiment: Optional comment sentiment analysis results
            
        Returns:
            Overall engagement score (0-100)
        """
        # Get individual metrics
        like_ratio, like_to_dislike_ratio = self.calculate_like_to_dislike_ratio(video)
        view_to_sub_ratio = self.calculate_view_to_subscriber_ratio(video)
        basic_engagement_rate = video.get_engagement_rate()
        
        # Component scores (0-100 each)
        scores = []
        weights = []
        
        # Like ratio component (30% weight)
        like_ratio_score = like_ratio * 100
        scores.append(like_ratio_score)
        weights.append(0.30)
        
        # Basic engagement rate component (25% weight)
        # Scale engagement rate to 0-100 (typical good engagement is 2-5%)
        engagement_rate_score = min(basic_engagement_rate * 2000, 100)
        scores.append(engagement_rate_score)
        weights.append(0.25)
        
        # View-to-subscriber ratio component (20% weight)
        # Good ratios vary by channel size, but 0.1-0.5 is typically good
        if view_to_sub_ratio > 0:
            view_sub_score = min(view_to_sub_ratio * 200, 100)
        else:
            view_sub_score = 50  # Neutral score if no subscriber data
        scores.append(view_sub_score)
        weights.append(0.20)
        
        # Comment sentiment component (15% weight)
        if comment_sentiment:
            # Convert sentiment score (-1 to 1) to 0-100 scale
            sentiment_normalized = (comment_sentiment['sentiment_score'] + 1) / 2
            sentiment_score = sentiment_normalized * 100
        else:
            sentiment_score = 50  # Neutral score if no sentiment data
        scores.append(sentiment_score)
        weights.append(0.15)
        
        # Like-to-dislike ratio bonus (10% weight)
        # Scale ratio to score (good ratio is 4:1 or higher)
        ltd_score = min(like_to_dislike_ratio * 25, 100) if like_to_dislike_ratio > 0 else 0
        scores.append(ltd_score)
        weights.append(0.10)
        
        # Calculate weighted average
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return min(weighted_score, 100.0)
    
    def analyze_video_engagement(self, video: VideoData, 
                               comments: Optional[List[str]] = None) -> EngagementMetrics:
        """
        Perform comprehensive engagement analysis on a video.
        
        Args:
            video: VideoData instance to analyze
            comments: Optional list of comment texts for sentiment analysis
            
        Returns:
            EngagementMetrics with complete analysis results
        """
        try:
            # Calculate core metrics
            like_ratio, like_to_dislike_ratio = self.calculate_like_to_dislike_ratio(video)
            view_to_sub_ratio = self.calculate_view_to_subscriber_ratio(video)
            
            # Analyze comment sentiment
            comment_sentiment = self.analyze_comment_sentiment(comments or [])
            
            # Calculate overall engagement score
            overall_score = self.calculate_overall_engagement_score(video, comment_sentiment)
            
            # Check if meets threshold
            meets_threshold = like_ratio >= self.like_ratio_threshold
            
            # Compile detailed metrics
            detailed_metrics = {
                'like_count': video.like_count,
                'view_count': video.view_count,
                'comment_count': video.comment_count,
                'subscriber_count': video.get_channel_subscriber_count(),
                'basic_engagement_rate': video.get_engagement_rate(),
                'comment_sentiment': comment_sentiment,
                'threshold_used': self.like_ratio_threshold,
                'calculation_method': 'estimated' if not hasattr(video, 'engagement_metrics') else 'enhanced'
            }
            
            return EngagementMetrics(
                like_ratio=like_ratio,
                like_to_dislike_ratio=like_to_dislike_ratio,
                view_to_subscriber_ratio=view_to_sub_ratio,
                comment_sentiment_score=comment_sentiment['sentiment_score'],
                engagement_rate=video.get_engagement_rate(),
                overall_engagement_score=overall_score,
                meets_threshold=meets_threshold,
                detailed_metrics=detailed_metrics
            )
            
        except Exception as e:
            logger.error(f"Error analyzing engagement for video {video.video_id}: {e}")
            
            # Return minimal metrics on error
            return EngagementMetrics(
                like_ratio=0.0,
                like_to_dislike_ratio=0.0,
                view_to_subscriber_ratio=0.0,
                comment_sentiment_score=0.0,
                engagement_rate=0.0,
                overall_engagement_score=0.0,
                meets_threshold=False,
                detailed_metrics={'error': str(e)}
            )
    
    def batch_analyze_engagement(self, videos: List[VideoData], 
                               video_comments: Optional[Dict[str, List[str]]] = None) -> Dict[str, EngagementMetrics]:
        """
        Analyze engagement metrics for multiple videos.
        
        Args:
            videos: List of VideoData instances to analyze
            video_comments: Optional dictionary mapping video_id to list of comments
            
        Returns:
            Dictionary mapping video_id to EngagementMetrics
        """
        results = {}
        
        for video in videos:
            comments = video_comments.get(video.video_id, []) if video_comments else []
            results[video.video_id] = self.analyze_video_engagement(video, comments)
        
        return results
    
    def filter_videos_by_engagement(self, videos: List[VideoData], 
                                  min_engagement_score: float = 60.0,
                                  require_threshold: bool = True) -> List[VideoData]:
        """
        Filter videos based on engagement criteria.
        
        Args:
            videos: List of videos to filter
            min_engagement_score: Minimum overall engagement score required
            require_threshold: Whether to require meeting the like ratio threshold
            
        Returns:
            List of videos meeting engagement criteria
        """
        filtered_videos = []
        
        for video in videos:
            metrics = self.analyze_video_engagement(video)
            
            # Check criteria
            meets_score = metrics.overall_engagement_score >= min_engagement_score
            meets_threshold_req = not require_threshold or metrics.meets_threshold
            
            if meets_score and meets_threshold_req:
                # Attach engagement metrics to video for later use
                if not hasattr(video, 'engagement_analysis'):
                    video.engagement_analysis = metrics
                filtered_videos.append(video)
        
        return filtered_videos
    
    def get_engagement_summary(self, videos: List[VideoData]) -> Dict[str, Any]:
        """
        Get summary statistics for engagement across multiple videos.
        
        Args:
            videos: List of videos to summarize
            
        Returns:
            Dictionary with engagement summary statistics
        """
        if not videos:
            return {
                'total_videos': 0,
                'avg_engagement_score': 0.0,
                'avg_like_ratio': 0.0,
                'videos_meeting_threshold': 0,
                'threshold_rate': 0.0
            }
        
        total_engagement = 0.0
        total_like_ratio = 0.0
        threshold_count = 0
        
        for video in videos:
            metrics = self.analyze_video_engagement(video)
            total_engagement += metrics.overall_engagement_score
            total_like_ratio += metrics.like_ratio
            if metrics.meets_threshold:
                threshold_count += 1
        
        return {
            'total_videos': len(videos),
            'avg_engagement_score': total_engagement / len(videos),
            'avg_like_ratio': total_like_ratio / len(videos),
            'videos_meeting_threshold': threshold_count,
            'threshold_rate': threshold_count / len(videos)
        }