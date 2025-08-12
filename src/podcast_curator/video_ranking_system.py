"""
Video ranking system for the podcast-curator.

This module provides comprehensive video ranking capabilities including:
- Combined quality scoring (engagement + content + freshness)
- Quality threshold evaluation with minimum video requirements
- Advanced ranking algorithms for top video selection
- Quality assessment feedback for search refinement
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .models import VideoData, CuratorState
from .config import get_config, Configuration
from .engagement_analyzer import EngagementAnalyzer, EngagementMetrics
from .content_quality_scorer import ContentQualityScorer, ContentQualityMetrics

# Configure logging
logger = logging.getLogger(__name__)


class RankingStrategy(Enum):
    """Available ranking strategies."""
    BALANCED = "balanced"  # Balanced content, engagement, and freshness
    CONTENT_FOCUSED = "content_focused"  # Prioritize content quality
    ENGAGEMENT_FOCUSED = "engagement_focused"  # Prioritize engagement metrics
    FRESHNESS_FOCUSED = "freshness_focused"  # Prioritize recent content
    HYBRID = "hybrid"  # Adaptive strategy based on available content


class QualityBand(Enum):
    """Quality bands for video categorization."""
    EXCELLENT = "excellent"  # 90-100%
    VERY_GOOD = "very_good"  # 80-89%
    GOOD = "good"  # 70-79%
    FAIR = "fair"  # 60-69%
    POOR = "poor"  # Below 60%


@dataclass
class RankingMetrics:
    """Container for comprehensive ranking metrics."""
    content_score: float
    engagement_score: float
    freshness_score: float
    combined_score: float
    quality_band: QualityBand
    rank_position: int
    meets_threshold: bool
    detailed_metrics: Dict[str, Any]


@dataclass
class RankingResult:
    """Result of video ranking operation."""
    ranked_videos: List[VideoData]
    quality_summary: Dict[str, Any]
    threshold_analysis: Dict[str, Any]
    refinement_suggestions: List[str]
    ranking_metadata: Dict[str, Any]


class VideoRankingSystem:
    """
    Comprehensive video ranking system.
    
    Provides advanced ranking capabilities combining content quality,
    engagement metrics, and freshness factors with configurable strategies
    and quality threshold evaluation.
    """
    
    def __init__(self, config: Optional[Configuration] = None):
        """
        Initialize the video ranking system.
        
        Args:
            config: Optional configuration instance. If not provided, uses global config.
        """
        self.config = config or get_config()
        self.content_scorer = ContentQualityScorer(config=self.config)
        self.engagement_analyzer = EngagementAnalyzer(config=self.config)
        
        # Ranking configuration
        self.default_strategy = RankingStrategy.BALANCED
        self.quality_threshold = getattr(self.config, 'quality_threshold', 70.0)
        self.min_quality_videos = getattr(self.config, 'min_quality_videos', 3)
        
        # Scoring weights for different strategies
        self.strategy_weights = {
            RankingStrategy.BALANCED: {
                'content': 0.40,
                'engagement': 0.35,
                'freshness': 0.25
            },
            RankingStrategy.CONTENT_FOCUSED: {
                'content': 0.60,
                'engagement': 0.25,
                'freshness': 0.15
            },
            RankingStrategy.ENGAGEMENT_FOCUSED: {
                'content': 0.25,
                'engagement': 0.60,
                'freshness': 0.15
            },
            RankingStrategy.FRESHNESS_FOCUSED: {
                'content': 0.30,
                'engagement': 0.30,
                'freshness': 0.40
            },
            RankingStrategy.HYBRID: {
                'content': 0.35,
                'engagement': 0.35,
                'freshness': 0.30
            }
        }
        
        # Quality band thresholds
        self.quality_bands = {
            QualityBand.EXCELLENT: (90.0, 100.0),
            QualityBand.VERY_GOOD: (80.0, 89.9),
            QualityBand.GOOD: (70.0, 79.9),
            QualityBand.FAIR: (60.0, 69.9),
            QualityBand.POOR: (0.0, 59.9)
        }
    
    def calculate_freshness_score(self, video: VideoData) -> float:
        """
        Calculate freshness score based on upload date.
        
        Args:
            video: VideoData instance
            
        Returns:
            Freshness score (0-100) where newer videos score higher
        """
        try:
            # Parse upload date
            upload_date = datetime.fromisoformat(video.upload_date.replace('Z', '+00:00'))
            current_date = datetime.now(timezone.utc)
            
            # Calculate age in days
            age_days = (current_date - upload_date).days
            
            # Freshness scoring curve
            if age_days <= 1:  # Same day or yesterday
                return 100.0
            elif age_days <= 7:  # Within a week
                return 100.0 - (age_days - 1) * 5  # Decrease by 5 points per day
            elif age_days <= 30:  # Within a month
                return 70.0 - (age_days - 7) * 2  # Decrease by 2 points per day
            elif age_days <= 90:  # Within 3 months
                return 24.0 - (age_days - 30) * 0.3  # Decrease by 0.3 points per day
            elif age_days <= 365:  # Within a year
                return 6.0 - (age_days - 90) * 0.02  # Decrease by 0.02 points per day
            else:  # Older than a year
                return max(0.0, 1.0 - (age_days - 365) * 0.001)  # Very slow decrease
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Error calculating freshness for video {video.video_id}: {e}")
            return 50.0  # Neutral score if date parsing fails
    
    def get_quality_band(self, score: float) -> QualityBand:
        """
        Determine quality band for a given score.
        
        Args:
            score: Quality score (0-100)
            
        Returns:
            QualityBand enum value
        """
        for band, (min_score, max_score) in self.quality_bands.items():
            if min_score <= score <= max_score:
                return band
        return QualityBand.POOR
    
    def calculate_combined_score(self, video: VideoData, 
                                content_metrics: ContentQualityMetrics,
                                engagement_metrics: EngagementMetrics,
                                strategy: RankingStrategy = None) -> RankingMetrics:
        """
        Calculate combined quality score using specified ranking strategy.
        
        Args:
            video: VideoData instance
            content_metrics: Content quality metrics
            engagement_metrics: Engagement metrics
            strategy: Ranking strategy to use
            
        Returns:
            RankingMetrics with combined scoring
        """
        if strategy is None:
            strategy = self.default_strategy
        
        # Get individual scores
        content_score = content_metrics.final_quality_score
        engagement_score = engagement_metrics.overall_engagement_score
        freshness_score = self.calculate_freshness_score(video)
        
        # Get strategy weights
        weights = self.strategy_weights[strategy]
        
        # Calculate combined score
        combined_score = (
            content_score * weights['content'] +
            engagement_score * weights['engagement'] +
            freshness_score * weights['freshness']
        )
        
        # Determine quality band and threshold compliance
        quality_band = self.get_quality_band(combined_score)
        meets_threshold = combined_score >= self.quality_threshold
        
        # Compile detailed metrics
        detailed_metrics = {
            'strategy_used': strategy.value,
            'weights_applied': weights,
            'individual_scores': {
                'content': content_score,
                'engagement': engagement_score,
                'freshness': freshness_score
            },
            'content_breakdown': content_metrics.detailed_metrics,
            'engagement_breakdown': engagement_metrics.detailed_metrics,
            'video_age_days': self._calculate_video_age_days(video),
            'upload_date': video.upload_date
        }
        
        return RankingMetrics(
            content_score=content_score,
            engagement_score=engagement_score,
            freshness_score=freshness_score,
            combined_score=combined_score,
            quality_band=quality_band,
            rank_position=0,  # Will be set during ranking
            meets_threshold=meets_threshold,
            detailed_metrics=detailed_metrics
        )
    
    def _calculate_video_age_days(self, video: VideoData) -> int:
        """Calculate video age in days."""
        try:
            upload_date = datetime.fromisoformat(video.upload_date.replace('Z', '+00:00'))
            current_date = datetime.now(timezone.utc)
            return (current_date - upload_date).days
        except (ValueError, TypeError):
            return 999  # Very old if date parsing fails
    
    def rank_videos(self, videos: List[VideoData], 
                   strategy: RankingStrategy = None,
                   video_comments: Optional[Dict[str, List[str]]] = None,
                   target_count: int = 5) -> RankingResult:
        """
        Rank videos using comprehensive quality assessment.
        
        Args:
            videos: List of videos to rank
            strategy: Ranking strategy to use
            video_comments: Optional comments for each video
            target_count: Target number of top videos to select (3-5 as per requirements)
            
        Returns:
            RankingResult with ranked videos and analysis
        """
        if not videos:
            return RankingResult(
                ranked_videos=[],
                quality_summary={'total_videos': 0, 'videos_above_threshold': 0},
                threshold_analysis={'meets_minimum_requirement': False, 'shortfall': self.min_quality_videos},
                refinement_suggestions=['No videos provided for ranking'],
                ranking_metadata={'strategy': strategy.value if strategy else 'none'}
            )
        
        if strategy is None:
            strategy = self._select_adaptive_strategy(videos)
        
        logger.info(f"Ranking {len(videos)} videos using {strategy.value} strategy")
        
        # Calculate comprehensive metrics for each video
        video_rankings = []
        
        for video in videos:
            try:
                # Get comments for this video
                comments = video_comments.get(video.video_id, []) if video_comments else []
                
                # Calculate content quality metrics
                content_metrics = self.content_scorer.calculate_combined_quality_score(video, comments=comments)
                
                # Calculate engagement metrics
                engagement_metrics = self.engagement_analyzer.analyze_video_engagement(video, comments)
                
                # Calculate combined ranking metrics
                ranking_metrics = self.calculate_combined_score(video, content_metrics, engagement_metrics, strategy)
                
                # Attach metrics to video for later reference
                video.quality_score = ranking_metrics.combined_score
                if not hasattr(video, 'content_quality_analysis'):
                    video.content_quality_analysis = content_metrics
                if not hasattr(video, 'engagement_analysis'):
                    video.engagement_analysis = engagement_metrics
                if not hasattr(video, 'ranking_metrics'):
                    video.ranking_metrics = ranking_metrics
                
                video_rankings.append((video, ranking_metrics))
                
            except Exception as e:
                logger.error(f"Error ranking video {video.video_id}: {e}")
                # Create minimal ranking metrics for failed video
                failed_metrics = RankingMetrics(
                    content_score=0.0, engagement_score=0.0, freshness_score=0.0,
                    combined_score=0.0, quality_band=QualityBand.POOR,
                    rank_position=999, meets_threshold=False,
                    detailed_metrics={'error': str(e)}
                )
                video.quality_score = 0.0
                video_rankings.append((video, failed_metrics))
        
        # Sort by combined score (descending)
        video_rankings.sort(key=lambda x: x[1].combined_score, reverse=True)
        
        # Assign rank positions
        for i, (video, metrics) in enumerate(video_rankings):
            metrics.rank_position = i + 1
        
        # Extract ranked videos
        ranked_videos = [video for video, metrics in video_rankings]
        
        # Select top videos (limited by target_count)
        top_videos = ranked_videos[:target_count]
        
        # Analyze quality distribution and threshold compliance
        quality_summary = self._analyze_quality_distribution(video_rankings)
        threshold_analysis = self._analyze_threshold_compliance(video_rankings)
        refinement_suggestions = self._generate_refinement_suggestions(video_rankings, threshold_analysis)
        
        # Compile ranking metadata
        ranking_metadata = {
            'strategy': strategy.value,
            'total_videos_ranked': len(videos),
            'target_count': target_count,
            'selected_count': len(top_videos),
            'timestamp': datetime.now().isoformat(),
            'quality_threshold': self.quality_threshold,
            'min_quality_videos': self.min_quality_videos
        }
        
        logger.info(f"Ranking complete: {threshold_analysis['videos_above_threshold']}/{len(videos)} "
                   f"videos above {self.quality_threshold}% threshold")
        
        return RankingResult(
            ranked_videos=top_videos,
            quality_summary=quality_summary,
            threshold_analysis=threshold_analysis,
            refinement_suggestions=refinement_suggestions,
            ranking_metadata=ranking_metadata
        )
    
    def _select_adaptive_strategy(self, videos: List[VideoData]) -> RankingStrategy:
        """
        Select optimal ranking strategy based on video characteristics.
        
        Args:
            videos: List of videos to analyze
            
        Returns:
            Recommended ranking strategy
        """
        if not videos:
            return self.default_strategy
        
        # Analyze video characteristics
        total_videos = len(videos)
        recent_videos = sum(1 for v in videos if self._calculate_video_age_days(v) <= 7)
        has_transcripts = sum(1 for v in videos if v.transcript)
        
        recent_ratio = recent_videos / total_videos
        transcript_ratio = has_transcripts / total_videos
        
        # Strategy selection logic
        if recent_ratio > 0.7:  # Mostly recent content
            return RankingStrategy.FRESHNESS_FOCUSED
        elif transcript_ratio > 0.8:  # Most videos have transcripts (good for content analysis)
            return RankingStrategy.CONTENT_FOCUSED
        elif transcript_ratio < 0.3:  # Few transcripts (rely more on engagement)
            return RankingStrategy.ENGAGEMENT_FOCUSED
        else:
            return RankingStrategy.BALANCED
    
    def _analyze_quality_distribution(self, video_rankings: List[Tuple[VideoData, RankingMetrics]]) -> Dict[str, Any]:
        """Analyze quality score distribution across ranked videos."""
        if not video_rankings:
            return {
                'total_videos': 0,
                'average_score': 0.0,
                'score_distribution': {},
                'quality_bands': {}
            }
        
        scores = [metrics.combined_score for video, metrics in video_rankings]
        band_counts = {}
        
        for video, metrics in video_rankings:
            band = metrics.quality_band
            band_counts[band.value] = band_counts.get(band.value, 0) + 1
        
        return {
            'total_videos': len(video_rankings),
            'average_score': sum(scores) / len(scores),
            'median_score': sorted(scores)[len(scores) // 2],
            'min_score': min(scores),
            'max_score': max(scores),
            'score_distribution': {
                '90-100': len([s for s in scores if s >= 90]),
                '80-89': len([s for s in scores if 80 <= s < 90]),
                '70-79': len([s for s in scores if 70 <= s < 80]),
                '60-69': len([s for s in scores if 60 <= s < 70]),
                'below_60': len([s for s in scores if s < 60])
            },
            'quality_bands': band_counts
        }
    
    def _analyze_threshold_compliance(self, video_rankings: List[Tuple[VideoData, RankingMetrics]]) -> Dict[str, Any]:
        """Analyze compliance with quality threshold requirements."""
        if not video_rankings:
            return {
                'videos_above_threshold': 0,
                'threshold_percentage': 0.0,
                'meets_minimum_requirement': False,
                'shortfall': self.min_quality_videos,
                'threshold_used': self.quality_threshold
            }
        
        videos_above_threshold = sum(1 for video, metrics in video_rankings if metrics.meets_threshold)
        total_videos = len(video_rankings)
        
        meets_minimum = videos_above_threshold >= self.min_quality_videos
        shortfall = max(0, self.min_quality_videos - videos_above_threshold)
        
        return {
            'videos_above_threshold': videos_above_threshold,
            'total_videos': total_videos,
            'threshold_percentage': (videos_above_threshold / total_videos) * 100,
            'meets_minimum_requirement': meets_minimum,
            'shortfall': shortfall,
            'threshold_used': self.quality_threshold,
            'minimum_required': self.min_quality_videos
        }
    
    def _generate_refinement_suggestions(self, video_rankings: List[Tuple[VideoData, RankingMetrics]], 
                                       threshold_analysis: Dict[str, Any]) -> List[str]:
        """Generate suggestions for search refinement based on quality analysis."""
        suggestions = []
        
        if not video_rankings:
            suggestions.append("No videos found. Try broader search terms or increase search scope.")
            return suggestions
        
        # Analyze threshold compliance
        if not threshold_analysis['meets_minimum_requirement']:
            shortfall = threshold_analysis['shortfall']
            suggestions.append(f"Need {shortfall} more videos above {self.quality_threshold}% threshold. "
                             f"Consider expanding search terms or reducing quality requirements.")
        
        # Analyze score patterns
        scores = [metrics.combined_score for video, metrics in video_rankings]
        avg_score = sum(scores) / len(scores)
        
        if avg_score < 50:
            suggestions.append("Overall video quality is low. Try more specific or authoritative search terms.")
        elif avg_score < self.quality_threshold:
            suggestions.append("Video quality is below threshold. Consider refining search to focus on "
                             "higher-quality sources or more recent content.")
        
        # Analyze individual score components
        content_scores = [metrics.content_score for video, metrics in video_rankings]
        engagement_scores = [metrics.engagement_score for video, metrics in video_rankings]
        freshness_scores = [metrics.freshness_score for video, metrics in video_rankings]
        
        avg_content = sum(content_scores) / len(content_scores) if content_scores else 0
        avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0
        avg_freshness = sum(freshness_scores) / len(freshness_scores) if freshness_scores else 0
        
        if avg_content < 60:
            suggestions.append("Content quality is low. Search for more authoritative sources, "
                             "research papers, or educational content.")
        
        if avg_engagement < 60:
            suggestions.append("Engagement metrics are low. Look for more popular or well-received content.")
        
        if avg_freshness < 60:
            suggestions.append("Content is not recent. Consider adding date restrictions or "
                             "searching for more current topics.")
        
        # Transcript availability
        videos_with_transcripts = sum(1 for video, metrics in video_rankings if video.transcript)
        if videos_with_transcripts < len(video_rankings) * 0.5:
            suggestions.append("Many videos lack transcripts. This limits content quality analysis. "
                             "Try searching for educational or professional content more likely to have transcripts.")
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def evaluate_search_refinement_need(self, videos: List[VideoData], 
                                      strategy: RankingStrategy = None) -> Dict[str, Any]:
        """
        Evaluate whether search refinement is needed based on quality assessment.
        
        Args:
            videos: List of videos to evaluate
            strategy: Ranking strategy to use for evaluation
            
        Returns:
            Dictionary with refinement assessment and recommendations
        """
        # Perform ranking to get quality metrics
        ranking_result = self.rank_videos(videos, strategy=strategy)
        
        # Determine if refinement is needed
        needs_refinement = not ranking_result.threshold_analysis['meets_minimum_requirement']
        
        # Calculate confidence in current results
        if ranking_result.quality_summary['total_videos'] == 0:
            confidence = 0.0
        else:
            avg_score = ranking_result.quality_summary['average_score']
            threshold_compliance = ranking_result.threshold_analysis['threshold_percentage']
            confidence = min(100.0, (avg_score + threshold_compliance) / 2)
        
        # Determine refinement priority
        shortfall = ranking_result.threshold_analysis['shortfall']
        if shortfall >= 3:
            priority = 'high'
        elif shortfall >= 1:
            priority = 'medium'
        else:
            priority = 'low'
        
        return {
            'needs_refinement': needs_refinement,
            'confidence_score': confidence,
            'refinement_priority': priority,
            'quality_summary': ranking_result.quality_summary,
            'threshold_analysis': ranking_result.threshold_analysis,
            'refinement_suggestions': ranking_result.refinement_suggestions,
            'current_best_videos': len(ranking_result.ranked_videos),
            'recommended_actions': self._generate_refinement_actions(ranking_result)
        }
    
    def _generate_refinement_actions(self, ranking_result: RankingResult) -> List[Dict[str, Any]]:
        """Generate specific refinement actions based on ranking results."""
        actions = []
        
        threshold_analysis = ranking_result.threshold_analysis
        quality_summary = ranking_result.quality_summary
        
        # Action: Expand search if too few videos
        if quality_summary['total_videos'] < 10:
            actions.append({
                'action': 'expand_search_scope',
                'description': 'Increase search scope to find more videos',
                'priority': 'high',
                'parameters': {'increase_max_videos': True, 'broaden_keywords': True}
            })
        
        # Action: Refine keywords if quality is low
        if quality_summary['average_score'] < 60:
            actions.append({
                'action': 'refine_keywords',
                'description': 'Use more specific or authoritative search terms',
                'priority': 'high',
                'parameters': {'add_quality_terms': True, 'focus_on_authority': True}
            })
        
        # Action: Adjust time range if freshness is an issue
        freshness_issues = any('recent' in suggestion for suggestion in ranking_result.refinement_suggestions)
        if freshness_issues:
            actions.append({
                'action': 'adjust_time_range',
                'description': 'Focus on more recent content',
                'priority': 'medium',
                'parameters': {'reduce_days_back': True, 'prioritize_recent': True}
            })
        
        # Action: Lower threshold if consistently falling short
        if threshold_analysis['shortfall'] > 2 and quality_summary['average_score'] > 50:
            actions.append({
                'action': 'consider_threshold_adjustment',
                'description': 'Consider lowering quality threshold given available content',
                'priority': 'low',
                'parameters': {'suggested_threshold': max(50.0, quality_summary['average_score'] - 5)}
            })
        
        return actions
    
    def update_curator_state(self, state: CuratorState, ranking_result: RankingResult) -> CuratorState:
        """
        Update curator state with ranking results.
        
        Args:
            state: Current curator state
            ranking_result: Results from video ranking
            
        Returns:
            Updated curator state
        """
        # Update ranked videos
        state.ranked_videos = ranking_result.ranked_videos
        
        # Update generation metadata
        state.update_generation_metadata(
            ranking_strategy=ranking_result.ranking_metadata['strategy'],
            quality_summary=ranking_result.quality_summary,
            threshold_analysis=ranking_result.threshold_analysis,
            videos_above_threshold=ranking_result.threshold_analysis['videos_above_threshold'],
            meets_quality_requirement=ranking_result.threshold_analysis['meets_minimum_requirement']
        )
        
        # Add refinement suggestions if quality requirements not met
        if not ranking_result.threshold_analysis['meets_minimum_requirement']:
            for suggestion in ranking_result.refinement_suggestions:
                state.add_error(f"Quality refinement needed: {suggestion}", "ranking_system")
        
        return state
    
    def get_ranking_summary(self, videos: List[VideoData]) -> Dict[str, Any]:
        """
        Get a comprehensive ranking summary for a list of videos.
        
        Args:
            videos: List of videos to analyze
            
        Returns:
            Summary statistics and insights
        """
        if not videos:
            return {
                'total_videos': 0,
                'strategy_recommendation': self.default_strategy.value,
                'quality_outlook': 'no_data'
            }
        
        # Quick analysis without full ranking
        recent_count = sum(1 for v in videos if self._calculate_video_age_days(v) <= 7)
        transcript_count = sum(1 for v in videos if v.transcript)
        
        # Estimate quality potential
        transcript_ratio = transcript_count / len(videos)
        recent_ratio = recent_count / len(videos)
        
        if transcript_ratio > 0.7 and recent_ratio > 0.5:
            quality_outlook = 'excellent'
        elif transcript_ratio > 0.5 or recent_ratio > 0.7:
            quality_outlook = 'good'
        elif transcript_ratio > 0.3 and recent_ratio > 0.3:
            quality_outlook = 'fair'
        else:
            quality_outlook = 'challenging'
        
        return {
            'total_videos': len(videos),
            'videos_with_transcripts': transcript_count,
            'recent_videos': recent_count,
            'transcript_ratio': transcript_ratio,
            'recent_ratio': recent_ratio,
            'strategy_recommendation': self._select_adaptive_strategy(videos).value,
            'quality_outlook': quality_outlook,
            'estimated_threshold_compliance': min(100, int((transcript_ratio + recent_ratio) * 60))
        }