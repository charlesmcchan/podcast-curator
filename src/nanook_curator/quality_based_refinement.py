"""
Quality-based search refinement system for the nanook-curator.

This module provides quality-based refinement capabilities including:
- Quality failure analysis and categorization
- Refinement triggers when quality threshold not met
- Search term adjustment based on quality failure analysis
- Logic to return to discovery with expanded parameters
- Weekly focus maintenance during refinement
"""

import logging
import statistics
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .models import VideoData, CuratorState
from .config import get_config, Configuration
from .video_ranking_system import VideoRankingSystem, RankingResult
from .search_refinement import SearchRefinementEngine, RefinementStrategy
from .youtube_client import YouTubeClient

# Configure logging
logger = logging.getLogger(__name__)


class QualityFailureType(Enum):
    """Types of quality failures that can trigger refinement."""
    INSUFFICIENT_COUNT = "insufficient_count"  # Not enough videos above threshold
    LOW_AVERAGE_QUALITY = "low_average_quality"  # Overall quality too low
    POOR_CONTENT_QUALITY = "poor_content_quality"  # Content analysis scores low
    LOW_ENGAGEMENT = "low_engagement"  # Engagement metrics below minimum
    STALE_CONTENT = "stale_content"  # Content is too old/not fresh
    NO_TRANSCRIPTS = "no_transcripts"  # Too many videos lack transcripts
    MIXED_QUALITY = "mixed_quality"  # High variance in quality scores


@dataclass
class QualityAnalysis:
    """Analysis of quality failures and refinement recommendations."""
    failure_types: List[QualityFailureType]
    content_issues: List[str]
    engagement_issues: List[str]
    freshness_issues: List[str]
    transcript_issues: List[str]
    score_statistics: Dict[str, Any]
    refinement_urgency: str  # 'low', 'medium', 'high', 'critical'
    suggested_adjustments: List[Dict[str, Any]]
    weekly_focus_impact: Dict[str, Any]


@dataclass
class QualityRefinementAction:
    """Specific refinement action to improve quality."""
    action_type: str  # 'expand_keywords', 'refine_keywords', 'enhance_freshness', etc.
    description: str
    parameters: Dict[str, Any]
    priority: str  # 'low', 'medium', 'high'
    expected_improvement: str
    weekly_focus_preservation: Dict[str, Any]


class QualityBasedRefinementEngine:
    """
    Quality-based search refinement engine.
    
    Analyzes quality failures and implements targeted refinement strategies
    to improve search results while maintaining weekly focus requirements.
    """
    
    def __init__(self, config: Optional[Configuration] = None):
        """
        Initialize the quality-based refinement engine.
        
        Args:
            config: Optional configuration instance. If not provided, uses global config.
        """
        self.config = config or get_config()
        self.ranking_system = VideoRankingSystem(config=self.config)
        self.search_refinement = SearchRefinementEngine(config=self.config)
        self.youtube_client = YouTubeClient(config=self.config)
        
        # Quality thresholds and requirements
        self.quality_threshold = getattr(self.config, 'quality_threshold', 70.0)
        self.min_quality_videos = getattr(self.config, 'min_quality_videos', 3)
        self.acceptable_average_quality = self.quality_threshold - 15.0  # 15 points below threshold
        
        # Weekly focus configuration
        self.core_weekly_days = 7  # Core weekly focus window
        self.max_weekly_extension = 14  # Maximum extension while maintaining weekly focus
        
        # Quality failure thresholds
        self.failure_thresholds = {
            'minimum_average_quality': 50.0,  # Below this triggers LOW_AVERAGE_QUALITY
            'minimum_content_score': 50.0,    # Below this triggers POOR_CONTENT_QUALITY
            'minimum_engagement_score': 40.0,  # Below this triggers LOW_ENGAGEMENT
            'minimum_freshness_score': 30.0,   # Below this triggers STALE_CONTENT
            'minimum_transcript_ratio': 0.3,   # Below this triggers NO_TRANSCRIPTS
            'quality_variance_threshold': 25.0  # Above this triggers MIXED_QUALITY
        }
        
        # Refinement priority mapping
        self.failure_priorities = {
            QualityFailureType.INSUFFICIENT_COUNT: 'critical',
            QualityFailureType.LOW_AVERAGE_QUALITY: 'high',
            QualityFailureType.POOR_CONTENT_QUALITY: 'medium',
            QualityFailureType.LOW_ENGAGEMENT: 'medium',
            QualityFailureType.STALE_CONTENT: 'medium',
            QualityFailureType.NO_TRANSCRIPTS: 'low',
            QualityFailureType.MIXED_QUALITY: 'low'
        }
    
    def analyze_quality_failures(self, videos: List[VideoData], 
                                ranking_result: RankingResult) -> QualityAnalysis:
        """
        Analyze quality failures in the current video set.
        
        Args:
            videos: List of videos to analyze
            ranking_result: Results from video ranking
            
        Returns:
            QualityAnalysis with detailed failure analysis
        """
        logger.info(f"Analyzing quality failures for {len(videos)} videos")
        
        failure_types = []
        content_issues = []
        engagement_issues = []
        freshness_issues = []
        transcript_issues = []
        
        # Calculate score statistics
        score_stats = self._calculate_score_statistics(videos, ranking_result)
        
        # Check for insufficient count (most critical)
        videos_above_threshold = ranking_result.threshold_analysis['videos_above_threshold']
        if videos_above_threshold < self.min_quality_videos:
            failure_types.append(QualityFailureType.INSUFFICIENT_COUNT)
            shortfall = self.min_quality_videos - videos_above_threshold
            content_issues.append(f"Need {shortfall} more videos above {self.quality_threshold}% threshold")
        
        # Check for low average quality - use ranking result if individual scores not available
        avg_quality = ranking_result.quality_summary.get('average_score', score_stats.get('avg_quality', 0))
        if avg_quality <= self.failure_thresholds['minimum_average_quality']:
            failure_types.append(QualityFailureType.LOW_AVERAGE_QUALITY)
            content_issues.append(f"Average quality ({avg_quality:.1f}%) at or below minimum ({self.failure_thresholds['minimum_average_quality']}%)")
        
        # Analyze individual score components
        content_scores = []
        engagement_scores = []
        freshness_scores = []
        
        for video in videos:
            if hasattr(video, 'ranking_metrics'):
                metrics = video.ranking_metrics
                content_scores.append(metrics.content_score)
                engagement_scores.append(metrics.engagement_score)
                freshness_scores.append(metrics.freshness_score)
        
        # Check content quality
        if content_scores:
            avg_content = sum(content_scores) / len(content_scores)
            if avg_content < self.failure_thresholds['minimum_content_score']:
                failure_types.append(QualityFailureType.POOR_CONTENT_QUALITY)
                content_issues.append(f"Average content quality ({avg_content:.1f}%) below minimum")
        
        # Check engagement
        if engagement_scores:
            avg_engagement = sum(engagement_scores) / len(engagement_scores)
            if avg_engagement < self.failure_thresholds['minimum_engagement_score']:
                failure_types.append(QualityFailureType.LOW_ENGAGEMENT)
                engagement_issues.append(f"Average engagement ({avg_engagement:.1f}%) below minimum")
        
        # Check freshness
        if freshness_scores:
            avg_freshness = sum(freshness_scores) / len(freshness_scores)
            if avg_freshness < self.failure_thresholds['minimum_freshness_score']:
                failure_types.append(QualityFailureType.STALE_CONTENT)
                freshness_issues.append(f"Average freshness ({avg_freshness:.1f}%) below minimum")
        
        # Check transcript availability
        videos_with_transcripts = sum(1 for v in videos if v.transcript)
        transcript_ratio = videos_with_transcripts / len(videos) if videos else 0
        if transcript_ratio < self.failure_thresholds['minimum_transcript_ratio']:
            failure_types.append(QualityFailureType.NO_TRANSCRIPTS)
            transcript_issues.append(f"Only {transcript_ratio:.1%} of videos have transcripts (minimum: {self.failure_thresholds['minimum_transcript_ratio']:.1%})")
        
        # Check for mixed quality (high variance) - check both direct scores and ranking metrics
        quality_scores = []
        for v in videos:
            if v.quality_score is not None and v.quality_score > 0:
                quality_scores.append(v.quality_score)
            elif hasattr(v, 'ranking_metrics') and v.ranking_metrics:
                quality_scores.append(v.ranking_metrics.combined_score)
        
        if len(quality_scores) > 2:
            quality_std = statistics.stdev(quality_scores)
            if quality_std > self.failure_thresholds['quality_variance_threshold']:
                failure_types.append(QualityFailureType.MIXED_QUALITY)
                content_issues.append(f"High quality variance ({quality_std:.1f}) indicates inconsistent results")
        
        # Determine refinement urgency
        refinement_urgency = self._determine_refinement_urgency(failure_types, videos_above_threshold)
        
        # Generate suggested adjustments
        suggested_adjustments = self._generate_suggested_adjustments(failure_types, score_stats, videos)
        
        # Analyze weekly focus impact
        weekly_focus_impact = self._analyze_weekly_focus_impact(videos, failure_types)
        
        logger.info(f"Quality analysis complete: {len(failure_types)} failure types, urgency: {refinement_urgency}")
        
        return QualityAnalysis(
            failure_types=failure_types,
            content_issues=content_issues,
            engagement_issues=engagement_issues,
            freshness_issues=freshness_issues,
            transcript_issues=transcript_issues,
            score_statistics=score_stats,
            refinement_urgency=refinement_urgency,
            suggested_adjustments=suggested_adjustments,
            weekly_focus_impact=weekly_focus_impact
        )
    
    def _calculate_score_statistics(self, videos: List[VideoData], 
                                  ranking_result: RankingResult) -> Dict[str, Any]:
        """Calculate comprehensive score statistics."""
        if not videos:
            return {
                'avg_quality': 0.0,
                'min_quality': 0.0,
                'max_quality': 0.0,
                'quality_std': 0.0,
                'videos_above_threshold': 0,
                'threshold_percentage': 0.0
            }
        
        # Get quality scores - only use actual scores, not None or 0
        quality_scores = [v.quality_score for v in videos if v.quality_score is not None and v.quality_score > 0]
        if not quality_scores:
            # If no quality scores available, use 0 as fallback
            quality_scores = [0.0]
        
        # Calculate statistics - use ranking result if individual scores not available
        if quality_scores and quality_scores != [0.0]:
            avg_quality = sum(quality_scores) / len(quality_scores)
            min_quality = min(quality_scores)
            max_quality = max(quality_scores)
            quality_std = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0
        else:
            # Use ranking result statistics if individual scores not available
            avg_quality = ranking_result.quality_summary.get('average_score', 0.0)
            min_quality = ranking_result.quality_summary.get('min_score', 0.0)
            max_quality = ranking_result.quality_summary.get('max_score', 0.0)
            quality_std = 0.0  # Can't calculate std dev without individual scores
        
        # Threshold analysis
        videos_above_threshold = ranking_result.threshold_analysis['videos_above_threshold']
        threshold_percentage = ranking_result.threshold_analysis['threshold_percentage']
        
        return {
            'avg_quality': avg_quality,
            'min_quality': min_quality,
            'max_quality': max_quality,
            'quality_std': quality_std,
            'videos_above_threshold': videos_above_threshold,
            'threshold_percentage': threshold_percentage,
            'total_videos': len(videos),
            'quality_scores': quality_scores
        }
    
    def _determine_refinement_urgency(self, failure_types: List[QualityFailureType], 
                                    videos_above_threshold: int) -> str:
        """Determine the urgency level for refinement."""
        if videos_above_threshold == 0:
            return 'critical'
        
        # Check for critical failure types
        critical_failures = [QualityFailureType.INSUFFICIENT_COUNT]
        if any(ft in critical_failures for ft in failure_types):
            return 'high'
        
        # Check for high priority failures
        high_priority_failures = [QualityFailureType.LOW_AVERAGE_QUALITY]
        if any(ft in high_priority_failures for ft in failure_types):
            return 'medium' if videos_above_threshold >= 2 else 'high'
        
        # Check for medium priority failures
        medium_priority_failures = [
            QualityFailureType.POOR_CONTENT_QUALITY,
            QualityFailureType.LOW_ENGAGEMENT,
            QualityFailureType.STALE_CONTENT
        ]
        if any(ft in medium_priority_failures for ft in failure_types):
            return 'medium'
        
        # Low priority or no significant failures
        return 'low'
    
    def _generate_suggested_adjustments(self, failure_types: List[QualityFailureType],
                                      score_stats: Dict[str, Any],
                                      videos: List[VideoData]) -> List[Dict[str, Any]]:
        """Generate specific adjustment suggestions based on failure analysis."""
        adjustments = []
        
        for failure_type in failure_types:
            if failure_type == QualityFailureType.INSUFFICIENT_COUNT:
                shortfall = self.min_quality_videos - score_stats['videos_above_threshold']
                adjustments.append({
                    'type': 'expand_search_scope',
                    'reason': 'insufficient_quality_videos',
                    'target_increase': shortfall,
                    'current_count': score_stats['videos_above_threshold'],
                    'required_count': self.min_quality_videos
                })
            
            elif failure_type == QualityFailureType.LOW_AVERAGE_QUALITY:
                quality_gap = self.quality_threshold - score_stats['avg_quality']
                adjustments.append({
                    'type': 'improve_search_terms',
                    'reason': 'low_average_quality',
                    'quality_gap': quality_gap,
                    'current_avg': score_stats['avg_quality'],
                    'target_avg': self.quality_threshold
                })
            
            elif failure_type == QualityFailureType.POOR_CONTENT_QUALITY:
                adjustments.append({
                    'type': 'target_authoritative_sources',
                    'reason': 'poor_content_quality',
                    'add_quality_indicators': True,
                    'focus_educational_content': True
                })
            
            elif failure_type == QualityFailureType.LOW_ENGAGEMENT:
                adjustments.append({
                    'type': 'target_popular_content',
                    'reason': 'low_engagement',
                    'increase_min_views': True,
                    'add_trending_indicators': True
                })
            
            elif failure_type == QualityFailureType.STALE_CONTENT:
                adjustments.append({
                    'type': 'enhance_freshness_focus',
                    'reason': 'stale_content',
                    'strengthen_weekly_focus': True,
                    'add_recency_terms': True
                })
            
            elif failure_type == QualityFailureType.NO_TRANSCRIPTS:
                adjustments.append({
                    'type': 'target_transcript_rich_content',
                    'reason': 'missing_transcripts',
                    'add_educational_terms': True,
                    'target_professional_channels': True
                })
            
            elif failure_type == QualityFailureType.MIXED_QUALITY:
                adjustments.append({
                    'type': 'refine_search_consistency',
                    'reason': 'mixed_quality',
                    'reduce_keyword_breadth': True,
                    'focus_specific_domains': True
                })
        
        return adjustments
    
    def _analyze_weekly_focus_impact(self, videos: List[VideoData], 
                                   failure_types: List[QualityFailureType]) -> Dict[str, Any]:
        """Analyze how refinement might impact weekly focus requirements."""
        if not videos:
            return {
                'weekly_content_ratio': 0.0,
                'can_maintain_focus': True,
                'extension_needed': False,
                'focus_preservation_strategy': 'maintain_timeframe'
            }
        
        # Calculate current weekly content ratio
        recent_videos = sum(1 for v in videos if v.is_recent(self.core_weekly_days))
        weekly_content_ratio = recent_videos / len(videos)
        
        # Determine if we can maintain weekly focus
        can_maintain_focus = weekly_content_ratio >= 0.6  # At least 60% recent content
        
        # Check if extension might be needed
        stale_content_issue = QualityFailureType.STALE_CONTENT in failure_types
        insufficient_count_issue = QualityFailureType.INSUFFICIENT_COUNT in failure_types
        
        extension_needed = stale_content_issue or (insufficient_count_issue and weekly_content_ratio < 0.4)
        
        # Determine focus preservation strategy
        if can_maintain_focus and not extension_needed:
            strategy = 'maintain_timeframe'
        elif weekly_content_ratio >= 0.4:
            strategy = 'selective_extension'  # Extend but prioritize recent content
        elif extension_needed:
            strategy = 'expand_timeframe'  # Need to expand timeframe
        else:
            strategy = 'gradual_extension'  # Gradually extend timeframe
        
        return {
            'weekly_content_ratio': weekly_content_ratio,
            'recent_video_count': recent_videos,
            'total_video_count': len(videos),
            'can_maintain_focus': can_maintain_focus,
            'extension_needed': extension_needed,
            'focus_preservation_strategy': strategy,
            'recommended_max_days': self.max_weekly_extension if extension_needed else self.core_weekly_days,
            'recommended_timeframe': self.max_weekly_extension if extension_needed else self.core_weekly_days
        }
    
    def should_trigger_refinement(self, videos: List[VideoData], 
                                ranking_result: RankingResult,
                                state: CuratorState) -> Tuple[bool, QualityAnalysis]:
        """
        Determine if quality-based refinement should be triggered.
        
        Args:
            videos: Current video set
            ranking_result: Results from video ranking
            state: Current curator state
            
        Returns:
            Tuple of (should_refine, quality_analysis)
        """
        # Check if we've reached maximum attempts
        if not state.can_refine_search():
            logger.info("Maximum search attempts reached, skipping quality-based refinement")
            return False, QualityAnalysis(
                failure_types=[],
                content_issues=[],
                engagement_issues=[],
                freshness_issues=[],
                transcript_issues=[],
                score_statistics={},
                refinement_urgency='low',
                suggested_adjustments=[],
                weekly_focus_impact={}
            )
        
        # Perform quality analysis
        quality_analysis = self.analyze_quality_failures(videos, ranking_result)
        
        # Check if quality requirements are already met
        if ranking_result.threshold_analysis['meets_minimum_requirement']:
            logger.info("Quality requirements already met, no refinement needed")
            return False, quality_analysis
        
        # Determine if refinement should be triggered based on urgency
        should_refine = quality_analysis.refinement_urgency in ['critical', 'high', 'medium']
        
        logger.info(f"Quality-based refinement decision: {should_refine} (urgency: {quality_analysis.refinement_urgency})")
        
        return should_refine, quality_analysis
    
    def generate_quality_based_refinement_actions(self, quality_analysis: QualityAnalysis,
                                                state: CuratorState) -> List[QualityRefinementAction]:
        """
        Generate specific refinement actions based on quality analysis.
        
        Args:
            quality_analysis: Results of quality failure analysis
            state: Current curator state
            
        Returns:
            List of prioritized refinement actions
        """
        actions = []
        
        # Generate actions based on failure types and suggested adjustments
        for adjustment in quality_analysis.suggested_adjustments:
            action_type = adjustment['type']
            reason = adjustment['reason']
            
            if action_type == 'expand_search_scope':
                actions.append(QualityRefinementAction(
                    action_type='expand_keywords',
                    description=f"Expand search keywords to find more quality videos (need {adjustment['target_increase']} more)",
                    parameters={
                        'strategy': RefinementStrategy.EXPAND_KEYWORDS,
                        'target_increase': adjustment['target_increase']
                    },
                    priority='high',
                    expected_improvement=f"Increase quality video count by {adjustment['target_increase']}",
                    weekly_focus_preservation={'maintain_timeframe': True}
                ))
            
            elif action_type == 'improve_search_terms':
                actions.append(QualityRefinementAction(
                    action_type='refine_keywords',
                    description=f"Refine search terms to improve average quality (gap: {adjustment['quality_gap']:.1f}%)",
                    parameters={
                        'add_quality_indicators': ['research', 'analysis', 'expert', 'professional'],
                        'add_authority_terms': ['university', 'institute', 'official', 'verified'],
                        'quality_gap': adjustment['quality_gap']
                    },
                    priority='high',
                    expected_improvement=f"Improve average quality by {adjustment['quality_gap']:.1f}%",
                    weekly_focus_preservation={'maintain_timeframe': True}
                ))
            
            elif action_type == 'target_authoritative_sources':
                actions.append(QualityRefinementAction(
                    action_type='refine_keywords',
                    description="Target authoritative sources to improve content quality",
                    parameters={
                        'add_quality_indicators': ['research', 'study', 'analysis', 'report'],
                        'add_authority_terms': ['expert', 'professor', 'researcher', 'official'],
                        'focus_educational': True
                    },
                    priority='medium',
                    expected_improvement="Improve content quality scores",
                    weekly_focus_preservation={'maintain_timeframe': True}
                ))
            
            elif action_type == 'target_popular_content':
                actions.append(QualityRefinementAction(
                    action_type='enhance_engagement',
                    description="Target more popular content to improve engagement metrics",
                    parameters={
                        'increase_min_views': True,
                        'add_trending_terms': ['trending', 'popular', 'viral', 'hot'],
                        'focus_engagement': True
                    },
                    priority='medium',
                    expected_improvement="Improve engagement scores",
                    weekly_focus_preservation={'maintain_timeframe': True}
                ))
            
            elif action_type == 'enhance_freshness_focus':
                actions.append(QualityRefinementAction(
                    action_type='enhance_freshness',
                    description="Enhance focus on fresh, recent content",
                    parameters={
                        'strengthen_weekly_focus': True,
                        'add_recency_terms': ['latest', 'new', 'recent', 'breaking', 'update'],
                        'core_window_days': self.core_weekly_days
                    },
                    priority='high',
                    expected_improvement="Improve content freshness scores",
                    weekly_focus_preservation={'strengthen_weekly_focus': True}
                ))
            
            elif action_type == 'target_transcript_rich_content':
                actions.append(QualityRefinementAction(
                    action_type='increase_transcripts',
                    description="Target content more likely to have transcripts",
                    parameters={
                        'educational_keywords': ['tutorial', 'course', 'lecture', 'lesson'],
                        'professional_terms': ['conference', 'presentation', 'webinar', 'seminar'],
                        'target_channels': ['educational', 'professional', 'corporate']
                    },
                    priority='medium',
                    expected_improvement="Increase transcript availability",
                    weekly_focus_preservation={'maintain_timeframe': True}
                ))
            
            elif action_type == 'refine_search_consistency':
                actions.append(QualityRefinementAction(
                    action_type='refine_keywords',
                    description="Refine search for more consistent quality results",
                    parameters={
                        'reduce_keyword_breadth': True,
                        'focus_specific_domains': True,
                        'remove_generic_terms': True
                    },
                    priority='low',
                    expected_improvement="Reduce quality variance",
                    weekly_focus_preservation={'maintain_timeframe': True}
                ))
        
        # Sort actions by priority (high -> medium -> low)
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        actions.sort(key=lambda x: priority_order.get(x.priority, 0), reverse=True)
        
        # Limit to top 3 actions to avoid over-refinement
        return actions[:3]
    
    def execute_quality_based_refinement(self, state: CuratorState, 
                                       quality_analysis: QualityAnalysis) -> CuratorState:
        """
        Execute quality-based refinement with targeted actions.
        
        Args:
            state: Current curator state
            quality_analysis: Results of quality failure analysis
            
        Returns:
            Updated curator state with refined search results
        """
        logger.info(f"Executing quality-based refinement (urgency: {quality_analysis.refinement_urgency})")
        
        # Generate refinement actions
        actions = self.generate_quality_based_refinement_actions(quality_analysis, state)
        
        if not actions:
            logger.warning("No refinement actions generated")
            state.add_error("No quality-based refinement actions could be generated", "quality_refinement")
            return state
        
        # Execute the primary action (highest priority)
        primary_action = actions[0]
        logger.info(f"Executing primary refinement action: {primary_action.action_type}")
        
        try:
            # Execute the appropriate refinement action
            if primary_action.action_type == 'expand_keywords':
                updated_state = self._execute_keyword_expansion(state, primary_action, quality_analysis)
            elif primary_action.action_type == 'refine_keywords':
                updated_state = self._execute_keyword_refinement(state, primary_action, quality_analysis)
            elif primary_action.action_type == 'enhance_freshness':
                updated_state = self._execute_freshness_enhancement(state, primary_action, quality_analysis)
            elif primary_action.action_type == 'increase_transcripts':
                updated_state = self._execute_transcript_targeting(state, primary_action, quality_analysis)
            elif primary_action.action_type == 'enhance_engagement':
                updated_state = self._execute_engagement_enhancement(state, primary_action, quality_analysis)
            else:
                logger.warning(f"Unknown action type: {primary_action.action_type}")
                updated_state = state
            
            # Update metadata
            updated_state.update_generation_metadata(
                quality_based_refinement=True,
                quality_analysis_summary={
                    'failure_types': [ft.value for ft in quality_analysis.failure_types],
                    'refinement_urgency': quality_analysis.refinement_urgency,
                    'primary_action': primary_action.action_type,
                    'expected_improvement': primary_action.expected_improvement,
                    'weekly_focus_maintained': quality_analysis.weekly_focus_impact.get('can_maintain_focus', True)
                },
                weekly_focus_preserved=primary_action.weekly_focus_preservation
            )
            
            logger.info(f"Quality-based refinement complete: {len(updated_state.discovered_videos)} videos found")
            return updated_state
            
        except Exception as e:
            error_msg = f"Error executing quality-based refinement: {e}"
            logger.error(error_msg)
            state.add_error(error_msg, "quality_refinement")
            return state
    
    def _execute_keyword_expansion(self, state: CuratorState, 
                                 action: QualityRefinementAction,
                                 quality_analysis: QualityAnalysis) -> CuratorState:
        """Execute keyword expansion refinement."""
        logger.info("Executing keyword expansion refinement")
        
        # Use the search refinement engine to expand keywords
        strategy = action.parameters.get('strategy', RefinementStrategy.EXPAND_KEYWORDS)
        refinement_result = self.search_refinement.perform_search_attempt(state, strategy)
        
        # Update state with results
        state.discovered_videos = refinement_result.videos
        state.current_search_terms = refinement_result.search_terms
        state.search_attempt += 1
        
        return state
    
    def _execute_keyword_refinement(self, state: CuratorState,
                                  action: QualityRefinementAction,
                                  quality_analysis: QualityAnalysis) -> CuratorState:
        """Execute keyword refinement for quality improvement."""
        logger.info("Executing keyword refinement for quality")
        
        # Build refined search terms
        current_terms = state.current_search_terms or state.search_keywords
        refined_terms = current_terms.copy()
        
        # Add quality indicators
        if 'add_quality_indicators' in action.parameters:
            refined_terms.extend(action.parameters['add_quality_indicators'])
        
        # Add authority terms
        if 'add_authority_terms' in action.parameters:
            refined_terms.extend(action.parameters['add_authority_terms'])
        
        # Remove duplicates while preserving order
        seen = set()
        refined_terms = [term for term in refined_terms if not (term in seen or seen.add(term))]
        
        # Perform search with refined terms
        discovered_videos = self.youtube_client.discover_videos(
            keywords=refined_terms,
            max_videos=state.max_videos * 2,  # Search for more to improve filtering
            days_back=state.days_back
        )
        
        # Update state
        state.discovered_videos = discovered_videos
        state.current_search_terms = refined_terms
        state.search_attempt += 1
        
        logger.info(f"Keyword refinement complete: {len(discovered_videos)} videos found with refined terms")
        return state
    
    def _execute_freshness_enhancement(self, state: CuratorState,
                                     action: QualityRefinementAction,
                                     quality_analysis: QualityAnalysis) -> CuratorState:
        """Execute freshness enhancement refinement."""
        logger.info("Executing freshness enhancement refinement")
        
        # Build search terms with freshness focus
        current_terms = state.current_search_terms or state.search_keywords
        enhanced_terms = current_terms.copy()
        
        # Add recency terms
        if 'add_recency_terms' in action.parameters:
            enhanced_terms.extend(action.parameters['add_recency_terms'])
        
        # Strengthen weekly focus by using core window
        days_back = action.parameters.get('core_window_days', self.core_weekly_days)
        
        # Perform search with enhanced freshness focus
        discovered_videos = self.youtube_client.discover_videos(
            keywords=enhanced_terms,
            max_videos=state.max_videos * 2,
            days_back=days_back
        )
        
        # Update state
        state.discovered_videos = discovered_videos
        state.current_search_terms = enhanced_terms
        state.days_back = days_back  # Update to maintain weekly focus
        state.search_attempt += 1
        
        logger.info(f"Freshness enhancement complete: {len(discovered_videos)} videos found with {days_back}-day focus")
        return state
    
    def _execute_transcript_targeting(self, state: CuratorState,
                                    action: QualityRefinementAction,
                                    quality_analysis: QualityAnalysis) -> CuratorState:
        """Execute transcript targeting refinement."""
        logger.info("Executing transcript targeting refinement")
        
        # Build search terms targeting educational/professional content
        current_terms = state.current_search_terms or state.search_keywords
        targeted_terms = current_terms.copy()
        
        # Add educational keywords
        if 'educational_keywords' in action.parameters:
            targeted_terms.extend(action.parameters['educational_keywords'])
        
        # Add professional terms
        if 'professional_terms' in action.parameters:
            targeted_terms.extend(action.parameters['professional_terms'])
        
        # Perform search
        discovered_videos = self.youtube_client.discover_videos(
            keywords=targeted_terms,
            max_videos=state.max_videos * 2,
            days_back=state.days_back
        )
        
        # Update state
        state.discovered_videos = discovered_videos
        state.current_search_terms = targeted_terms
        state.search_attempt += 1
        
        logger.info(f"Transcript targeting complete: {len(discovered_videos)} videos found")
        return state
    
    def _execute_engagement_enhancement(self, state: CuratorState,
                                      action: QualityRefinementAction,
                                      quality_analysis: QualityAnalysis) -> CuratorState:
        """Execute engagement enhancement refinement."""
        logger.info("Executing engagement enhancement refinement")
        
        # Build search terms with engagement focus
        current_terms = state.current_search_terms or state.search_keywords
        enhanced_terms = current_terms.copy()
        
        # Add trending terms
        if 'add_trending_terms' in action.parameters:
            enhanced_terms.extend(action.parameters['add_trending_terms'])
        
        # Perform search (min_views filtering is handled internally by the client)
        discovered_videos = self.youtube_client.discover_videos(
            keywords=enhanced_terms,
            max_videos=state.max_videos * 2,
            days_back=state.days_back
        )
        
        # Update state
        state.discovered_videos = discovered_videos
        state.current_search_terms = enhanced_terms
        state.search_attempt += 1
        
        logger.info(f"Engagement enhancement complete: {len(discovered_videos)} videos found with min {min_views} views")
        return state


def quality_based_refinement_node(state: CuratorState) -> CuratorState:
    """
    LangGraph node function for quality-based search refinement.
    
    This function analyzes video quality and triggers refinement when
    quality thresholds are not met, while maintaining weekly focus.
    
    Args:
        state: Current curator state
        
    Returns:
        Updated state with quality-based refinement results
    """
    logger.info("Starting quality-based refinement node")
    
    try:
        refinement_engine = QualityBasedRefinementEngine()
        
        # Rank current videos to assess quality
        ranking_result = refinement_engine.ranking_system.rank_videos(
            state.discovered_videos,
            target_count=state.max_videos
        )
        
        # Check if quality-based refinement should be triggered
        should_refine, quality_analysis = refinement_engine.should_trigger_refinement(
            state.discovered_videos, ranking_result, state
        )
        
        # Update metadata with analysis results
        state.update_generation_metadata(
            quality_analysis_performed=True,
            quality_threshold_met=ranking_result.threshold_analysis['meets_minimum_requirement']
        )
        
        if should_refine:
            logger.info("Quality-based refinement triggered")
            updated_state = refinement_engine.execute_quality_based_refinement(state, quality_analysis)
            
            # Re-rank videos after refinement
            final_ranking_result = refinement_engine.ranking_system.rank_videos(
                updated_state.discovered_videos,
                target_count=updated_state.max_videos
            )
            
            # Update state with final ranking
            updated_state.ranked_videos = final_ranking_result.ranked_videos
            updated_state.update_generation_metadata(
                final_quality_threshold_met=final_ranking_result.threshold_analysis['meets_minimum_requirement'],
                final_videos_above_threshold=final_ranking_result.threshold_analysis['videos_above_threshold']
            )
            
            return updated_state
        else:
            logger.info("Quality-based refinement not needed or not possible")
            state.update_generation_metadata(
                quality_based_refinement=False,
                refinement_skipped_reason='quality_thresholds_met' if ranking_result.threshold_analysis['meets_minimum_requirement'] else 'max_attempts_reached'
            )
            
            # Update ranked videos with current ranking
            state.ranked_videos = ranking_result.ranked_videos
            return state
            
    except Exception as e:
        error_msg = f"Critical error in quality-based refinement node: {e}"
        logger.error(error_msg)
        state.add_error(error_msg, "quality_based_refinement_node")
        return state