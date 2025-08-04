"""
Tests for the quality-based search refinement system.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from src.nanook_curator.quality_based_refinement import (
    QualityBasedRefinementEngine, QualityFailureType, QualityAnalysis, 
    QualityRefinementAction, quality_based_refinement_node
)
from src.nanook_curator.models import VideoData, CuratorState
from src.nanook_curator.config import Configuration
from src.nanook_curator.video_ranking_system import RankingResult, RankingMetrics
from src.nanook_curator.search_refinement import RefinementStrategy


def create_mock_config():
    """Create a mock configuration for testing."""
    mock_config = Mock(spec=Configuration)
    mock_config.default_search_keywords = ["AI news", "AI tools", "artificial intelligence"]
    mock_config.youtube_api_key = "test_api_key"
    mock_config.max_videos = 10
    mock_config.days_back = 7
    mock_config.quality_threshold = 70.0
    mock_config.min_quality_videos = 3
    return mock_config


def create_test_video(**kwargs):
    """Create a test video with default values."""
    defaults = {
        'video_id': 'test1234567',
        'title': 'Test Video',
        'channel': 'Test Channel',
        'view_count': 10000,
        'like_count': 800,
        'comment_count': 50,
        'upload_date': '2024-01-15T10:00:00Z',
        'transcript': 'This is a test transcript for analyzing content quality.'
    }
    defaults.update(kwargs)
    
    # Ensure video_id is exactly 11 characters
    if 'video_id' in kwargs:
        video_id = kwargs['video_id']
        if len(video_id) != 11:
            # Pad or truncate to 11 characters
            if len(video_id) < 11:
                video_id = video_id + '0' * (11 - len(video_id))
            else:
                video_id = video_id[:11]
            defaults['video_id'] = video_id
    
    return VideoData(**defaults)


def create_test_curator_state(**kwargs):
    """Create a test curator state with default values."""
    defaults = {
        'search_keywords': ['artificial intelligence', 'AI news'],
        'max_videos': 10,
        'days_back': 7,
        'search_attempt': 0,
        'max_search_attempts': 3,
        'current_search_terms': [],
        'quality_threshold': 70.0,
        'min_quality_videos': 3
    }
    defaults.update(kwargs)
    return CuratorState(**defaults)


def create_mock_ranking_result(videos, meets_requirement=True, avg_score=75.0):
    """Create a mock ranking result."""
    videos_above_threshold = len(videos) if meets_requirement else 0
    return RankingResult(
        ranked_videos=videos,
        quality_summary={
            'total_videos': len(videos),
            'average_score': avg_score,
            'min_score': min(60.0, avg_score - 10),
            'max_score': min(90.0, avg_score + 10)
        },
        threshold_analysis={
            'meets_minimum_requirement': meets_requirement,
            'videos_above_threshold': videos_above_threshold,
            'shortfall': max(0, 3 - videos_above_threshold),
            'threshold_percentage': (videos_above_threshold / len(videos)) * 100 if videos else 0,
            'threshold_used': 70.0
        },
        refinement_suggestions=['Test suggestion'] if not meets_requirement else [],
        ranking_metadata={'strategy': 'balanced'}
    )


def create_mock_ranking_metrics(content_score=75.0, engagement_score=75.0, freshness_score=75.0):
    """Create mock ranking metrics."""
    combined_score = (content_score + engagement_score + freshness_score) / 3
    return RankingMetrics(
        content_score=content_score,
        engagement_score=engagement_score,
        freshness_score=freshness_score,
        combined_score=combined_score,
        quality_band=None,  # Will be set by ranking system
        rank_position=1,
        meets_threshold=combined_score >= 70.0,
        detailed_metrics={}
    )


class TestQualityBasedRefinementEngine:
    """Test cases for QualityBasedRefinementEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        mock_config = create_mock_config()
        
        # Mock the dependencies to avoid initialization issues
        with patch('src.nanook_curator.quality_based_refinement.VideoRankingSystem'), \
             patch('src.nanook_curator.quality_based_refinement.SearchRefinementEngine'), \
             patch('src.nanook_curator.quality_based_refinement.YouTubeClient'):
            self.refinement_engine = QualityBasedRefinementEngine(config=mock_config)
    
    def test_init(self):
        """Test refinement engine initialization."""
        assert self.refinement_engine.config is not None
        assert self.refinement_engine.quality_threshold == 70.0
        assert self.refinement_engine.min_quality_videos == 3
        assert self.refinement_engine.core_weekly_days == 7
        assert self.refinement_engine.max_weekly_extension == 14
        assert len(self.refinement_engine.failure_thresholds) > 0
    
    def test_analyze_quality_failures_insufficient_count(self):
        """Test quality failure analysis for insufficient video count."""
        # Create videos but insufficient above threshold
        videos = [create_test_video(video_id=f'insuff{i:06d}') for i in range(5)]
        ranking_result = create_mock_ranking_result(videos[:1], meets_requirement=False, avg_score=65.0)
        
        analysis = self.refinement_engine.analyze_quality_failures(videos, ranking_result)
        
        assert QualityFailureType.INSUFFICIENT_COUNT in analysis.failure_types
        assert analysis.refinement_urgency in ['critical', 'high']
        assert len(analysis.suggested_adjustments) > 0
        assert any('insufficient_quality_videos' in adj['reason'] for adj in analysis.suggested_adjustments)
    
    def test_analyze_quality_failures_low_average_quality(self):
        """Test quality failure analysis for low average quality."""
        videos = [create_test_video(video_id=f'lowavg{i:06d}') for i in range(4)]
        ranking_result = create_mock_ranking_result(videos, meets_requirement=False, avg_score=50.0)
        
        analysis = self.refinement_engine.analyze_quality_failures(videos, ranking_result)
        
        assert QualityFailureType.LOW_AVERAGE_QUALITY in analysis.failure_types
        assert analysis.refinement_urgency in ['high', 'critical']
        assert any('low_average_quality' in adj['reason'] for adj in analysis.suggested_adjustments)
    
    def test_analyze_quality_failures_poor_content_quality(self):
        """Test quality failure analysis for poor content quality."""
        videos = [create_test_video(video_id=f'poorcont{i:04d}') for i in range(3)]
        
        # Add mock ranking metrics with low content scores
        for i, video in enumerate(videos):
            video.ranking_metrics = create_mock_ranking_metrics(
                content_score=30.0,  # Below minimum_content_score (50.0)
                engagement_score=70.0,
                freshness_score=70.0
            )
        
        ranking_result = create_mock_ranking_result(videos, meets_requirement=False, avg_score=60.0)
        
        analysis = self.refinement_engine.analyze_quality_failures(videos, ranking_result)
        
        assert QualityFailureType.POOR_CONTENT_QUALITY in analysis.failure_types
        assert any('poor_content_quality' in adj['reason'] for adj in analysis.suggested_adjustments)
    
    def test_analyze_quality_failures_low_engagement(self):
        """Test quality failure analysis for low engagement."""
        videos = [create_test_video(video_id=f'loweng{i:05d}') for i in range(3)]
        
        # Add mock ranking metrics with low engagement scores
        for video in videos:
            video.ranking_metrics = create_mock_ranking_metrics(
                content_score=70.0,
                engagement_score=25.0,  # Below minimum_engagement_score (40.0)
                freshness_score=70.0
            )
        
        ranking_result = create_mock_ranking_result(videos, meets_requirement=False, avg_score=60.0)
        
        analysis = self.refinement_engine.analyze_quality_failures(videos, ranking_result)
        
        assert QualityFailureType.LOW_ENGAGEMENT in analysis.failure_types
        assert any('low_engagement' in adj['reason'] for adj in analysis.suggested_adjustments)
    
    def test_analyze_quality_failures_stale_content(self):
        """Test quality failure analysis for stale content."""
        videos = [create_test_video(video_id=f'stale{i:06d}') for i in range(3)]
        
        # Add mock ranking metrics with low freshness scores
        for video in videos:
            video.ranking_metrics = create_mock_ranking_metrics(
                content_score=70.0,
                engagement_score=70.0,
                freshness_score=15.0  # Below minimum_freshness_score (30.0)
            )
        
        ranking_result = create_mock_ranking_result(videos, meets_requirement=False, avg_score=60.0)
        
        analysis = self.refinement_engine.analyze_quality_failures(videos, ranking_result)
        
        assert QualityFailureType.STALE_CONTENT in analysis.failure_types
        assert any('stale_content' in adj['reason'] for adj in analysis.suggested_adjustments)
    
    def test_analyze_quality_failures_no_transcripts(self):
        """Test quality failure analysis for lack of transcripts."""
        videos = [create_test_video(video_id=f'notrans{i:05d}', transcript=None) for i in range(5)]
        ranking_result = create_mock_ranking_result(videos, meets_requirement=False, avg_score=60.0)
        
        analysis = self.refinement_engine.analyze_quality_failures(videos, ranking_result)
        
        assert QualityFailureType.NO_TRANSCRIPTS in analysis.failure_types
        assert any('missing_transcripts' in adj['reason'] for adj in analysis.suggested_adjustments)
    
    def test_analyze_quality_failures_mixed_quality(self):
        """Test quality failure analysis for mixed quality (high variance)."""
        videos = [create_test_video(video_id=f'mixed{i:06d}') for i in range(5)]
        
        # Add mock ranking metrics with high variance
        scores = [20.0, 90.0, 30.0, 85.0, 25.0]  # High variance
        for i, video in enumerate(videos):
            video.ranking_metrics = create_mock_ranking_metrics(
                content_score=scores[i],
                engagement_score=scores[i],
                freshness_score=scores[i]
            )
        
        ranking_result = create_mock_ranking_result(videos, meets_requirement=False, avg_score=50.0)
        
        analysis = self.refinement_engine.analyze_quality_failures(videos, ranking_result)
        
        assert QualityFailureType.MIXED_QUALITY in analysis.failure_types
        assert any('mixed_quality' in adj['reason'] for adj in analysis.suggested_adjustments)
    
    def test_analyze_quality_failures_no_failures(self):
        """Test quality failure analysis when no failures are detected."""
        videos = [create_test_video(video_id=f'good{i:07d}') for i in range(4)]
        
        # Add good ranking metrics
        for video in videos:
            video.ranking_metrics = create_mock_ranking_metrics(
                content_score=80.0,
                engagement_score=75.0,
                freshness_score=70.0
            )
        
        ranking_result = create_mock_ranking_result(videos, meets_requirement=True, avg_score=80.0)
        
        analysis = self.refinement_engine.analyze_quality_failures(videos, ranking_result)
        
        assert len(analysis.failure_types) == 0
        assert analysis.refinement_urgency == 'low'
        assert len(analysis.suggested_adjustments) == 0
    
    def test_determine_refinement_urgency(self):
        """Test refinement urgency determination."""
        # Critical urgency - no videos above threshold
        urgency = self.refinement_engine._determine_refinement_urgency(
            [QualityFailureType.INSUFFICIENT_COUNT], 0
        )
        assert urgency == 'critical'
        
        # High urgency - critical failure types
        urgency = self.refinement_engine._determine_refinement_urgency(
            [QualityFailureType.INSUFFICIENT_COUNT], 1
        )
        assert urgency == 'high'
        
        # Medium urgency - high priority failures
        urgency = self.refinement_engine._determine_refinement_urgency(
            [QualityFailureType.POOR_CONTENT_QUALITY], 2
        )
        assert urgency == 'medium'
        
        # Low urgency - minor failures
        urgency = self.refinement_engine._determine_refinement_urgency(
            [QualityFailureType.MIXED_QUALITY], 3
        )
        assert urgency == 'low'
    
    def test_calculate_score_statistics(self):
        """Test score statistics calculation."""
        videos = [
            create_test_video(video_id='stats123456'),
            create_test_video(video_id='stats234567'),
            create_test_video(video_id='stats345678')
        ]
        
        # Add quality scores
        for i, video in enumerate(videos):
            video.quality_score = 70.0 + i * 10  # Scores: 70, 80, 90
        
        ranking_result = create_mock_ranking_result(videos, meets_requirement=True, avg_score=80.0)
        
        stats = self.refinement_engine._calculate_score_statistics(videos, ranking_result)
        
        assert stats['avg_quality'] == 80.0
        assert stats['min_quality'] == 70.0
        assert stats['max_quality'] == 90.0
        assert stats['quality_std'] > 0  # Should have some standard deviation
        assert 'videos_above_threshold' in stats
        assert 'threshold_percentage' in stats
    
    def test_analyze_weekly_focus_impact(self):
        """Test weekly focus impact analysis."""
        # Create videos with different upload dates
        recent_date = (datetime.now() - timedelta(days=2)).isoformat() + 'Z'
        old_date = (datetime.now() - timedelta(days=10)).isoformat() + 'Z'
        
        videos = [
            create_test_video(video_id='recent12345', upload_date=recent_date),
            create_test_video(video_id='recent23456', upload_date=recent_date),
            create_test_video(video_id='old1234567', upload_date=old_date)
        ]
        
        failure_types = [QualityFailureType.INSUFFICIENT_COUNT]
        
        impact = self.refinement_engine._analyze_weekly_focus_impact(videos, failure_types)
        
        assert 'weekly_content_ratio' in impact
        assert 'can_maintain_focus' in impact
        assert 'extension_needed' in impact
        assert 'focus_preservation_strategy' in impact
        assert impact['weekly_content_ratio'] > 0.5  # Most videos should be recent
    
    def test_generate_quality_based_refinement_actions(self):
        """Test refinement action generation."""
        # Create quality analysis with multiple failure types
        quality_analysis = QualityAnalysis(
            failure_types=[QualityFailureType.INSUFFICIENT_COUNT, QualityFailureType.LOW_AVERAGE_QUALITY],
            content_issues=['Not enough quality videos'],
            engagement_issues=[],
            freshness_issues=[],
            transcript_issues=[],
            score_statistics={'avg_quality': 60.0},
            refinement_urgency='high',
            suggested_adjustments=[
                {'type': 'expand_search_scope', 'reason': 'insufficient_quality_videos', 'target_increase': 2},
                {'type': 'improve_search_terms', 'reason': 'low_average_quality', 'quality_gap': 10.0}
            ],
            weekly_focus_impact={'can_maintain_focus': True}
        )
        
        state = create_test_curator_state()
        actions = self.refinement_engine.generate_quality_based_refinement_actions(quality_analysis, state)
        
        assert len(actions) > 0
        assert all(isinstance(action, QualityRefinementAction) for action in actions)
        assert any(action.action_type == 'expand_keywords' for action in actions)
        assert any(action.action_type == 'refine_keywords' for action in actions)
        
        # Check that actions are sorted by priority
        priorities = [action.priority for action in actions]
        priority_values = {'high': 3, 'medium': 2, 'low': 1}
        priority_scores = [priority_values.get(p, 0) for p in priorities]
        assert priority_scores == sorted(priority_scores, reverse=True)
    
    def test_should_trigger_refinement_critical_urgency(self):
        """Test refinement trigger for critical urgency."""
        videos = [create_test_video(video_id='critical123')]
        ranking_result = create_mock_ranking_result([], meets_requirement=False, avg_score=30.0)
        state = create_test_curator_state()
        
        should_refine, analysis = self.refinement_engine.should_trigger_refinement(videos, ranking_result, state)
        
        assert should_refine == True
        assert analysis.refinement_urgency in ['critical', 'high']
    
    def test_should_trigger_refinement_max_attempts(self):
        """Test refinement trigger when max attempts reached."""
        videos = [create_test_video(video_id='maxattempt1')]
        ranking_result = create_mock_ranking_result([], meets_requirement=False, avg_score=30.0)
        state = create_test_curator_state(search_attempt=3, max_search_attempts=3)  # At max attempts
        
        should_refine, analysis = self.refinement_engine.should_trigger_refinement(videos, ranking_result, state)
        
        assert should_refine == False  # Should not refine when at max attempts
    
    def test_should_trigger_refinement_quality_met(self):
        """Test refinement trigger when quality requirements are met."""
        videos = [create_test_video(video_id=f'qualmet{i:04d}') for i in range(4)]
        ranking_result = create_mock_ranking_result(videos, meets_requirement=True, avg_score=80.0)
        state = create_test_curator_state()
        
        should_refine, analysis = self.refinement_engine.should_trigger_refinement(videos, ranking_result, state)
        
        assert should_refine == False  # Should not refine when quality is good
        assert analysis.refinement_urgency == 'low'
    
    def test_execute_quality_based_refinement(self):
        """Test quality-based refinement execution."""
        state = create_test_curator_state()
        
        # Create quality analysis that triggers refinement
        quality_analysis = QualityAnalysis(
            failure_types=[QualityFailureType.INSUFFICIENT_COUNT],
            content_issues=['Not enough quality videos'],
            engagement_issues=[],
            freshness_issues=[],
            transcript_issues=[],
            score_statistics={'avg_quality': 60.0},
            refinement_urgency='high',
            suggested_adjustments=[
                {'type': 'expand_search_scope', 'reason': 'insufficient_quality_videos', 'target_increase': 2}
            ],
            weekly_focus_impact={'can_maintain_focus': True}
        )
        
        # Mock the refinement execution
        mock_refinement_result = Mock()
        mock_refinement_result.videos = [create_test_video(video_id=f'refined{i:04d}') for i in range(5)]
        mock_refinement_result.search_terms = ['ai', 'artificial intelligence', 'machine learning']
        
        with patch.object(self.refinement_engine.search_refinement, 'perform_search_attempt', return_value=mock_refinement_result):
            
            updated_state = self.refinement_engine.execute_quality_based_refinement(state, quality_analysis)
            
            assert len(updated_state.discovered_videos) == 5
            assert updated_state.search_attempt == 1
            assert updated_state.generation_metadata['quality_based_refinement'] == True
            assert 'primary_action' in updated_state.generation_metadata['quality_analysis_summary']
    
    def test_execute_keyword_expansion(self):
        """Test keyword expansion refinement execution."""
        state = create_test_curator_state()
        
        action = QualityRefinementAction(
            action_type='expand_keywords',
            description='Expand search terms',
            parameters={'strategy': RefinementStrategy.EXPAND_KEYWORDS},
            priority='high',
            expected_improvement='Increase quality video count',
            weekly_focus_preservation={'maintain_timeframe': True}
        )
        
        quality_analysis = Mock()
        
        # Mock the search refinement
        mock_result = Mock()
        mock_result.videos = [create_test_video(video_id=f'expand{i:05d}') for i in range(3)]
        mock_result.search_terms = ['ai', 'artificial intelligence', 'machine learning']
        
        with patch.object(self.refinement_engine.search_refinement, 'perform_search_attempt', return_value=mock_result):
            
            updated_state = self.refinement_engine._execute_keyword_expansion(state, action, quality_analysis)
            
            assert len(updated_state.discovered_videos) == 3
            assert updated_state.current_search_terms == mock_result.search_terms
            assert updated_state.search_attempt == 1
    
    def test_execute_keyword_refinement(self):
        """Test keyword refinement execution."""
        state = create_test_curator_state()
        state.current_search_terms = ['ai']
        
        action = QualityRefinementAction(
            action_type='refine_keywords',
            description='Refine search terms for quality',
            parameters={
                'add_quality_indicators': ['research', 'analysis'],
                'add_authority_terms': ['expert', 'professional']
            },
            priority='high',
            expected_improvement='Improve quality',
            weekly_focus_preservation={'maintain_timeframe': True}
        )
        
        quality_analysis = Mock()
        
        # Mock the YouTube client
        mock_videos = [create_test_video(video_id=f'refine{i:05d}') for i in range(4)]
        
        with patch.object(self.refinement_engine.youtube_client, 'discover_videos', return_value=mock_videos):
            
            updated_state = self.refinement_engine._execute_keyword_refinement(state, action, quality_analysis)
            
            assert len(updated_state.discovered_videos) == 4
            assert len(updated_state.current_search_terms) > 1  # Should have added terms
            assert 'research' in updated_state.current_search_terms
            assert updated_state.search_attempt == 1
    
    def test_execute_freshness_enhancement(self):
        """Test freshness enhancement execution."""
        state = create_test_curator_state()
        
        action = QualityRefinementAction(
            action_type='enhance_freshness',
            description='Enhance content freshness',
            parameters={
                'strengthen_weekly_focus': True,
                'core_window_days': 7
            },
            priority='high',
            expected_improvement='Improve freshness',
            weekly_focus_preservation={'strengthen_weekly_focus': True}
        )
        
        quality_analysis = Mock()
        
        # Mock the YouTube client
        mock_videos = [create_test_video(video_id=f'fresh{i:06d}') for i in range(3)]
        
        with patch.object(self.refinement_engine.youtube_client, 'discover_videos', return_value=mock_videos):
            
            updated_state = self.refinement_engine._execute_freshness_enhancement(state, action, quality_analysis)
            
            assert len(updated_state.discovered_videos) == 3
            assert updated_state.days_back == 7  # Should maintain weekly focus
            assert updated_state.search_attempt == 1
    
    def test_execute_transcript_targeting(self):
        """Test transcript targeting execution."""
        state = create_test_curator_state()
        
        action = QualityRefinementAction(
            action_type='increase_transcripts',
            description='Target educational content',
            parameters={
                'educational_keywords': ['tutorial', 'course'],
                'professional_terms': ['conference', 'presentation']
            },
            priority='medium',
            expected_improvement='Increase transcript availability',
            weekly_focus_preservation={'maintain_timeframe': True}
        )
        
        quality_analysis = Mock()
        
        # Mock the YouTube client
        mock_videos = [create_test_video(video_id=f'transc{i:05d}') for i in range(6)]
        
        with patch.object(self.refinement_engine.youtube_client, 'discover_videos', return_value=mock_videos):
            
            updated_state = self.refinement_engine._execute_transcript_targeting(state, action, quality_analysis)
            
            assert len(updated_state.discovered_videos) == 6
            assert 'tutorial' in updated_state.current_search_terms
            assert 'conference' in updated_state.current_search_terms
            assert updated_state.search_attempt == 1
    
    def test_failure_type_enum(self):
        """Test QualityFailureType enum values."""
        assert QualityFailureType.INSUFFICIENT_COUNT.value == 'insufficient_count'
        assert QualityFailureType.LOW_AVERAGE_QUALITY.value == 'low_average_quality'
        assert QualityFailureType.POOR_CONTENT_QUALITY.value == 'poor_content_quality'
        assert QualityFailureType.LOW_ENGAGEMENT.value == 'low_engagement'
        assert QualityFailureType.STALE_CONTENT.value == 'stale_content'
        assert QualityFailureType.NO_TRANSCRIPTS.value == 'no_transcripts'
        assert QualityFailureType.MIXED_QUALITY.value == 'mixed_quality'
    
    def test_quality_analysis_dataclass(self):
        """Test QualityAnalysis dataclass functionality."""
        analysis = QualityAnalysis(
            failure_types=[QualityFailureType.INSUFFICIENT_COUNT],
            content_issues=['Test issue'],
            engagement_issues=[],
            freshness_issues=[],
            transcript_issues=[],
            score_statistics={'avg_quality': 60.0},
            refinement_urgency='high',
            suggested_adjustments=[{'type': 'test', 'reason': 'test'}],
            weekly_focus_impact={'can_maintain_focus': True}
        )
        
        assert len(analysis.failure_types) == 1
        assert analysis.failure_types[0] == QualityFailureType.INSUFFICIENT_COUNT
        assert analysis.refinement_urgency == 'high'
        assert len(analysis.suggested_adjustments) == 1
        assert analysis.weekly_focus_impact['can_maintain_focus'] == True
    
    def test_quality_refinement_action_dataclass(self):
        """Test QualityRefinementAction dataclass functionality."""
        action = QualityRefinementAction(
            action_type='expand_keywords',
            description='Test action',
            parameters={'test': 'value'},
            priority='high',
            expected_improvement='Test improvement',
            weekly_focus_preservation={'maintain_timeframe': True}
        )
        
        assert action.action_type == 'expand_keywords'
        assert action.description == 'Test action'
        assert action.parameters['test'] == 'value'
        assert action.priority == 'high'
        assert action.expected_improvement == 'Test improvement'
        assert action.weekly_focus_preservation['maintain_timeframe'] == True


def test_quality_based_refinement_node():
    """Test the LangGraph node function."""
    state = create_test_curator_state()
    state.discovered_videos = [create_test_video(video_id=f'node{i:07d}') for i in range(2)]
    
    # Mock the QualityBasedRefinementEngine
    mock_engine = Mock()
    mock_ranking_result = create_mock_ranking_result(state.discovered_videos, meets_requirement=False)
    mock_engine.ranking_system.rank_videos.return_value = mock_ranking_result
    mock_engine.should_trigger_refinement.return_value = (True, Mock())
    mock_engine.execute_quality_based_refinement.return_value = state
    
    final_ranking_result = create_mock_ranking_result(state.discovered_videos, meets_requirement=True)
    mock_engine.ranking_system.rank_videos.return_value = final_ranking_result
    
    with patch('src.nanook_curator.quality_based_refinement.QualityBasedRefinementEngine', return_value=mock_engine):
        result_state = quality_based_refinement_node(state)
        
        assert result_state == state
        mock_engine.should_trigger_refinement.assert_called_once()
        mock_engine.execute_quality_based_refinement.assert_called_once()


def test_quality_based_refinement_node_no_refinement_needed():
    """Test the node function when no refinement is needed."""
    state = create_test_curator_state()
    state.discovered_videos = [create_test_video(video_id=f'noref{i:06d}') for i in range(4)]
    
    # Mock the engine to return that no refinement is needed
    mock_engine = Mock()
    mock_ranking_result = create_mock_ranking_result(state.discovered_videos, meets_requirement=True)
    mock_engine.ranking_system.rank_videos.return_value = mock_ranking_result
    mock_engine.should_trigger_refinement.return_value = (False, Mock())
    
    with patch('src.nanook_curator.quality_based_refinement.QualityBasedRefinementEngine', return_value=mock_engine):
        result_state = quality_based_refinement_node(state)
        
        assert result_state.generation_metadata['quality_based_refinement'] == False
        assert result_state.generation_metadata['quality_analysis_performed'] == True
        assert result_state.generation_metadata['refinement_skipped_reason'] == 'quality_thresholds_met'
        mock_engine.execute_quality_based_refinement.assert_not_called()


def test_quality_based_refinement_node_error_handling():
    """Test error handling in the node function."""
    state = create_test_curator_state()
    
    # Mock the engine to raise an exception
    with patch('src.nanook_curator.quality_based_refinement.QualityBasedRefinementEngine', side_effect=Exception("Engine error")):
        result_state = quality_based_refinement_node(state)
        
        assert len(result_state.errors) > 0
        assert 'Critical error in quality-based refinement node' in result_state.errors[-1]


class TestWeeklyFocusPreservation:
    """Test weekly focus preservation during quality-based refinement."""
    
    def setup_method(self):
        """Set up test fixtures."""
        mock_config = create_mock_config()
        
        with patch('src.nanook_curator.quality_based_refinement.VideoRankingSystem'), \
             patch('src.nanook_curator.quality_based_refinement.SearchRefinementEngine'), \
             patch('src.nanook_curator.quality_based_refinement.YouTubeClient'):
            self.refinement_engine = QualityBasedRefinementEngine(config=mock_config)
    
    def test_weekly_focus_maintained_with_recent_content(self):
        """Test that weekly focus is maintained when most content is recent."""
        # Create mostly recent videos
        recent_date = (datetime.now() - timedelta(days=3)).isoformat() + 'Z'
        old_date = (datetime.now() - timedelta(days=15)).isoformat() + 'Z'
        
        videos = [
            create_test_video(video_id='recent12345', upload_date=recent_date),
            create_test_video(video_id='recent23456', upload_date=recent_date),
            create_test_video(video_id='recent34567', upload_date=recent_date),
            create_test_video(video_id='old1234567', upload_date=old_date)
        ]
        
        failure_types = [QualityFailureType.LOW_AVERAGE_QUALITY]
        impact = self.refinement_engine._analyze_weekly_focus_impact(videos, failure_types)
        
        assert impact['can_maintain_focus'] == True
        assert impact['weekly_content_ratio'] >= 0.5
        assert impact['focus_preservation_strategy'] in ['maintain_timeframe', 'prioritize_recent']
    
    def test_weekly_focus_extension_needed(self):
        """Test that extension is needed when most content is old."""
        old_date = (datetime.now() - timedelta(days=15)).isoformat() + 'Z'
        recent_date = (datetime.now() - timedelta(days=3)).isoformat() + 'Z'
        
        videos = [
            create_test_video(video_id='old12345678', upload_date=old_date),
            create_test_video(video_id='old23456789', upload_date=old_date),
            create_test_video(video_id='old34567890', upload_date=old_date),
            create_test_video(video_id='recent45678', upload_date=recent_date)
        ]
        
        failure_types = [QualityFailureType.INSUFFICIENT_COUNT]
        impact = self.refinement_engine._analyze_weekly_focus_impact(videos, failure_types)
        
        assert impact['can_maintain_focus'] == False
        assert impact['extension_needed'] == True
        assert impact['focus_preservation_strategy'] == 'expand_timeframe'
        assert impact['recommended_timeframe'] > 7
    
    def test_freshness_enhancement_preserves_weekly_focus(self):
        """Test that freshness enhancement maintains weekly focus."""
        state = create_test_curator_state()
        state.days_back = 14  # Currently extended
        
        action = QualityRefinementAction(
            action_type='enhance_freshness',
            description='Enhance freshness',
            parameters={
                'strengthen_weekly_focus': True,
                'core_window_days': 7
            },
            priority='high',
            expected_improvement='Better freshness',
            weekly_focus_preservation={'strengthen_weekly_focus': True}
        )
        
        mock_videos = [create_test_video(video_id=f'fresh{i:06d}') for i in range(3)]
        
        with patch.object(self.refinement_engine.youtube_client, 'discover_videos', return_value=mock_videos):
            updated_state = self.refinement_engine._execute_freshness_enhancement(state, action, Mock())
            
            # Should reduce back to core weekly window
            assert updated_state.days_back == 7
    
    def test_quality_actions_preserve_weekly_focus(self):
        """Test that all quality actions preserve weekly focus metadata."""
        quality_analysis = QualityAnalysis(
            failure_types=[QualityFailureType.INSUFFICIENT_COUNT],
            content_issues=['Not enough videos'],
            engagement_issues=[],
            freshness_issues=[],
            transcript_issues=[],
            score_statistics={'avg_quality': 60.0},
            refinement_urgency='high',
            suggested_adjustments=[
                {'type': 'expand_search_scope', 'reason': 'insufficient_quality_videos', 'target_increase': 2}
            ],
            weekly_focus_impact={'can_maintain_focus': True}
        )
        
        state = create_test_curator_state()
        actions = self.refinement_engine.generate_quality_based_refinement_actions(quality_analysis, state)
        
        # All actions should have weekly focus preservation metadata
        for action in actions:
            assert 'weekly_focus_preservation' in action.__dict__
            assert isinstance(action.weekly_focus_preservation, dict)
            assert len(action.weekly_focus_preservation) > 0


def test_integration_with_existing_systems():
    """Test integration with existing ranking and search systems."""
    mock_config = create_mock_config()
    
    with patch('src.nanook_curator.quality_based_refinement.VideoRankingSystem') as mock_ranking_class, \
         patch('src.nanook_curator.quality_based_refinement.SearchRefinementEngine') as mock_search_class, \
         patch('src.nanook_curator.quality_based_refinement.YouTubeClient') as mock_youtube_class:
        
        mock_ranking = Mock()
        mock_search = Mock()
        mock_youtube = Mock()
        
        mock_ranking_class.return_value = mock_ranking
        mock_search_class.return_value = mock_search
        mock_youtube_class.return_value = mock_youtube
        
        engine = QualityBasedRefinementEngine(config=mock_config)
        
        # Verify that dependencies were initialized with config
        mock_ranking_class.assert_called_once_with(config=mock_config)
        mock_search_class.assert_called_once_with(config=mock_config)
        mock_youtube_class.assert_called_once_with(config=mock_config)
        
        assert engine.ranking_system == mock_ranking
        assert engine.search_refinement == mock_search
        assert engine.youtube_client == mock_youtube


def test_quality_thresholds_configuration():
    """Test that quality thresholds are properly configured."""
    mock_config = create_mock_config()
    mock_config.quality_threshold = 75.0
    mock_config.min_quality_videos = 4
    
    with patch('src.nanook_curator.quality_based_refinement.VideoRankingSystem'), \
         patch('src.nanook_curator.quality_based_refinement.SearchRefinementEngine'), \
         patch('src.nanook_curator.quality_based_refinement.YouTubeClient'):
        
        engine = QualityBasedRefinementEngine(config=mock_config)
        
        assert engine.quality_threshold == 75.0
        assert engine.min_quality_videos == 4
        assert engine.acceptable_average_quality == 60.0  # 75 - 15
        
        # Check that failure thresholds are reasonable
        for failure_type, threshold in engine.failure_thresholds.items():
            assert 0.0 <= threshold <= 100.0
            assert isinstance(threshold, (int, float))


def test_comprehensive_quality_workflow():
    """Test the complete quality-based refinement workflow."""
    mock_config = create_mock_config()
    
    with patch('src.nanook_curator.quality_based_refinement.VideoRankingSystem') as mock_ranking_class, \
         patch('src.nanook_curator.quality_based_refinement.SearchRefinementEngine') as mock_search_class, \
         patch('src.nanook_curator.quality_based_refinement.YouTubeClient') as mock_youtube_class:
        
        # Setup mocks
        mock_ranking = Mock()
        mock_search = Mock()
        mock_youtube = Mock()
        
        mock_ranking_class.return_value = mock_ranking
        mock_search_class.return_value = mock_search
        mock_youtube_class.return_value = mock_youtube
        
        engine = QualityBasedRefinementEngine(config=mock_config)
        
        # Create initial low-quality videos
        initial_videos = [create_test_video(video_id=f'initial{i:04d}') for i in range(2)]
        initial_ranking = create_mock_ranking_result(initial_videos[:1], meets_requirement=False, avg_score=50.0)
        
        # Mock the ranking system
        mock_ranking.rank_videos.return_value = initial_ranking
        
        # Create state
        state = create_test_curator_state()
        state.discovered_videos = initial_videos
        
        # Test should_trigger_refinement
        should_refine, quality_analysis = engine.should_trigger_refinement(initial_videos, initial_ranking, state)
        
        assert should_refine == True
        assert len(quality_analysis.failure_types) > 0
        assert quality_analysis.refinement_urgency in ['high', 'critical']
        
        # Mock improved results after refinement
        improved_videos = [create_test_video(video_id=f'improved{i:02d}') for i in range(5)]
        mock_refinement_result = Mock()
        mock_refinement_result.videos = improved_videos
        mock_refinement_result.search_terms = ['ai', 'artificial intelligence']
        
        mock_search.perform_search_attempt.return_value = mock_refinement_result
        
        # Execute refinement
        updated_state = engine.execute_quality_based_refinement(state, quality_analysis)
        
        # Verify results
        assert len(updated_state.discovered_videos) == 5
        assert updated_state.search_attempt == 1
        assert updated_state.generation_metadata['quality_based_refinement'] == True
        assert 'quality_analysis_summary' in updated_state.generation_metadata
        
        # Verify quality analysis summary contains expected fields
        summary = updated_state.generation_metadata['quality_analysis_summary']
        assert 'failure_types' in summary
        assert 'refinement_urgency' in summary
        assert 'weekly_focus_maintained' in summary
        assert 'primary_action' in summary
        assert 'expected_improvement' in summary