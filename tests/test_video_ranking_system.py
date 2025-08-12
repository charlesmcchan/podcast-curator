"""
Tests for the video ranking system.
"""
from unittest.mock import Mock, patch
from datetime import datetime, timezone, timedelta
from src.nanook_curator.video_ranking_system import (
    VideoRankingSystem, RankingStrategy, QualityBand, RankingMetrics, RankingResult
)
from src.nanook_curator.engagement_analyzer import EngagementMetrics
from src.nanook_curator.content_quality_scorer import ContentQualityMetrics
from src.nanook_curator.models import VideoData, CuratorState
from src.nanook_curator.config import Configuration


def create_mock_config():
    """Create a mock configuration for testing."""
    mock_config = Mock(spec=Configuration)
    mock_config.default_search_keywords = ["AI news", "AI tools", "artificial intelligence"]
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
    return VideoData(**defaults)


def create_recent_video(**kwargs):
    """Create a test video uploaded recently."""
    recent_date = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    defaults = {'upload_date': recent_date}
    defaults.update(kwargs)
    return create_test_video(**defaults)


def create_old_video(**kwargs):
    """Create a test video uploaded long ago."""
    old_date = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
    defaults = {'upload_date': old_date}
    defaults.update(kwargs)
    return create_test_video(**defaults)


def create_mock_content_metrics(score=75.0):
    """Create mock content quality metrics."""
    return ContentQualityMetrics(
        coherence_score=score,
        information_density=score,
        technical_accuracy=score,
        clarity_score=score,
        depth_score=score,
        structure_score=score,
        overall_content_score=score,
        engagement_integration_score=score,
        final_quality_score=score,
        detailed_metrics={}
    )


def create_mock_engagement_metrics(score=75.0):
    """Create mock engagement metrics."""
    return EngagementMetrics(
        like_ratio=0.85,
        like_to_dislike_ratio=5.67,
        view_to_subscriber_ratio=0.2,
        comment_sentiment_score=0.4,
        engagement_rate=0.075,
        overall_engagement_score=score,
        meets_threshold=score >= 70.0,
        detailed_metrics={}
    )


class TestVideoRankingSystem:
    """Test cases for VideoRankingSystem class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        mock_config = create_mock_config()
        
        # Mock the dependencies to avoid initialization issues
        with patch('src.nanook_curator.video_ranking_system.ContentQualityScorer'), \
             patch('src.nanook_curator.video_ranking_system.EngagementAnalyzer'):
            self.ranking_system = VideoRankingSystem(config=mock_config)
    
    def test_init(self):
        """Test ranking system initialization."""
        assert self.ranking_system.config is not None
        assert self.ranking_system.default_strategy == RankingStrategy.BALANCED
        assert self.ranking_system.quality_threshold == 70.0
        assert self.ranking_system.min_quality_videos == 3
        assert len(self.ranking_system.strategy_weights) == 5
        assert len(self.ranking_system.quality_bands) == 5
    
    def test_calculate_freshness_score_recent(self):
        """Test freshness score calculation for recent videos."""
        recent_video = create_recent_video()
        score = self.ranking_system.calculate_freshness_score(recent_video)
        
        assert 90.0 <= score <= 100.0  # Should be very high for recent videos
    
    def test_calculate_freshness_score_old(self):
        """Test freshness score calculation for old videos."""
        old_video = create_old_video()
        score = self.ranking_system.calculate_freshness_score(old_video)
        
        assert 0.0 <= score <= 30.0  # Should be low for old videos
    
    def test_calculate_freshness_score_invalid_date(self):
        """Test freshness score with invalid upload date."""
        # Create video with valid format first, then test error handling directly
        video = create_test_video()
        
        # Mock the datetime parsing to raise an exception
        with patch('src.nanook_curator.video_ranking_system.datetime') as mock_datetime:
            mock_datetime.fromisoformat.side_effect = ValueError("Invalid date")
            score = self.ranking_system.calculate_freshness_score(video)
            
            assert score == 50.0  # Should return neutral score for invalid date
    
    def test_get_quality_band(self):
        """Test quality band determination."""
        assert self.ranking_system.get_quality_band(95.0) == QualityBand.EXCELLENT
        assert self.ranking_system.get_quality_band(85.0) == QualityBand.VERY_GOOD
        assert self.ranking_system.get_quality_band(75.0) == QualityBand.GOOD
        assert self.ranking_system.get_quality_band(65.0) == QualityBand.FAIR
        assert self.ranking_system.get_quality_band(45.0) == QualityBand.POOR
    
    def test_calculate_combined_score_balanced(self):
        """Test combined score calculation with balanced strategy."""
        video = create_test_video()
        content_metrics = create_mock_content_metrics(80.0)
        engagement_metrics = create_mock_engagement_metrics(75.0)
        
        ranking_metrics = self.ranking_system.calculate_combined_score(
            video, content_metrics, engagement_metrics, RankingStrategy.BALANCED
        )
        
        assert isinstance(ranking_metrics, RankingMetrics)
        assert ranking_metrics.content_score == 80.0
        assert ranking_metrics.engagement_score == 75.0
        assert 0.0 <= ranking_metrics.freshness_score <= 100.0
        assert 0.0 <= ranking_metrics.combined_score <= 100.0
        assert ranking_metrics.quality_band in QualityBand
        assert isinstance(ranking_metrics.meets_threshold, bool)
        assert 'strategy_used' in ranking_metrics.detailed_metrics
    
    def test_calculate_combined_score_content_focused(self):
        """Test combined score calculation with content-focused strategy."""
        video = create_test_video()
        content_metrics = create_mock_content_metrics(90.0)
        engagement_metrics = create_mock_engagement_metrics(60.0)
        
        ranking_metrics = self.ranking_system.calculate_combined_score(
            video, content_metrics, engagement_metrics, RankingStrategy.CONTENT_FOCUSED
        )
        
        # Content should have higher weight (0.60 vs 0.25)
        # Adjusted expectation based on freshness score being very low for old video
        assert ranking_metrics.combined_score > 65.0
        assert ranking_metrics.detailed_metrics['strategy_used'] == 'content_focused'
    
    def test_calculate_combined_score_engagement_focused(self):
        """Test combined score calculation with engagement-focused strategy."""
        video = create_test_video()
        content_metrics = create_mock_content_metrics(60.0)
        engagement_metrics = create_mock_engagement_metrics(90.0)
        
        ranking_metrics = self.ranking_system.calculate_combined_score(
            video, content_metrics, engagement_metrics, RankingStrategy.ENGAGEMENT_FOCUSED
        )
        
        # Engagement should have higher weight (0.60 vs 0.25)
        # Adjusted expectation based on freshness score being very low for old video
        assert ranking_metrics.combined_score > 65.0
        assert ranking_metrics.detailed_metrics['strategy_used'] == 'engagement_focused'
    
    def test_calculate_combined_score_freshness_focused(self):
        """Test combined score calculation with freshness-focused strategy."""
        recent_video = create_recent_video()
        content_metrics = create_mock_content_metrics(60.0)
        engagement_metrics = create_mock_engagement_metrics(60.0)
        
        ranking_metrics = self.ranking_system.calculate_combined_score(
            recent_video, content_metrics, engagement_metrics, RankingStrategy.FRESHNESS_FOCUSED
        )
        
        # Should get boost from high freshness score
        assert ranking_metrics.freshness_score > 90.0
        assert ranking_metrics.combined_score > 65.0
        assert ranking_metrics.detailed_metrics['strategy_used'] == 'freshness_focused'
    
    def test_rank_videos_empty_list(self):
        """Test ranking with empty video list."""
        result = self.ranking_system.rank_videos([])
        
        assert isinstance(result, RankingResult)
        assert len(result.ranked_videos) == 0
        assert result.quality_summary['total_videos'] == 0
        assert not result.threshold_analysis['meets_minimum_requirement']
        assert len(result.refinement_suggestions) > 0
        assert 'No videos provided' in result.refinement_suggestions[0]
    
    def test_rank_videos_single_video(self):
        """Test ranking with single video."""
        video = create_test_video(video_id='single12345')
        
        # Mock the scoring methods
        mock_content = create_mock_content_metrics(80.0)
        mock_engagement = create_mock_engagement_metrics(75.0)
        
        with patch.object(self.ranking_system.content_scorer, 'calculate_combined_quality_score', return_value=mock_content), \
             patch.object(self.ranking_system.engagement_analyzer, 'analyze_video_engagement', return_value=mock_engagement):
            
            result = self.ranking_system.rank_videos([video])
            
            assert len(result.ranked_videos) == 1
            assert result.ranked_videos[0].video_id == 'single12345'
            assert hasattr(result.ranked_videos[0], 'quality_score')
            assert result.quality_summary['total_videos'] == 1
    
    def test_rank_videos_multiple_videos(self):
        """Test ranking with multiple videos."""
        videos = [
            create_test_video(video_id='highqual123', title='High Quality Video'),
            create_test_video(video_id='medium_qual', title='Medium Quality Video'),
            create_test_video(video_id='lowqual_456', title='Low Quality Video')
        ]
        
        # Mock different quality scores for each video
        def mock_content_scorer(video, **kwargs):
            scores = {
                'highqual123': 90.0,
                'medium_qual': 70.0,
                'lowqual_456': 50.0
            }
            return create_mock_content_metrics(scores.get(video.video_id, 60.0))
        
        def mock_engagement_analyzer(video, comments):
            scores = {
                'highqual123': 85.0,
                'medium_qual': 65.0,
                'lowqual_456': 45.0
            }
            return create_mock_engagement_metrics(scores.get(video.video_id, 60.0))
        
        with patch.object(self.ranking_system.content_scorer, 'calculate_combined_quality_score', side_effect=mock_content_scorer), \
             patch.object(self.ranking_system.engagement_analyzer, 'analyze_video_engagement', side_effect=mock_engagement_analyzer):
            
            result = self.ranking_system.rank_videos(videos, target_count=3)
            
            assert len(result.ranked_videos) == 3
            # Should be ranked by quality (high to low)
            assert result.ranked_videos[0].video_id == 'highqual123'
            assert result.ranked_videos[1].video_id == 'medium_qual'
            assert result.ranked_videos[2].video_id == 'lowqual_456'
            
            # Check ranking metadata
            assert result.ranking_metadata['total_videos_ranked'] == 3
            assert result.ranking_metadata['selected_count'] == 3
            assert 'strategy' in result.ranking_metadata
    
    def test_rank_videos_with_comments(self):
        """Test ranking with video comments provided."""
        video = create_test_video(video_id='video123456')
        video_comments = {
            'video123456': ['Great content!', 'Very helpful', 'Learned a lot']
        }
        
        mock_content = create_mock_content_metrics(75.0)
        mock_engagement = create_mock_engagement_metrics(80.0)
        
        with patch.object(self.ranking_system.content_scorer, 'calculate_combined_quality_score', return_value=mock_content) as mock_content_call, \
             patch.object(self.ranking_system.engagement_analyzer, 'analyze_video_engagement', return_value=mock_engagement) as mock_engagement_call:
            
            result = self.ranking_system.rank_videos([video], video_comments=video_comments)
            
            # Verify comments were passed to the analyzers
            mock_content_call.assert_called_once()
            mock_engagement_call.assert_called_once_with(video, ['Great content!', 'Very helpful', 'Learned a lot'])
    
    def test_rank_videos_error_handling(self):
        """Test error handling during video ranking."""
        video = create_test_video(video_id='error123456')
        
        # Mock content scorer to raise an exception
        with patch.object(self.ranking_system.content_scorer, 'calculate_combined_quality_score', side_effect=Exception("Content scoring error")), \
             patch.object(self.ranking_system.engagement_analyzer, 'analyze_video_engagement', return_value=create_mock_engagement_metrics()):
            
            result = self.ranking_system.rank_videos([video])
            
            assert len(result.ranked_videos) == 1
            # Video should have quality_score of 0.0 due to error
            assert result.ranked_videos[0].quality_score == 0.0
            # Should still include the video in results
            assert result.ranked_videos[0].video_id == 'error123456'
    
    def test_select_adaptive_strategy(self):
        """Test adaptive strategy selection."""
        # Test with mostly recent videos
        recent_videos = [create_recent_video(video_id=f'recent{i:05d}', transcript='test transcript') for i in range(5)]
        strategy = self.ranking_system._select_adaptive_strategy(recent_videos)
        assert strategy == RankingStrategy.FRESHNESS_FOCUSED
        
        # Test with mostly videos with transcripts
        transcript_videos = [create_test_video(video_id=f'trans{i:06d}', transcript='detailed transcript content') for i in range(5)]
        strategy = self.ranking_system._select_adaptive_strategy(transcript_videos)
        assert strategy == RankingStrategy.CONTENT_FOCUSED
        
        # Test with few transcripts
        no_transcript_videos = [create_test_video(video_id=f'notran{i:05d}', transcript=None) for i in range(5)]
        strategy = self.ranking_system._select_adaptive_strategy(no_transcript_videos)
        assert strategy == RankingStrategy.ENGAGEMENT_FOCUSED
        
        # Test with balanced mix
        mixed_videos = [
            create_test_video(video_id='mixed123456', transcript='some transcript'),
            create_test_video(video_id='mixed234567', transcript='another transcript'),
            create_test_video(video_id='mixed345678', transcript=None)
        ]
        strategy = self.ranking_system._select_adaptive_strategy(mixed_videos)
        assert strategy == RankingStrategy.BALANCED
    
    def test_analyze_quality_distribution(self):
        """Test quality distribution analysis."""
        # Create mock video rankings with different scores
        video1 = create_test_video(video_id='excellent12')
        video2 = create_test_video(video_id='good1234567')
        video3 = create_test_video(video_id='poor1234567')
        
        rankings = [
            (video1, RankingMetrics(85.0, 80.0, 90.0, 85.0, QualityBand.VERY_GOOD, 1, True, {})),
            (video2, RankingMetrics(70.0, 65.0, 75.0, 70.0, QualityBand.GOOD, 2, True, {})),
            (video3, RankingMetrics(40.0, 35.0, 45.0, 40.0, QualityBand.POOR, 3, False, {}))
        ]
        
        distribution = self.ranking_system._analyze_quality_distribution(rankings)
        
        assert distribution['total_videos'] == 3
        assert 60.0 < distribution['average_score'] < 70.0
        assert distribution['min_score'] == 40.0
        assert distribution['max_score'] == 85.0
        assert 'score_distribution' in distribution
        assert 'quality_bands' in distribution
    
    def test_analyze_threshold_compliance(self):
        """Test threshold compliance analysis."""
        # Create rankings with some above and below threshold
        video1 = create_test_video(video_id='above123456')
        video2 = create_test_video(video_id='above234567')
        video3 = create_test_video(video_id='below123456')
        
        rankings = [
            (video1, RankingMetrics(80.0, 75.0, 85.0, 80.0, QualityBand.VERY_GOOD, 1, True, {})),
            (video2, RankingMetrics(75.0, 70.0, 80.0, 75.0, QualityBand.GOOD, 2, True, {})),
            (video3, RankingMetrics(60.0, 55.0, 65.0, 60.0, QualityBand.FAIR, 3, False, {}))
        ]
        
        analysis = self.ranking_system._analyze_threshold_compliance(rankings)
        
        assert analysis['videos_above_threshold'] == 2
        assert analysis['total_videos'] == 3
        assert analysis['threshold_percentage'] == (2/3) * 100
        assert not analysis['meets_minimum_requirement']  # Need 3, only have 2
        assert analysis['shortfall'] == 1
        assert analysis['threshold_used'] == 70.0
    
    def test_generate_refinement_suggestions(self):
        """Test refinement suggestions generation."""
        # Create rankings that don't meet minimum requirements
        video1 = create_test_video(video_id='low12345678')
        video2 = create_test_video(video_id='low23456789')
        
        rankings = [
            (video1, RankingMetrics(60.0, 55.0, 65.0, 60.0, QualityBand.FAIR, 1, False, {})),
            (video2, RankingMetrics(50.0, 45.0, 55.0, 50.0, QualityBand.POOR, 2, False, {}))
        ]
        
        threshold_analysis = {
            'meets_minimum_requirement': False,
            'shortfall': 3,
            'videos_above_threshold': 0,
            'total_videos': 2
        }
        
        suggestions = self.ranking_system._generate_refinement_suggestions(rankings, threshold_analysis)
        
        assert len(suggestions) > 0
        assert any('Need 3 more videos' in suggestion for suggestion in suggestions)
        assert any('quality is low' in suggestion for suggestion in suggestions)
    
    def test_evaluate_search_refinement_need(self):
        """Test search refinement evaluation."""
        videos = [
            create_test_video(video_id='eval1234567', transcript='Good content'),
            create_test_video(video_id='eval2345678', transcript='Poor content'),
        ]
        
        # Mock the ranking result
        mock_result = RankingResult(
            ranked_videos=videos,
            quality_summary={'total_videos': 2, 'average_score': 60.0},
            threshold_analysis={
                'meets_minimum_requirement': False,
                'shortfall': 2,
                'threshold_percentage': 0.0,
                'videos_above_threshold': 0
            },
            refinement_suggestions=['Need better quality videos'],
            ranking_metadata={'strategy': 'balanced'}
        )
        
        with patch.object(self.ranking_system, 'rank_videos', return_value=mock_result):
            evaluation = self.ranking_system.evaluate_search_refinement_need(videos)
            
            assert evaluation['needs_refinement'] == True
            assert 0.0 <= evaluation['confidence_score'] <= 100.0
            assert evaluation['refinement_priority'] in ['high', 'medium', 'low']
            assert 'quality_summary' in evaluation
            assert 'recommended_actions' in evaluation
    
    def test_generate_refinement_actions(self):
        """Test refinement actions generation."""
        ranking_result = RankingResult(
            ranked_videos=[],
            quality_summary={'total_videos': 5, 'average_score': 45.0},
            threshold_analysis={'shortfall': 2, 'meets_minimum_requirement': False},
            refinement_suggestions=['Content is not recent'],
            ranking_metadata={}
        )
        
        actions = self.ranking_system._generate_refinement_actions(ranking_result)
        
        assert len(actions) > 0
        action_types = [action['action'] for action in actions]
        assert 'refine_keywords' in action_types  # Due to low average score
        assert 'adjust_time_range' in action_types  # Due to freshness suggestion
    
    def test_update_curator_state(self):
        """Test curator state update with ranking results."""
        state = CuratorState(search_keywords=['test'])
        
        videos = [create_test_video(video_id='state123456')]
        ranking_result = RankingResult(
            ranked_videos=videos,
            quality_summary={'average_score': 75.0},
            threshold_analysis={
                'meets_minimum_requirement': True,
                'videos_above_threshold': 1
            },
            refinement_suggestions=[],
            ranking_metadata={'strategy': 'balanced'}
        )
        
        updated_state = self.ranking_system.update_curator_state(state, ranking_result)
        
        assert len(updated_state.ranked_videos) == 1
        assert updated_state.ranked_videos[0].video_id == 'state123456'
        assert 'ranking_strategy' in updated_state.generation_metadata
        assert 'quality_summary' in updated_state.generation_metadata
    
    def test_update_curator_state_with_refinement_needed(self):
        """Test curator state update when refinement is needed."""
        state = CuratorState(search_keywords=['test'])
        
        ranking_result = RankingResult(
            ranked_videos=[],
            quality_summary={'average_score': 45.0},
            threshold_analysis={
                'meets_minimum_requirement': False,
                'videos_above_threshold': 0
            },
            refinement_suggestions=['Need better quality videos', 'Expand search scope'],
            ranking_metadata={'strategy': 'balanced'}
        )
        
        updated_state = self.ranking_system.update_curator_state(state, ranking_result)
        
        # Should add errors for refinement suggestions
        assert len(updated_state.errors) >= 2
        assert any('Quality refinement needed' in error for error in updated_state.errors)
    
    def test_get_ranking_summary_empty(self):
        """Test ranking summary with empty video list."""
        summary = self.ranking_system.get_ranking_summary([])
        
        assert summary['total_videos'] == 0
        assert summary['strategy_recommendation'] == 'balanced'
        assert summary['quality_outlook'] == 'no_data'
    
    def test_get_ranking_summary_with_videos(self):
        """Test ranking summary with videos."""
        videos = [
            create_test_video(video_id='summary1234', transcript='Good transcript'),
            create_recent_video(video_id='summary5678', transcript='Recent content'),
            create_test_video(video_id='summary9012', transcript=None)
        ]
        
        summary = self.ranking_system.get_ranking_summary(videos)
        
        assert summary['total_videos'] == 3
        assert summary['videos_with_transcripts'] == 2
        assert summary['recent_videos'] >= 1  # At least the recent video
        assert 0.0 <= summary['transcript_ratio'] <= 1.0
        assert 0.0 <= summary['recent_ratio'] <= 1.0
        assert summary['strategy_recommendation'] in ['balanced', 'content_focused', 'engagement_focused', 'freshness_focused']
        assert summary['quality_outlook'] in ['excellent', 'good', 'fair', 'challenging', 'no_data']
    
    def test_ranking_metrics_dataclass(self):
        """Test RankingMetrics dataclass functionality."""
        metrics = RankingMetrics(
            content_score=80.0,
            engagement_score=75.0,
            freshness_score=85.0,
            combined_score=79.0,
            quality_band=QualityBand.GOOD,
            rank_position=1,
            meets_threshold=True,
            detailed_metrics={'test': 'data'}
        )
        
        assert metrics.content_score == 80.0
        assert metrics.engagement_score == 75.0
        assert metrics.freshness_score == 85.0
        assert metrics.combined_score == 79.0
        assert metrics.quality_band == QualityBand.GOOD
        assert metrics.rank_position == 1
        assert metrics.meets_threshold == True
        assert metrics.detailed_metrics['test'] == 'data'
    
    def test_ranking_result_dataclass(self):
        """Test RankingResult dataclass functionality."""
        video = create_test_video()
        result = RankingResult(
            ranked_videos=[video],
            quality_summary={'total': 1},
            threshold_analysis={'meets_requirement': True},
            refinement_suggestions=['suggestion1'],
            ranking_metadata={'strategy': 'balanced'}
        )
        
        assert len(result.ranked_videos) == 1
        assert result.quality_summary['total'] == 1
        assert result.threshold_analysis['meets_requirement'] == True
        assert result.refinement_suggestions == ['suggestion1']
        assert result.ranking_metadata['strategy'] == 'balanced'
    
    def test_quality_band_enum(self):
        """Test QualityBand enum values."""
        assert QualityBand.EXCELLENT.value == 'excellent'
        assert QualityBand.VERY_GOOD.value == 'very_good'
        assert QualityBand.GOOD.value == 'good'
        assert QualityBand.FAIR.value == 'fair'
        assert QualityBand.POOR.value == 'poor'
    
    def test_ranking_strategy_enum(self):
        """Test RankingStrategy enum values."""
        assert RankingStrategy.BALANCED.value == 'balanced'
        assert RankingStrategy.CONTENT_FOCUSED.value == 'content_focused'
        assert RankingStrategy.ENGAGEMENT_FOCUSED.value == 'engagement_focused'
        assert RankingStrategy.FRESHNESS_FOCUSED.value == 'freshness_focused'
        assert RankingStrategy.HYBRID.value == 'hybrid'
    
    def test_strategy_weights_configuration(self):
        """Test that all strategies have proper weight configurations."""
        for strategy in RankingStrategy:
            weights = self.ranking_system.strategy_weights[strategy]
            
            # Check that all required keys are present
            assert 'content' in weights
            assert 'engagement' in weights
            assert 'freshness' in weights
            
            # Check that weights sum to approximately 1.0
            total_weight = weights['content'] + weights['engagement'] + weights['freshness']
            assert 0.99 <= total_weight <= 1.01
            
            # Check that all weights are positive
            assert weights['content'] > 0
            assert weights['engagement'] > 0
            assert weights['freshness'] > 0
    
    def test_quality_band_thresholds(self):
        """Test quality band threshold ranges."""
        bands = self.ranking_system.quality_bands
        
        # Check that ranges don't overlap and cover 0-100
        sorted_bands = sorted(bands.items(), key=lambda x: x[1][0])
        
        for i, (band, (min_score, max_score)) in enumerate(sorted_bands):
            assert min_score <= max_score
            if i > 0:
                prev_max = sorted_bands[i-1][1][1]
                assert min_score > prev_max or abs(min_score - prev_max) < 0.1  # Allow small overlap for boundary cases
        
        # Check that we cover the full range
        assert sorted_bands[0][1][0] == 0.0  # Lowest minimum should be 0
        assert sorted_bands[-1][1][1] == 100.0  # Highest maximum should be 100
    
    def test_video_age_calculation(self):
        """Test video age calculation helper method."""
        # Test with valid date
        recent_video = create_recent_video()
        age = self.ranking_system._calculate_video_age_days(recent_video)
        assert 0 <= age <= 10  # Should be recent
        
        # Test with old date
        old_video = create_old_video()
        age = self.ranking_system._calculate_video_age_days(old_video)
        assert age > 100  # Should be old
        
        # Test with invalid date by mocking datetime parsing
        video = create_test_video()
        with patch('src.nanook_curator.video_ranking_system.datetime') as mock_datetime:
            mock_datetime.fromisoformat.side_effect = ValueError("Invalid date")
            age = self.ranking_system._calculate_video_age_days(video)
            assert age == 999  # Should return fallback value
    
    def test_comprehensive_ranking_workflow(self):
        """Test the complete ranking workflow with realistic data."""
        # Create diverse set of videos
        videos = [
            create_test_video(
                video_id='excellent12', 
                title='Excellent AI Research Paper Review',
                transcript='Comprehensive analysis of transformer architecture with detailed technical explanations.',
                view_count=50000,
                like_count=2000,
                comment_count=150
            ),
            create_recent_video(
                video_id='recent12345',
                title='Latest AI News Update',
                transcript='Recent developments in artificial intelligence and machine learning.',
                view_count=25000,
                like_count=800,
                comment_count=60
            ),
            create_test_video(
                video_id='average1234',
                title='Average AI Tutorial',
                transcript='Basic introduction to machine learning concepts.',
                view_count=15000,
                like_count=600,
                comment_count=40
            ),
            create_old_video(
                video_id='old12345678',
                title='Old AI Discussion',
                transcript='Historical perspective on artificial intelligence development.',
                view_count=8000,
                like_count=200,
                comment_count=15
            ),
            create_test_video(
                video_id='noTranscr12',
                title='No Transcript Video',
                transcript=None,
                view_count=20000,
                like_count=900,
                comment_count=80
            )
        ]
        
        comments = {
            'excellent12': ['Amazing content!', 'Very detailed explanation', 'Learned so much'],
            'recent12345': ['Great update', 'Thanks for sharing', 'Very timely'],
            'average1234': ['Good basics', 'Helpful for beginners'],
            'old12345678': ['Still relevant', 'Good historical context'],
            'noTranscr12': ['Great visuals', 'Could use captions']
        }
        
        # Mock scoring to return realistic values
        def mock_content_scoring(video, **kwargs):
            scores = {
                'excellent12': 90.0,
                'recent12345': 75.0,
                'average1234': 65.0,
                'old12345678': 70.0,
                'noTranscr12': 30.0  # Low due to no transcript
            }
            return create_mock_content_metrics(scores.get(video.video_id, 50.0))
        
        def mock_engagement_scoring(video, comments_list):
            scores = {
                'excellent12': 85.0,
                'recent12345': 70.0,
                'average1234': 60.0,
                'old12345678': 50.0,
                'noTranscr12': 75.0  # Good engagement despite no transcript
            }
            return create_mock_engagement_metrics(scores.get(video.video_id, 50.0))
        
        with patch.object(self.ranking_system.content_scorer, 'calculate_combined_quality_score', side_effect=mock_content_scoring), \
             patch.object(self.ranking_system.engagement_analyzer, 'analyze_video_engagement', side_effect=mock_engagement_scoring):
            
            result = self.ranking_system.rank_videos(
                videos, 
                strategy=RankingStrategy.BALANCED,
                video_comments=comments,
                target_count=3
            )
            
            # Verify result structure
            assert isinstance(result, RankingResult)
            assert len(result.ranked_videos) == 3  # Should select top 3
            
            # Verify ranking order (should be ranked by combined score)
            # The recent video might score higher due to freshness boost, so let's check top 2
            top_video_ids = [v.video_id for v in result.ranked_videos[:2]]
            assert 'excellent12' in top_video_ids  # Should be in top 2
            assert 'recent12345' in top_video_ids  # Recent video should also be in top 2
            
            # Verify metadata
            assert result.ranking_metadata['total_videos_ranked'] == 5
            assert result.ranking_metadata['selected_count'] == 3
            assert result.ranking_metadata['strategy'] == 'balanced'
            
            # Verify quality analysis
            assert result.quality_summary['total_videos'] == 5
            assert result.quality_summary['average_score'] > 0
            
            # Verify threshold analysis
            assert 'meets_minimum_requirement' in result.threshold_analysis
            assert 'videos_above_threshold' in result.threshold_analysis
            
            # Verify that videos have attached metrics
            for video in result.ranked_videos:
                assert hasattr(video, 'quality_score')
                assert hasattr(video, 'ranking_metrics')
                assert video.quality_score > 0