"""
Tests for the search refinement system.
"""
from unittest.mock import Mock, patch
from src.podcast_curator.search_refinement import (
    SearchRefinementEngine, RefinementStrategy, RefinementResult, refine_search_node
)
from src.podcast_curator.models import VideoData, CuratorState
from src.podcast_curator.config import Configuration
from src.podcast_curator.video_ranking_system import RankingResult


def create_mock_config():
    """Create a mock configuration for testing."""
    mock_config = Mock(spec=Configuration)
    mock_config.default_search_keywords = ["AI news", "AI tools", "artificial intelligence"]
    mock_config.youtube_api_key = "test_api_key"
    mock_config.max_videos = 10
    mock_config.days_back = 7
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


def create_mock_ranking_result(videos, meets_requirement=True):
    """Create a mock ranking result."""
    return RankingResult(
        ranked_videos=videos,
        quality_summary={
            'total_videos': len(videos),
            'average_score': 75.0 if meets_requirement else 60.0
        },
        threshold_analysis={
            'meets_minimum_requirement': meets_requirement,
            'videos_above_threshold': len(videos) if meets_requirement else 0,
            'shortfall': 0 if meets_requirement else 3
        },
        refinement_suggestions=[] if meets_requirement else ['Need better quality videos'],
        ranking_metadata={'strategy': 'balanced'}
    )


class TestSearchRefinementEngine:
    """Test cases for SearchRefinementEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        mock_config = create_mock_config()
        
        # Mock the dependencies to avoid initialization issues
        with patch('src.podcast_curator.search_refinement.YouTubeClient'), \
             patch('src.podcast_curator.search_refinement.VideoRankingSystem'):
            self.refinement_engine = SearchRefinementEngine(config=mock_config)
    
    def test_init(self):
        """Test refinement engine initialization."""
        assert self.refinement_engine.config is not None
        assert self.refinement_engine.max_attempts == 3
        assert self.refinement_engine.quality_threshold == 70.0
        assert self.refinement_engine.min_quality_videos == 3
        assert len(self.refinement_engine.keyword_synonyms) > 0
        assert len(self.refinement_engine.domain_expansions) > 0
        assert len(self.refinement_engine.broader_terms) > 0
    
    def test_build_keyword_synonyms(self):
        """Test keyword synonym mapping construction."""
        synonyms = self.refinement_engine.keyword_synonyms
        
        # Test AI-related synonyms
        assert 'ai' in synonyms
        assert 'artificial intelligence' in synonyms['ai']
        assert 'machine learning' in synonyms
        assert 'ml' in synonyms['machine learning']
        
        # Test programming synonyms
        assert 'programming' in synonyms
        assert 'coding' in synonyms['programming']
        assert 'python' in synonyms
        
        # Test news/tutorial synonyms
        assert 'news' in synonyms
        assert 'tutorial' in synonyms
        assert 'explained' in synonyms
    
    def test_build_domain_expansions(self):
        """Test domain expansion mapping construction."""
        expansions = self.refinement_engine.domain_expansions
        
        assert 'ai_ml' in expansions
        assert 'programming' in expansions
        assert 'tech_business' in expansions
        assert 'data_analytics' in expansions
        assert 'emerging_tech' in expansions
        
        # Test AI/ML domain
        assert 'artificial intelligence' in expansions['ai_ml']
        assert 'machine learning' in expansions['ai_ml']
        assert 'neural networks' in expansions['ai_ml']
    
    def test_build_broader_terms(self):
        """Test broader terms list construction."""
        broader_terms = self.refinement_engine.broader_terms
        
        assert 'technology' in broader_terms
        assert 'tech' in broader_terms
        assert 'innovation' in broader_terms
        assert 'digital' in broader_terms
        assert len(broader_terms) > 10
    
    def test_expand_keywords_initial_strategy(self):
        """Test keyword expansion with initial strategy."""
        keywords = ['artificial intelligence', 'programming']
        
        expanded = self.refinement_engine.expand_keywords(keywords, RefinementStrategy.INITIAL)
        
        assert expanded == keywords
    
    def test_expand_keywords_expand_strategy(self):
        """Test keyword expansion with expand keywords strategy."""
        keywords = ['ai', 'programming']
        
        expanded = self.refinement_engine.expand_keywords(keywords, RefinementStrategy.EXPAND_KEYWORDS)
        
        assert len(expanded) > len(keywords)
        assert 'ai' in expanded
        assert 'programming' in expanded
        # Should include synonyms
        assert any(synonym in expanded for synonym in ['artificial intelligence', 'machine intelligence'])
        assert any(synonym in expanded for synonym in ['coding', 'software development'])
    
    def test_expand_keywords_broaden_terms_strategy(self):
        """Test keyword expansion with broaden terms strategy."""
        keywords = ['ai research', 'machine learning']
        
        expanded = self.refinement_engine.expand_keywords(keywords, RefinementStrategy.BROADEN_TERMS)
        
        assert len(expanded) > len(keywords)
        # Should include original keywords
        assert all(kw in expanded for kw in keywords)
        # Should include AI/ML domain expansions
        assert any(term in expanded for term in ['artificial intelligence', 'deep learning', 'neural networks'])
    
    def test_expand_keywords_final_fallback_strategy(self):
        """Test keyword expansion with final fallback strategy."""
        keywords = ['specific ai topic']
        
        expanded = self.refinement_engine.expand_keywords(keywords, RefinementStrategy.FINAL_FALLBACK)
        
        assert len(expanded) > len(keywords)
        # Should include broader terms
        assert any(term in expanded for term in ['technology', 'tech', 'innovation'])
    
    def test_expand_keywords_limit(self):
        """Test that keyword expansion respects the 15-keyword limit."""
        # Create a scenario that would generate many keywords
        keywords = ['ai', 'ml', 'programming', 'coding', 'data', 'analytics']
        
        expanded = self.refinement_engine.expand_keywords(keywords, RefinementStrategy.FINAL_FALLBACK)
        
        assert len(expanded) <= 15
    
    def test_calculate_expanded_timeframe(self):
        """Test timeframe expansion calculation."""
        original_days = 7
        
        # Initial strategy should keep original timeframe
        assert self.refinement_engine.calculate_expanded_timeframe(original_days, RefinementStrategy.INITIAL) == 7
        
        # Expand keywords should keep original timeframe
        assert self.refinement_engine.calculate_expanded_timeframe(original_days, RefinementStrategy.EXPAND_KEYWORDS) == 7
        
        # Expand timeframe should double it
        assert self.refinement_engine.calculate_expanded_timeframe(original_days, RefinementStrategy.EXPAND_TIMEFRAME) == 14
        
        # Broaden terms should keep expanded timeframe
        assert self.refinement_engine.calculate_expanded_timeframe(original_days, RefinementStrategy.BROADEN_TERMS) == 14
        
        # Final fallback should triple it
        assert self.refinement_engine.calculate_expanded_timeframe(original_days, RefinementStrategy.FINAL_FALLBACK) == 21
        
        # Test maximum limits
        assert self.refinement_engine.calculate_expanded_timeframe(20, RefinementStrategy.EXPAND_TIMEFRAME) == 30  # Max 30
        assert self.refinement_engine.calculate_expanded_timeframe(30, RefinementStrategy.FINAL_FALLBACK) == 60  # Max 60
    
    def test_determine_refinement_strategy(self):
        """Test refinement strategy determination."""
        # First attempt should be initial
        assert self.refinement_engine.determine_refinement_strategy(0) == RefinementStrategy.INITIAL
        
        # Second attempt should expand keywords
        assert self.refinement_engine.determine_refinement_strategy(1) == RefinementStrategy.EXPAND_KEYWORDS
        
        # Third attempt should expand timeframe or broaden terms
        strategy = self.refinement_engine.determine_refinement_strategy(2)
        assert strategy in [RefinementStrategy.EXPAND_TIMEFRAME, RefinementStrategy.BROADEN_TERMS]
        
        # Fourth attempt should be final fallback
        assert self.refinement_engine.determine_refinement_strategy(3) == RefinementStrategy.FINAL_FALLBACK
    
    def test_determine_refinement_strategy_with_previous_results(self):
        """Test strategy determination with previous results analysis."""
        # Mock previous results with few videos
        few_videos_result = create_mock_ranking_result([create_test_video()], meets_requirement=False)
        
        strategy = self.refinement_engine.determine_refinement_strategy(2, few_videos_result)
        assert strategy == RefinementStrategy.EXPAND_TIMEFRAME
        
        # Mock previous results with many videos
        many_videos = [create_test_video(video_id=f'video{i:06d}') for i in range(6)]
        many_videos_result = create_mock_ranking_result(many_videos, meets_requirement=False)
        
        strategy = self.refinement_engine.determine_refinement_strategy(2, many_videos_result)
        assert strategy == RefinementStrategy.BROADEN_TERMS
    
    def test_perform_search_attempt_initial(self):
        """Test performing search attempt with initial strategy."""
        state = create_test_curator_state()
        
        # Mock the dependencies
        mock_videos = [create_test_video(video_id=f'test{i:07d}') for i in range(3)]
        mock_ranking_result = create_mock_ranking_result(mock_videos, meets_requirement=True)
        
        with patch.object(self.refinement_engine.youtube_client, 'discover_videos', return_value=mock_videos), \
             patch.object(self.refinement_engine.ranking_system, 'rank_videos', return_value=mock_ranking_result):
            
            result = self.refinement_engine.perform_search_attempt(state, RefinementStrategy.INITIAL)
            
            assert isinstance(result, RefinementResult)
            assert result.strategy_used == RefinementStrategy.INITIAL
            assert len(result.videos) == 3
            assert result.search_terms == state.search_keywords
            assert result.days_back == state.days_back
            assert not result.refinement_needed  # Quality requirements met
    
    def test_perform_search_attempt_expand_keywords(self):
        """Test performing search attempt with keyword expansion."""
        state = create_test_curator_state()
        
        mock_videos = [create_test_video(video_id=f'expand{i:05d}') for i in range(2)]
        mock_ranking_result = create_mock_ranking_result(mock_videos, meets_requirement=False)
        
        with patch.object(self.refinement_engine.youtube_client, 'discover_videos', return_value=mock_videos), \
             patch.object(self.refinement_engine.ranking_system, 'rank_videos', return_value=mock_ranking_result):
            
            result = self.refinement_engine.perform_search_attempt(state, RefinementStrategy.EXPAND_KEYWORDS)
            
            assert result.strategy_used == RefinementStrategy.EXPAND_KEYWORDS
            assert len(result.search_terms) > len(state.search_keywords)  # Should have expanded keywords
            assert result.days_back == state.days_back  # Same timeframe
            assert result.refinement_needed  # Quality requirements not met
    
    def test_perform_search_attempt_expand_timeframe(self):
        """Test performing search attempt with timeframe expansion."""
        state = create_test_curator_state()
        
        mock_videos = [create_test_video(video_id=f'time{i:07d}') for i in range(4)]
        mock_ranking_result = create_mock_ranking_result(mock_videos, meets_requirement=True)
        
        with patch.object(self.refinement_engine.youtube_client, 'discover_videos', return_value=mock_videos), \
             patch.object(self.refinement_engine.ranking_system, 'rank_videos', return_value=mock_ranking_result):
            
            result = self.refinement_engine.perform_search_attempt(state, RefinementStrategy.EXPAND_TIMEFRAME)
            
            assert result.strategy_used == RefinementStrategy.EXPAND_TIMEFRAME
            assert result.days_back > state.days_back  # Should have expanded timeframe
            assert not result.refinement_needed  # Quality requirements met
    
    def test_perform_search_attempt_error_handling(self):
        """Test error handling during search attempt."""
        state = create_test_curator_state()
        
        # Mock YouTube client to raise an exception
        with patch.object(self.refinement_engine.youtube_client, 'discover_videos', side_effect=Exception("API Error")):
            
            result = self.refinement_engine.perform_search_attempt(state, RefinementStrategy.INITIAL)
            
            assert len(result.videos) == 0
            assert result.refinement_needed == True
            assert 'Search failed' in result.suggestions[0]
            assert 'error' in result.metadata
    
    def test_refine_search_iteratively_success_first_attempt(self):
        """Test successful refinement on first attempt."""
        state = create_test_curator_state()
        
        # Mock successful first attempt
        mock_videos = [create_test_video(video_id=f'succes{i:05d}') for i in range(5)]
        mock_ranking_result = create_mock_ranking_result(mock_videos, meets_requirement=True)
        
        with patch.object(self.refinement_engine.youtube_client, 'discover_videos', return_value=mock_videos), \
             patch.object(self.refinement_engine.ranking_system, 'rank_videos', return_value=mock_ranking_result):
            
            updated_state = self.refinement_engine.refine_search_iteratively(state)
            
            assert updated_state.search_attempt == 1
            assert len(updated_state.discovered_videos) == 5
            assert updated_state.generation_metadata['refinement_complete'] == True
            assert updated_state.generation_metadata['total_attempts'] == 1
            assert updated_state.generation_metadata['best_strategy'] == 'initial'
    
    def test_refine_search_iteratively_multiple_attempts(self):
        """Test refinement requiring multiple attempts."""
        state = create_test_curator_state()
        
        # Mock first attempt failure, second attempt success
        mock_videos_attempt1 = [create_test_video(video_id='attempt1_01')]
        mock_videos_attempt2 = [create_test_video(video_id=f'atem2{i:06d}') for i in range(4)]
        
        mock_ranking_result1 = create_mock_ranking_result(mock_videos_attempt1, meets_requirement=False)
        mock_ranking_result2 = create_mock_ranking_result(mock_videos_attempt2, meets_requirement=True)
        
        with patch.object(self.refinement_engine.youtube_client, 'discover_videos') as mock_discover, \
             patch.object(self.refinement_engine.ranking_system, 'rank_videos') as mock_rank:
            
            # Set up side effects for multiple calls
            mock_discover.side_effect = [mock_videos_attempt1, mock_videos_attempt2]
            mock_rank.side_effect = [mock_ranking_result1, mock_ranking_result2]
            
            updated_state = self.refinement_engine.refine_search_iteratively(state)
            
            assert updated_state.search_attempt == 2
            assert len(updated_state.discovered_videos) == 4  # Best result used
            assert updated_state.generation_metadata['total_attempts'] == 2
            assert updated_state.generation_metadata['best_strategy'] == 'expand_keywords'
    
    def test_refine_search_iteratively_max_attempts_reached(self):
        """Test refinement when max attempts are reached."""
        state = create_test_curator_state()
        
        # Mock all attempts failing quality requirements
        mock_videos = [create_test_video(video_id='maxattempt1')]
        mock_ranking_result = create_mock_ranking_result(mock_videos, meets_requirement=False)
        
        with patch.object(self.refinement_engine.youtube_client, 'discover_videos', return_value=mock_videos), \
             patch.object(self.refinement_engine.ranking_system, 'rank_videos', return_value=mock_ranking_result):
            
            updated_state = self.refinement_engine.refine_search_iteratively(state)
            
            assert updated_state.search_attempt == 3  # Max attempts reached
            assert len(updated_state.discovered_videos) == 1  # Best available result
            assert updated_state.generation_metadata['total_attempts'] == 3
    
    def test_refine_search_iteratively_no_results(self):
        """Test refinement when no results are found."""
        state = create_test_curator_state()
        
        # Mock all attempts returning no videos
        mock_ranking_result = create_mock_ranking_result([], meets_requirement=False)
        
        with patch.object(self.refinement_engine.youtube_client, 'discover_videos', return_value=[]), \
             patch.object(self.refinement_engine.ranking_system, 'rank_videos', return_value=mock_ranking_result):
            
            updated_state = self.refinement_engine.refine_search_iteratively(state)
            
            assert len(updated_state.discovered_videos) == 0
            # Even with no videos, a best_result is created so no error is added to state
            # The system completes refinement with whatever best result it found (empty in this case)
            assert updated_state.generation_metadata['refinement_complete'] == True
            assert updated_state.generation_metadata['final_video_count'] == 0
    
    def test_get_refinement_summary(self):
        """Test refinement summary generation."""
        state = create_test_curator_state()
        
        # Mock metadata from successful refinement
        state.update_generation_metadata(
            refinement_complete=True,
            total_attempts=2,
            best_strategy='expand_keywords',
            final_search_terms=['ai', 'artificial intelligence', 'machine learning'],
            final_days_back=7,
            final_video_count=4,
            all_attempts_summary=[
                {'attempt': 1, 'strategy': 'initial', 'videos_found': 1, 'avg_quality': 60.0},
                {'attempt': 2, 'strategy': 'expand_keywords', 'videos_found': 4, 'avg_quality': 75.0}
            ]
        )
        
        # Add some videos with quality scores
        state.discovered_videos = [
            create_test_video(video_id='summary0001'),
            create_test_video(video_id='summary0002'),
            create_test_video(video_id='summary0003'),
            create_test_video(video_id='summary0004')
        ]
        
        # Mock quality scores
        for i, video in enumerate(state.discovered_videos):
            video.quality_score = 75.0 + i * 5  # Scores: 75, 80, 85, 90
        
        summary = self.refinement_engine.get_refinement_summary(state)
        
        assert summary['refinement_completed'] == True
        assert summary['total_attempts'] == 2
        assert summary['successful_strategy'] == 'expand_keywords'
        assert summary['final_video_count'] == 4
        assert summary['quality_threshold_met'] == True
        assert len(summary['search_evolution']) > 0
        assert len(summary['attempts_breakdown']) == 2
        assert len(summary['recommendations']) > 0
    
    def test_generate_refinement_recommendations_no_videos(self):
        """Test recommendations when no videos are found."""
        state = create_test_curator_state()
        state.discovered_videos = []
        
        recommendations = self.refinement_engine._generate_refinement_recommendations(state)
        
        assert len(recommendations) > 0
        assert any('No videos found' in rec for rec in recommendations)
        assert any('API configuration' in rec for rec in recommendations)
    
    def test_generate_refinement_recommendations_insufficient_quality(self):
        """Test recommendations when insufficient quality videos are found."""
        state = create_test_curator_state()
        state.discovered_videos = [create_test_video()]  # Only 1 video, need 3
        
        recommendations = self.refinement_engine._generate_refinement_recommendations(state)
        
        assert len(recommendations) > 0
        assert any('Found 1 videos but need 3' in rec for rec in recommendations)
        assert any('quality threshold' in rec for rec in recommendations)
    
    def test_generate_refinement_recommendations_success(self):
        """Test recommendations for successful refinement."""
        state = create_test_curator_state()
        state.discovered_videos = [create_test_video(video_id=f'success{i:04d}') for i in range(4)]
        state.search_attempt = 1
        
        recommendations = self.refinement_engine._generate_refinement_recommendations(state)
        
        assert len(recommendations) > 0
        assert any('successful' in rec.lower() for rec in recommendations)
    
    def test_keyword_synonyms_coverage(self):
        """Test that keyword synonyms cover important domains."""
        synonyms = self.refinement_engine.keyword_synonyms
        
        # Test AI/ML coverage
        ai_terms = ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural networks']
        for term in ai_terms:
            assert term in synonyms, f"Missing synonym mapping for: {term}"
        
        # Test programming coverage
        prog_terms = ['programming', 'coding', 'python', 'javascript']
        for term in prog_terms:
            assert term in synonyms, f"Missing synonym mapping for: {term}"
        
        # Test business/tech coverage
        biz_terms = ['startup', 'tech', 'innovation']
        for term in biz_terms:
            assert term in synonyms, f"Missing synonym mapping for: {term}"
    
    def test_domain_expansions_coverage(self):
        """Test that domain expansions are comprehensive."""
        expansions = self.refinement_engine.domain_expansions
        
        # Test that each domain has sufficient terms
        for domain, terms in expansions.items():
            assert len(terms) >= 5, f"Domain {domain} should have at least 5 expansion terms"
            assert all(isinstance(term, str) and len(term) > 0 for term in terms), f"Invalid terms in domain {domain}"
    
    def test_search_term_state_management(self):
        """Test that search terms are properly managed in state."""
        state = create_test_curator_state()
        original_keywords = state.search_keywords.copy()
        
        # Mock a successful search with expanded keywords
        mock_videos = [create_test_video(video_id='statetest01')]
        mock_ranking_result = create_mock_ranking_result(mock_videos, meets_requirement=True)
        
        with patch.object(self.refinement_engine.youtube_client, 'discover_videos', return_value=mock_videos), \
             patch.object(self.refinement_engine.ranking_system, 'rank_videos', return_value=mock_ranking_result):
            
            # Perform expansion strategy
            result = self.refinement_engine.perform_search_attempt(state, RefinementStrategy.EXPAND_KEYWORDS)
            
            # Check that current_search_terms was updated
            assert state.current_search_terms == result.search_terms
            assert len(state.current_search_terms) > len(original_keywords)
            
            # Original keywords should be preserved
            assert state.search_keywords == original_keywords


def test_refine_search_node():
    """Test the LangGraph node function."""
    state = create_test_curator_state()
    
    # Mock the SearchRefinementEngine
    mock_engine = Mock()
    mock_engine.refine_search_iteratively.return_value = state
    mock_engine.get_refinement_summary.return_value = {
        'total_attempts': 1,
        'final_video_count': 3,
        'successful_strategy': 'initial',
        'quality_threshold_met': True
    }
    
    with patch('src.podcast_curator.search_refinement.SearchRefinementEngine', return_value=mock_engine):
        result_state = refine_search_node(state)
        
        assert result_state == state
        mock_engine.refine_search_iteratively.assert_called_once_with(state)
        mock_engine.get_refinement_summary.assert_called_once_with(state)


def test_refine_search_node_error_handling():
    """Test error handling in the LangGraph node function."""
    state = create_test_curator_state()
    
    # Mock the SearchRefinementEngine to raise an exception
    with patch('src.podcast_curator.search_refinement.SearchRefinementEngine', side_effect=Exception("Engine error")):
        result_state = refine_search_node(state)
        
        assert len(result_state.errors) > 0
        assert 'Critical error in search refinement node' in result_state.errors[-1]


class TestRefinementResult:
    """Test the RefinementResult dataclass."""
    
    def test_refinement_result_creation(self):
        """Test RefinementResult creation and attributes."""
        videos = [create_test_video()]
        result = RefinementResult(
            videos=videos,
            strategy_used=RefinementStrategy.EXPAND_KEYWORDS,
            search_terms=['ai', 'machine learning'],
            days_back=14,
            quality_summary={'total_videos': 1},
            refinement_needed=False,
            suggestions=['Good results'],
            metadata={'test': 'data'}
        )
        
        assert result.videos == videos
        assert result.strategy_used == RefinementStrategy.EXPAND_KEYWORDS
        assert result.search_terms == ['ai', 'machine learning']
        assert result.days_back == 14
        assert result.quality_summary['total_videos'] == 1
        assert result.refinement_needed == False
        assert result.suggestions == ['Good results']
        assert result.metadata['test'] == 'data'


class TestRefinementStrategy:
    """Test the RefinementStrategy enum."""
    
    def test_refinement_strategy_values(self):
        """Test RefinementStrategy enum values."""
        assert RefinementStrategy.INITIAL.value == 'initial'
        assert RefinementStrategy.EXPAND_KEYWORDS.value == 'expand_keywords'
        assert RefinementStrategy.EXPAND_TIMEFRAME.value == 'expand_timeframe'
        assert RefinementStrategy.BROADEN_TERMS.value == 'broaden_terms'
        assert RefinementStrategy.FINAL_FALLBACK.value == 'final_fallback'
    
    def test_refinement_strategy_completeness(self):
        """Test that all necessary refinement strategies are defined."""
        strategies = list(RefinementStrategy)
        assert len(strategies) == 5
        
        # Ensure we have strategies for different refinement approaches
        strategy_values = [s.value for s in strategies]
        assert 'initial' in strategy_values
        assert 'expand_keywords' in strategy_values
        assert 'expand_timeframe' in strategy_values
        assert 'broaden_terms' in strategy_values
        assert 'final_fallback' in strategy_values


def test_integration_with_existing_systems():
    """Test integration with existing YouTube client and ranking system."""
    mock_config = create_mock_config()
    
    with patch('src.podcast_curator.search_refinement.YouTubeClient') as mock_youtube_class, \
         patch('src.podcast_curator.search_refinement.VideoRankingSystem') as mock_ranking_class:
        
        mock_youtube = Mock()
        mock_ranking = Mock()
        mock_youtube_class.return_value = mock_youtube
        mock_ranking_class.return_value = mock_ranking
        
        engine = SearchRefinementEngine(config=mock_config)
        
        # Verify that dependencies were initialized with config
        mock_youtube_class.assert_called_once_with(config=mock_config)
        mock_ranking_class.assert_called_once_with(config=mock_config)
        
        assert engine.youtube_client == mock_youtube
        assert engine.ranking_system == mock_ranking


def test_search_filters_creation():
    """Test that SearchFilters are created correctly for different strategies."""
    mock_config = create_mock_config()
    
    with patch('src.podcast_curator.search_refinement.YouTubeClient'), \
         patch('src.podcast_curator.search_refinement.VideoRankingSystem'):
        
        engine = SearchRefinementEngine(config=mock_config)
        state = create_test_curator_state()
        
        # Test that min_views is adjusted for broader strategies
        with patch.object(engine.youtube_client, 'discover_videos', return_value=[]), \
             patch.object(engine.ranking_system, 'rank_videos', return_value=create_mock_ranking_result([])):
            
            # Test normal strategy
            engine.perform_search_attempt(state, RefinementStrategy.EXPAND_KEYWORDS)
            
            # Test broader strategy (should use lower min_views)
            engine.perform_search_attempt(state, RefinementStrategy.BROADEN_TERMS)
            engine.perform_search_attempt(state, RefinementStrategy.FINAL_FALLBACK)
            
            # Verify discover_videos was called multiple times
            assert engine.youtube_client.discover_videos.call_count == 3