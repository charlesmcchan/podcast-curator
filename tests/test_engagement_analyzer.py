"""
Tests for the engagement analyzer module.
"""
from unittest.mock import Mock, patch
from src.nanook_curator.engagement_analyzer import EngagementAnalyzer, EngagementMetrics
from src.nanook_curator.models import VideoData
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
        'upload_date': '2024-01-15T10:00:00Z'
    }
    defaults.update(kwargs)
    return VideoData(**defaults)


class TestEngagementAnalyzer:
    """Test cases for EngagementAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        mock_config = create_mock_config()
        self.analyzer = EngagementAnalyzer(config=mock_config)
        
    def test_init(self):
        """Test analyzer initialization."""
        assert self.analyzer.like_ratio_threshold == 0.8
        assert len(self.analyzer.positive_keywords) > 10
        assert len(self.analyzer.negative_keywords) > 10
        
    def test_calculate_like_to_dislike_ratio_basic(self):
        """Test basic like-to-dislike ratio calculation."""
        video = create_test_video(like_count=800, view_count=10000)
        
        like_ratio, ltd_ratio = self.analyzer.calculate_like_to_dislike_ratio(video)
        
        # Should return reasonable ratios
        assert 0.0 <= like_ratio <= 1.0
        assert ltd_ratio >= 0.0
        
        # Like ratio should be reasonably high for good engagement
        assert like_ratio > 0.8
        
    def test_calculate_like_to_dislike_ratio_zero_likes(self):
        """Test like-to-dislike ratio with zero likes."""
        video = create_test_video(like_count=0, view_count=10000)
        
        like_ratio, ltd_ratio = self.analyzer.calculate_like_to_dislike_ratio(video)
        
        assert like_ratio == 0.0
        assert ltd_ratio == 0.0
        
    def test_calculate_like_to_dislike_ratio_with_enhanced_metrics(self):
        """Test like-to-dislike ratio with enhanced metrics available."""
        video = create_test_video(like_count=800, view_count=10000)
        video.engagement_metrics = {'likeRatio': 0.92}
        
        like_ratio, ltd_ratio = self.analyzer.calculate_like_to_dislike_ratio(video)
        
        assert like_ratio == 0.92
        assert ltd_ratio > 10.0  # Should be high for 92% like ratio
        
    def test_analyze_comment_sentiment_empty(self):
        """Test comment sentiment analysis with empty comments."""
        result = self.analyzer.analyze_comment_sentiment([])
        
        assert result['sentiment_score'] == 0.0
        assert result['total_analyzed'] == 0
        assert result['positive_count'] == 0
        assert result['negative_count'] == 0
        assert result['neutral_count'] == 0
        
    def test_analyze_comment_sentiment_positive(self):
        """Test comment sentiment analysis with positive comments."""
        comments = [
            "This is amazing! Great explanation!",
            "Excellent content, very helpful",
            "Thanks for this wonderful tutorial"
        ]
        
        result = self.analyzer.analyze_comment_sentiment(comments)
        
        assert result['sentiment_score'] > 0.0
        assert result['positive_count'] > 0
        assert result['total_analyzed'] == 3
        assert result['sentiment_distribution']['positive'] > 0.5
        
    def test_analyze_comment_sentiment_negative(self):
        """Test comment sentiment analysis with negative comments."""
        comments = [
            "This is terrible and boring",
            "Awful content, waste of time",
            "Horrible explanation, very confusing"
        ]
        
        result = self.analyzer.analyze_comment_sentiment(comments)
        
        assert result['sentiment_score'] < 0.0
        assert result['negative_count'] > 0
        assert result['total_analyzed'] == 3
        assert result['sentiment_distribution']['negative'] > 0.5
        
    def test_analyze_comment_sentiment_mixed(self):
        """Test comment sentiment analysis with mixed comments."""
        comments = [
            "This is great and helpful!",
            "This is terrible and boring",
            "Just a normal comment without sentiment",
            "Excellent tutorial, learned a lot",
            "Bad explanation, very unclear"
        ]
        
        result = self.analyzer.analyze_comment_sentiment(comments)
        
        assert -1.0 <= result['sentiment_score'] <= 1.0
        assert result['total_analyzed'] == 5
        assert result['positive_count'] + result['negative_count'] + result['neutral_count'] == 5
        
    def test_calculate_view_to_subscriber_ratio_with_enhanced_metrics(self):
        """Test view-to-subscriber ratio with enhanced metrics."""
        video = create_test_video(view_count=10000)
        video.engagement_metrics = {'viewToSubscriberRatio': 0.25}
        
        ratio = self.analyzer.calculate_view_to_subscriber_ratio(video)
        
        assert ratio == 0.25
        
    def test_calculate_view_to_subscriber_ratio_with_subscriber_count(self):
        """Test view-to-subscriber ratio with subscriber count."""
        video = create_test_video(view_count=10000)
        video.channel_subscriber_count = 50000
        
        ratio = self.analyzer.calculate_view_to_subscriber_ratio(video)
        
        assert ratio == 0.2  # 10000 / 50000
        
    def test_calculate_view_to_subscriber_ratio_no_subscribers(self):
        """Test view-to-subscriber ratio with no subscriber data."""
        video = create_test_video(view_count=10000)
        
        ratio = self.analyzer.calculate_view_to_subscriber_ratio(video)
        
        assert ratio == 0.0
        
    def test_calculate_overall_engagement_score(self):
        """Test overall engagement score calculation."""
        video = create_test_video(
            view_count=10000,
            like_count=800,
            comment_count=50
        )
        
        score = self.analyzer.calculate_overall_engagement_score(video)
        
        assert 0.0 <= score <= 100.0
        assert score > 50.0  # Should be decent for good engagement
        
    def test_calculate_overall_engagement_score_with_sentiment(self):
        """Test overall engagement score with comment sentiment."""
        video = create_test_video(
            view_count=10000,
            like_count=800,
            comment_count=50
        )
        
        comment_sentiment = {
            'sentiment_score': 0.6,
            'positive_count': 8,
            'negative_count': 2,
            'neutral_count': 0,
            'total_analyzed': 10
        }
        
        score = self.analyzer.calculate_overall_engagement_score(video, comment_sentiment)
        
        assert 0.0 <= score <= 100.0
        assert score > 60.0  # Should be higher with positive sentiment
        
    def test_analyze_video_engagement_complete(self):
        """Test complete video engagement analysis."""
        video = create_test_video(
            view_count=10000,
            like_count=900,
            comment_count=100
        )
        
        comments = [
            "Amazing content!",
            "Very helpful tutorial",
            "Great explanation"
        ]
        
        metrics = self.analyzer.analyze_video_engagement(video, comments)
        
        assert isinstance(metrics, EngagementMetrics)
        assert metrics.like_ratio > 0.0
        assert metrics.overall_engagement_score > 0.0
        assert metrics.comment_sentiment_score > 0.0
        assert 'like_count' in metrics.detailed_metrics
        
    def test_analyze_video_engagement_meets_threshold(self):
        """Test video engagement analysis meeting threshold."""
        video = create_test_video(
            view_count=10000,
            like_count=950,  # Very high like count
            comment_count=100
        )
        
        metrics = self.analyzer.analyze_video_engagement(video)
        
        assert metrics.meets_threshold is True
        assert metrics.like_ratio >= 0.8
        
    def test_analyze_video_engagement_below_threshold(self):
        """Test video engagement analysis below threshold."""
        video = create_test_video(
            view_count=10000,
            like_count=100,  # Low like count
            comment_count=10
        )
        
        # Mock the like ratio calculation to return below threshold
        with patch.object(self.analyzer, 'calculate_like_to_dislike_ratio', return_value=(0.7, 2.33)):
            metrics = self.analyzer.analyze_video_engagement(video)
            
            assert metrics.meets_threshold is False
            assert metrics.like_ratio < 0.8
        
    def test_analyze_video_engagement_error_handling(self):
        """Test engagement analysis error handling."""
        video = create_test_video()
        
        # Mock an error in one of the calculation methods
        with patch.object(self.analyzer, 'calculate_like_to_dislike_ratio', side_effect=Exception("Test error")):
            metrics = self.analyzer.analyze_video_engagement(video)
            
            # Should return minimal metrics with error info
            assert metrics.overall_engagement_score == 0.0
            assert 'error' in metrics.detailed_metrics
            
    def test_batch_analyze_engagement(self):
        """Test batch engagement analysis."""
        videos = [
            create_test_video(video_id='video123456', like_count=800),
            create_test_video(video_id='video234567', like_count=600),
            create_test_video(video_id='video345678', like_count=900)
        ]
        
        video_comments = {
            'video123456': ['Great video!', 'Very helpful'],
            'video234567': ['Okay content'],
            'video345678': ['Amazing!', 'Excellent work', 'Perfect explanation']
        }
        
        results = self.analyzer.batch_analyze_engagement(videos, video_comments)
        
        assert len(results) == 3
        assert 'video123456' in results
        assert 'video234567' in results
        assert 'video345678' in results
        
        # All should have valid metrics
        for video_id, metrics in results.items():
            assert isinstance(metrics, EngagementMetrics)
            assert metrics.overall_engagement_score >= 0.0
        
    def test_filter_videos_by_engagement(self):
        """Test filtering videos by engagement criteria."""
        videos = [
            create_test_video(video_id='highEngaged', like_count=950, comment_count=100),
            create_test_video(video_id='medEngaged1', like_count=700, comment_count=50),
            create_test_video(video_id='lowEngaged1', like_count=100, comment_count=5),
        ]
        
        filtered = self.analyzer.filter_videos_by_engagement(
            videos, 
            min_engagement_score=50.0, 
            require_threshold=False
        )
        
        # Should filter based on engagement score
        assert len(filtered) >= 1  # At least the high engagement video
        assert filtered[0].video_id in ['highEngaged', 'medEngaged1']
        
        # Check that engagement analysis is attached
        for video in filtered:
            assert hasattr(video, 'engagement_analysis')
            assert isinstance(video.engagement_analysis, EngagementMetrics)
        
    def test_filter_videos_by_engagement_with_threshold(self):
        """Test filtering videos requiring threshold."""
        videos = [
            create_test_video(video_id='meetsThresh', like_count=950),
            create_test_video(video_id='belowThresh', like_count=200)
        ]
        
        # Mock threshold results
        def mock_analyze(video, comments=None):
            if video.video_id == 'meetsThresh':
                return EngagementMetrics(
                    like_ratio=0.85, like_to_dislike_ratio=5.67, view_to_subscriber_ratio=0.2,
                    comment_sentiment_score=0.0, engagement_rate=0.085, overall_engagement_score=75.0,
                    meets_threshold=True, detailed_metrics={}
                )
            else:
                return EngagementMetrics(
                    like_ratio=0.7, like_to_dislike_ratio=2.33, view_to_subscriber_ratio=0.2,
                    comment_sentiment_score=0.0, engagement_rate=0.02, overall_engagement_score=45.0,
                    meets_threshold=False, detailed_metrics={}
                )
        
        with patch.object(self.analyzer, 'analyze_video_engagement', side_effect=mock_analyze):
            filtered = self.analyzer.filter_videos_by_engagement(
                videos, 
                min_engagement_score=40.0, 
                require_threshold=True
            )
            
            assert len(filtered) == 1
            assert filtered[0].video_id == 'meetsThresh'
        
    def test_get_engagement_summary_empty(self):
        """Test engagement summary with empty video list."""
        summary = self.analyzer.get_engagement_summary([])
        
        assert summary['total_videos'] == 0
        assert summary['avg_engagement_score'] == 0.0
        assert summary['threshold_rate'] == 0.0
        
    def test_get_engagement_summary_with_videos(self):
        """Test engagement summary with videos."""
        videos = [
            create_test_video(video_id='video123456', like_count=900),
            create_test_video(video_id='video234567', like_count=800),
            create_test_video(video_id='video345678', like_count=200)
        ]
        
        summary = self.analyzer.get_engagement_summary(videos)
        
        assert summary['total_videos'] == 3
        assert summary['avg_engagement_score'] > 0.0
        assert summary['avg_like_ratio'] > 0.0
        assert 0.0 <= summary['threshold_rate'] <= 1.0
        
    def test_threshold_configuration(self):
        """Test that threshold is configurable."""
        # Test with different threshold
        analyzer = EngagementAnalyzer(config=create_mock_config())
        analyzer.like_ratio_threshold = 0.9  # Higher threshold
        
        video = create_test_video(like_count=850)  # 85% like ratio
        
        # Mock like ratio calculation
        with patch.object(analyzer, 'calculate_like_to_dislike_ratio', return_value=(0.85, 5.67)):
            metrics = analyzer.analyze_video_engagement(video)
            assert metrics.meets_threshold is False  # Should not meet 90% threshold
        
        # Test with lower threshold
        analyzer.like_ratio_threshold = 0.7
        with patch.object(analyzer, 'calculate_like_to_dislike_ratio', return_value=(0.85, 5.67)):
            metrics = analyzer.analyze_video_engagement(video)
            assert metrics.meets_threshold is True  # Should meet 70% threshold
        
    def test_sentiment_keyword_detection(self):
        """Test sentiment keyword detection accuracy."""
        # Test positive keywords
        positive_comment = "This is absolutely amazing and excellent work!"
        result = self.analyzer.analyze_comment_sentiment([positive_comment])
        assert result['positive_count'] == 1
        assert result['sentiment_score'] > 0
        
        # Test negative keywords
        negative_comment = "This is terrible and awful content, worst video ever!"
        result = self.analyzer.analyze_comment_sentiment([negative_comment])
        assert result['negative_count'] == 1
        assert result['sentiment_score'] < 0
        
        # Test neutral comment
        neutral_comment = "This is a video about machine learning algorithms."
        result = self.analyzer.analyze_comment_sentiment([neutral_comment])
        assert result['neutral_count'] == 1
        assert result['sentiment_score'] == 0.0
        
    def test_engagement_metrics_dataclass(self):
        """Test EngagementMetrics dataclass functionality."""
        metrics = EngagementMetrics(
            like_ratio=0.85,
            like_to_dislike_ratio=5.67,
            view_to_subscriber_ratio=0.2,
            comment_sentiment_score=0.4,
            engagement_rate=0.075,
            overall_engagement_score=78.5,
            meets_threshold=True,
            detailed_metrics={'test': 'data'}
        )
        
        assert metrics.like_ratio == 0.85
        assert metrics.meets_threshold is True
        assert metrics.detailed_metrics['test'] == 'data'