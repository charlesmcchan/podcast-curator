"""
Tests for YouTube API client wrapper.

This module contains unit tests for the YouTubeClient class,
including mock API responses and error handling scenarios.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from googleapiclient.errors import HttpError
from google.auth.exceptions import GoogleAuthError

from src.podcast_curator.youtube_client import (
    YouTubeClient, SearchFilters, RateLimitError, 
    AuthenticationError, YouTubeAPIError
)
from src.podcast_curator.config import Configuration
from src.podcast_curator.models import VideoData


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock(spec=Configuration)
    config.youtube_api_key = "test_api_key_12345678901234567890"
    config.openai_api_key = "sk-test_openai_key"
    config.max_videos = 10
    config.days_back = 7
    config.quality_threshold = 80.0
    config.min_quality_videos = 3
    return config


@pytest.fixture
def search_filters():
    """Create test search filters."""
    return SearchFilters(
        keywords=["AI news", "machine learning"],
        days_back=7,
        max_results=10,
        min_views=1000
    )


@pytest.fixture
def mock_youtube_service():
    """Create a mock YouTube service."""
    service = Mock()
    
    # Mock search response
    search_response = {
        'items': [
            {
                'id': {'videoId': 'dQw4w9WgXcQ'},
                'snippet': {
                    'title': 'Test AI Video 1',
                    'channelId': 'test_channel_1',
                    'channelTitle': 'Test Channel 1',
                    'publishedAt': '2024-01-01T12:00:00Z'
                }
            },
            {
                'id': {'videoId': 'jNQXAC9IVRw'},
                'snippet': {
                    'title': 'Test AI Video 2',
                    'channelId': 'test_channel_2',
                    'channelTitle': 'Test Channel 2',
                    'publishedAt': '2024-01-01T13:00:00Z'
                }
            }
        ]
    }
    
    # Mock video details response
    videos_response = {
        'items': [
            {
                'id': 'dQw4w9WgXcQ',
                'snippet': {
                    'title': 'Test AI Video 1',
                    'channelId': 'test_channel_1',
                    'channelTitle': 'Test Channel 1',
                    'publishedAt': '2024-01-01T12:00:00Z'
                },
                'statistics': {
                    'viewCount': '5000',
                    'likeCount': '100',
                    'commentCount': '20'
                }
            },
            {
                'id': 'jNQXAC9IVRw',
                'snippet': {
                    'title': 'Test AI Video 2',
                    'channelId': 'test_channel_2',
                    'channelTitle': 'Test Channel 2',
                    'publishedAt': '2024-01-01T13:00:00Z'
                },
                'statistics': {
                    'viewCount': '10000',
                    'likeCount': '200',
                    'commentCount': '50'
                }
            }
        ]
    }
    
    # Mock channel response
    channels_response = {
        'items': [
            {
                'id': 'test_channel_1',
                'snippet': {
                    'title': 'Test Channel 1'
                },
                'statistics': {
                    'subscriberCount': '1000'
                }
            },
            {
                'id': 'test_channel_2',
                'snippet': {
                    'title': 'Test Channel 2'
                },
                'statistics': {
                    'subscriberCount': '5000'
                }
            }
        ]
    }
    
    # Configure mock methods
    service.search().list().execute.return_value = search_response
    service.videos().list().execute.return_value = videos_response
    service.channels().list().execute.return_value = channels_response
    
    return service


class TestYouTubeClient:
    """Test cases for YouTubeClient class."""
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_client_initialization_success(self, mock_build, mock_config):
        """Test successful client initialization."""
        mock_build.return_value = Mock()
        
        client = YouTubeClient(mock_config)
        
        assert client.config == mock_config
        assert client.api_key == mock_config.youtube_api_key
        assert client.quota_used == 0
        mock_build.assert_called_once_with('youtube', 'v3', developerKey=mock_config.youtube_api_key)
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_client_initialization_auth_error(self, mock_build, mock_config):
        """Test client initialization with authentication error."""
        mock_build.side_effect = GoogleAuthError("Invalid API key")
        
        with pytest.raises(AuthenticationError, match="YouTube API authentication failed"):
            YouTubeClient(mock_config)
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_search_videos_success(self, mock_build, mock_config, search_filters, mock_youtube_service):
        """Test successful video search."""
        mock_build.return_value = mock_youtube_service
        client = YouTubeClient(mock_config)
        
        results = client.search_videos(search_filters)
        
        assert len(results) == 2
        assert results[0]['id']['videoId'] == 'dQw4w9WgXcQ'
        assert results[1]['id']['videoId'] == 'jNQXAC9IVRw'
        assert client.quota_used == 100  # Search operation costs 100 quota
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_get_video_details_success(self, mock_build, mock_config, mock_youtube_service):
        """Test successful video details fetching."""
        mock_build.return_value = mock_youtube_service
        client = YouTubeClient(mock_config)
        
        video_ids = ['dQw4w9WgXcQ', 'jNQXAC9IVRw']
        results = client.get_video_details(video_ids)
        
        assert len(results) == 2
        assert results[0]['id'] == 'dQw4w9WgXcQ'
        assert results[1]['id'] == 'jNQXAC9IVRw'
        assert client.quota_used == 1  # Videos operation costs 1 quota
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_get_channel_info_success(self, mock_build, mock_config, mock_youtube_service):
        """Test successful channel info fetching."""
        mock_build.return_value = mock_youtube_service
        client = YouTubeClient(mock_config)
        
        channel_ids = ['test_channel_1', 'test_channel_2']
        results = client.get_channel_info(channel_ids)
        
        assert len(results) == 2
        assert 'test_channel_1' in results
        assert 'test_channel_2' in results
        assert client.quota_used == 1  # Channels operation costs 1 quota
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_quota_limit_enforcement(self, mock_build, mock_config, search_filters, mock_youtube_service):
        """Test quota limit enforcement."""
        mock_build.return_value = mock_youtube_service
        client = YouTubeClient(mock_config)
        client.quota_limit = 50  # Set low limit for testing
        
        with pytest.raises(RateLimitError, match="Quota limit would be exceeded"):
            client.search_videos(search_filters)
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_http_error_handling(self, mock_build, mock_config, search_filters):
        """Test HTTP error handling with retry logic."""
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        # Create mock HTTP error
        error_response = Mock()
        error_response.status = 500
        http_error = HttpError(error_response, b'Server Error')
        http_error.error_details = [{'reason': 'internalServerError'}]
        
        mock_service.search().list().execute.side_effect = http_error
        
        client = YouTubeClient(mock_config)
        
        with pytest.raises(YouTubeAPIError, match="All retry attempts failed"):
            client.search_videos(search_filters)
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_rate_limit_error_handling(self, mock_build, mock_config, search_filters):
        """Test rate limit error handling."""
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        # Create mock rate limit error
        error_response = Mock()
        error_response.status = 403
        http_error = HttpError(error_response, b'Quota Exceeded')
        http_error.error_details = [{'reason': 'quotaExceeded'}]
        
        mock_service.search().list().execute.side_effect = http_error
        
        client = YouTubeClient(mock_config)
        
        with pytest.raises(RateLimitError, match="YouTube API quota exceeded"):
            client.search_videos(search_filters)
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_discover_trending_videos_success(self, mock_build, mock_config, search_filters, mock_youtube_service):
        """Test successful trending video discovery."""
        mock_build.return_value = mock_youtube_service
        client = YouTubeClient(mock_config)
        
        # Mock recent upload dates
        recent_date = (datetime.now() - timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%SZ')
        mock_youtube_service.search().list().execute.return_value['items'][0]['snippet']['publishedAt'] = recent_date
        mock_youtube_service.search().list().execute.return_value['items'][1]['snippet']['publishedAt'] = recent_date
        mock_youtube_service.videos().list().execute.return_value['items'][0]['snippet']['publishedAt'] = recent_date
        mock_youtube_service.videos().list().execute.return_value['items'][1]['snippet']['publishedAt'] = recent_date
        
        trending_videos = client.discover_trending_videos(search_filters)
        
        assert len(trending_videos) == 2
        assert all(isinstance(video, VideoData) for video in trending_videos)
        assert all(video.view_count >= search_filters.min_views for video in trending_videos)
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_convert_to_video_data(self, mock_build, mock_config):
        """Test conversion of API response to VideoData object."""
        mock_build.return_value = Mock()
        client = YouTubeClient(mock_config)
        
        api_video = {
            'id': 'dQw4w9WgXcQ',
            'snippet': {
                'title': 'Test Video',
                'channelId': 'test_channel_1',
                'channelTitle': 'Test Channel',
                'publishedAt': '2024-01-01T12:00:00Z'
            },
            'statistics': {
                'viewCount': '5000',
                'likeCount': '100',
                'commentCount': '20'
            }
        }
        
        channel_info = {
            'test_channel_1': {
                'snippet': {'title': 'Test Channel'},
                'statistics': {'subscriberCount': '1000'}
            }
        }
        
        video_data = client._convert_to_video_data(api_video, channel_info)
        
        assert video_data.video_id == 'dQw4w9WgXcQ'
        assert video_data.title == 'Test Video'
        assert video_data.channel == 'Test Channel'
        assert video_data.view_count == 5000
        assert video_data.like_count == 100
        assert video_data.comment_count == 20
        assert video_data.upload_date == '2024-01-01T12:00:00Z'
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_trending_score_calculation(self, mock_build, mock_config):
        """Test trending score calculation."""
        mock_build.return_value = Mock()
        client = YouTubeClient(mock_config)
        
        # Create test video with recent upload date
        recent_date = (datetime.now() - timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%SZ')
        video = VideoData(
            video_id='dQw4w9WgXcQ',
            title='Test Video',
            channel='Test Channel',
            view_count=50000,
            like_count=1000,
            comment_count=100,
            upload_date=recent_date
        )
        
        score = client._calculate_trending_score(video)
        
        assert score > 0
        assert isinstance(score, float)
    
    def test_search_filters_creation(self):
        """Test SearchFilters dataclass creation."""
        filters = SearchFilters(
            keywords=["AI", "machine learning"],
            days_back=14,
            max_results=20,
            min_views=5000
        )
        
        assert filters.keywords == ["AI", "machine learning"]
        assert filters.days_back == 14
        assert filters.max_results == 20
        assert filters.min_views == 5000
        assert filters.order == "relevance"  # Default value
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_quota_usage_tracking(self, mock_build, mock_config):
        """Test quota usage tracking."""
        mock_build.return_value = Mock()
        client = YouTubeClient(mock_config)
        
        initial_used, initial_limit = client.get_quota_usage()
        assert initial_used == 0
        assert initial_limit == client.DEFAULT_QUOTA_LIMIT
        
        # Simulate quota usage
        client._update_quota('search')
        client._update_quota('videos')
        
        used, limit = client.get_quota_usage()
        assert used == 101  # 100 for search + 1 for videos
        
        # Test quota reset
        client.reset_quota()
        used, limit = client.get_quota_usage()
        assert used == 0
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_custom_quota_limit(self, mock_build, mock_config):
        """Test setting custom quota limit."""
        mock_build.return_value = Mock()
        client = YouTubeClient(mock_config)
        
        client.set_quota_limit(5000)
        used, limit = client.get_quota_usage()
        assert limit == 5000
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_discover_videos_success(self, mock_build, mock_config, mock_youtube_service):
        """Test the main video discovery function."""
        mock_build.return_value = mock_youtube_service
        client = YouTubeClient(mock_config)
        
        # Mock recent upload dates for weekly focus
        recent_date = (datetime.now() - timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%SZ')
        mock_youtube_service.search().list().execute.return_value['items'][0]['snippet']['publishedAt'] = recent_date
        mock_youtube_service.search().list().execute.return_value['items'][1]['snippet']['publishedAt'] = recent_date
        mock_youtube_service.videos().list().execute.return_value['items'][0]['snippet']['publishedAt'] = recent_date
        mock_youtube_service.videos().list().execute.return_value['items'][1]['snippet']['publishedAt'] = recent_date
        
        keywords = ["AI news", "machine learning"]
        discovered_videos = client.discover_videos(keywords, max_videos=5, days_back=7)
        
        assert len(discovered_videos) <= 5
        assert all(isinstance(video, VideoData) for video in discovered_videos)
        assert all(video.view_count >= 1000 for video in discovered_videos)  # Min view requirement
        
        # Check that videos are sorted by trending score (recalculate for verification)
        if len(discovered_videos) > 1:
            scores = [client._calculate_trending_score(video) for video in discovered_videos]
            assert scores == sorted(scores, reverse=True)
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_discover_videos_no_results(self, mock_build, mock_config):
        """Test video discovery with no search results."""
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        # Mock empty search response
        mock_service.search().list().execute.return_value = {'items': []}
        
        client = YouTubeClient(mock_config)
        keywords = ["very specific nonexistent topic"]
        discovered_videos = client.discover_videos(keywords)
        
        assert len(discovered_videos) == 0
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_meets_discovery_criteria(self, mock_build, mock_config):
        """Test the discovery criteria filtering."""
        mock_build.return_value = Mock()
        client = YouTubeClient(mock_config)
        
        # Test video that meets all criteria
        recent_date = (datetime.now() - timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%SZ')
        good_video = VideoData(
            video_id='dQw4w9WgXcQ',  # Valid 11-char YouTube ID
            title='Good AI Video',
            channel='Test Channel',
            view_count=5000,  # Above 1000 minimum
            like_count=100,
            comment_count=20,
            upload_date=recent_date  # Recent
        )
        
        assert client._meets_discovery_criteria(good_video, days_back=7) is True
        
        # Test video with insufficient views
        low_views_video = VideoData(
            video_id='jNQXAC9IVRw',  # Valid 11-char YouTube ID
            title='Low Views Video',
            channel='Test Channel',
            view_count=500,  # Below 1000 minimum
            like_count=10,
            comment_count=2,
            upload_date=recent_date
        )
        
        assert client._meets_discovery_criteria(low_views_video, days_back=7) is False
        
        # Test old video
        old_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%dT%H:%M:%SZ')
        old_video = VideoData(
            video_id='M7lc1UVf-VE',  # Valid 11-char YouTube ID
            title='Old Video',
            channel='Test Channel',
            view_count=5000,
            like_count=100,
            comment_count=20,
            upload_date=old_date  # Too old
        )
        
        assert client._meets_discovery_criteria(old_video, days_back=7) is False
        
        # Test video with no engagement
        no_engagement_video = VideoData(
            video_id='9bZkp7q19f0',  # Valid 11-char YouTube ID
            title='No Engagement Video',
            channel='Test Channel',
            view_count=5000,
            like_count=0,  # No likes
            comment_count=0,  # No comments
            upload_date=recent_date
        )
        
        assert client._meets_discovery_criteria(no_engagement_video, days_back=7) is False
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_enhanced_trending_score_calculation(self, mock_build, mock_config):
        """Test the enhanced trending score calculation with multiple factors."""
        mock_build.return_value = Mock()
        client = YouTubeClient(mock_config)
        
        # Test high-trending video
        recent_date = (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%SZ')
        trending_video = VideoData(
            video_id='dQw4w9WgXcQ',  # Valid 11-char YouTube ID
            title='Trending AI Video',
            channel='Popular Channel',
            view_count=100000,  # High views
            like_count=5000,    # Good engagement
            comment_count=500,  # Active discussion
            upload_date=recent_date  # Very recent
        )
        
        trending_score = client._calculate_trending_score(trending_video)
        
        # Test moderate video
        moderate_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%dT%H:%M:%SZ')
        moderate_video = VideoData(
            video_id='jNQXAC9IVRw',  # Valid 11-char YouTube ID
            title='Moderate AI Video',
            channel='Regular Channel',
            view_count=10000,
            like_count=200,
            comment_count=50,
            upload_date=moderate_date
        )
        
        moderate_score = client._calculate_trending_score(moderate_video)
        
        # Trending video should have higher score
        assert trending_score > moderate_score
        assert trending_score > 0
        assert moderate_score > 0
    
    @patch('src.podcast_curator.youtube_client.build')
    def test_discover_videos_with_filtering(self, mock_build, mock_config):
        """Test video discovery with mixed quality videos to verify filtering."""
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        # Create mixed search results - some good, some filtered out
        recent_date = (datetime.now() - timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%SZ')
        old_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        search_response = {
            'items': [
                {
                    'id': {'videoId': 'dQw4w9WgXcQ'},  # Valid 11-char YouTube ID
                    'snippet': {
                        'title': 'Good AI Video',
                        'channelId': 'channel_1',
                        'channelTitle': 'Channel 1',
                        'publishedAt': recent_date
                    }
                },
                {
                    'id': {'videoId': 'jNQXAC9IVRw'},  # Valid 11-char YouTube ID
                    'snippet': {
                        'title': 'Old AI Video',
                        'channelId': 'channel_2',
                        'channelTitle': 'Channel 2',
                        'publishedAt': old_date  # Too old
                    }
                },
                {
                    'id': {'videoId': 'M7lc1UVf-VE'},  # Valid 11-char YouTube ID
                    'snippet': {
                        'title': 'Low Views AI Video',
                        'channelId': 'channel_3',
                        'channelTitle': 'Channel 3',
                        'publishedAt': recent_date
                    }
                }
            ]
        }
        
        videos_response = {
            'items': [
                {
                    'id': 'dQw4w9WgXcQ',
                    'snippet': {
                        'title': 'Good AI Video',
                        'channelId': 'channel_1',
                        'channelTitle': 'Channel 1',
                        'publishedAt': recent_date
                    },
                    'statistics': {
                        'viewCount': '5000',  # Good views
                        'likeCount': '100',
                        'commentCount': '20'
                    }
                },
                {
                    'id': 'jNQXAC9IVRw',
                    'snippet': {
                        'title': 'Old AI Video',
                        'channelId': 'channel_2',
                        'channelTitle': 'Channel 2',
                        'publishedAt': old_date
                    },
                    'statistics': {
                        'viewCount': '10000',
                        'likeCount': '200',
                        'commentCount': '50'
                    }
                },
                {
                    'id': 'M7lc1UVf-VE',
                    'snippet': {
                        'title': 'Low Views AI Video',
                        'channelId': 'channel_3',
                        'channelTitle': 'Channel 3',
                        'publishedAt': recent_date
                    },
                    'statistics': {
                        'viewCount': '500',  # Too few views
                        'likeCount': '10',
                        'commentCount': '2'
                    }
                }
            ]
        }
        
        channels_response = {
            'items': [
                {'id': 'channel_1', 'snippet': {'title': 'Channel 1'}},
                {'id': 'channel_2', 'snippet': {'title': 'Channel 2'}},
                {'id': 'channel_3', 'snippet': {'title': 'Channel 3'}}
            ]
        }
        
        mock_service.search().list().execute.return_value = search_response
        mock_service.videos().list().execute.return_value = videos_response
        mock_service.channels().list().execute.return_value = channels_response
        
        client = YouTubeClient(mock_config)
        keywords = ["AI news"]
        discovered_videos = client.discover_videos(keywords, max_videos=10, days_back=7)
        
        # Should only return the good video (others filtered out)
        assert len(discovered_videos) == 1
        assert discovered_videos[0].video_id == 'dQw4w9WgXcQ'
        assert discovered_videos[0].view_count >= 1000


if __name__ == '__main__':
    pytest.main([__file__])