"""
YouTube Data API client wrapper for nanook-curator.

This module provides a comprehensive wrapper around the YouTube Data API v3
with authentication, rate limiting, error handling, and retry logic.
Implements video search with date filtering and trending indicators.
"""

import time
import logging
import math
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import random

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.exceptions import GoogleAuthError

from .models import VideoData
from .config import Configuration


logger = logging.getLogger(__name__)


@dataclass
class SearchFilters:
    """Search filters for YouTube video discovery."""
    keywords: List[str]
    days_back: int = 7
    max_results: int = 10
    min_views: int = 1000
    order: str = "relevance"  # relevance, date, rating, viewCount, title
    video_duration: str = "any"  # any, short, medium, long
    video_definition: str = "any"  # any, high, standard


class RateLimitError(Exception):
    """Raised when API rate limit is exceeded."""
    pass


class AuthenticationError(Exception):
    """Raised when API authentication fails."""
    pass


class YouTubeAPIError(Exception):
    """Base exception for YouTube API related errors."""
    pass


class YouTubeClient:
    """
    YouTube Data API v3 client with rate limiting and error handling.
    
    Provides methods for video search, details fetching, and trending analysis
    with automatic retry logic and exponential backoff.
    """
    
    # API quota costs (per YouTube API documentation)
    QUOTA_COSTS = {
        'search': 100,
        'videos': 1,
        'channels': 1,
    }
    
    # Rate limiting settings
    DEFAULT_QUOTA_LIMIT = 10000  # Daily quota limit
    REQUESTS_PER_SECOND = 10     # Max requests per second
    
    def __init__(self, config: Configuration):
        """
        Initialize YouTube client with configuration.
        
        Args:
            config: Configuration instance with API key and settings
            
        Raises:
            AuthenticationError: If API key is invalid
            YouTubeAPIError: If client initialization fails
        """
        self.config = config
        self.api_key = config.youtube_api_key
        self.quota_used = 0
        self.quota_limit = self.DEFAULT_QUOTA_LIMIT
        self.last_request_time = 0.0
        
        # Initialize YouTube API client
        try:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            logger.info("YouTube API client initialized successfully")
        except GoogleAuthError as e:
            raise AuthenticationError(f"YouTube API authentication failed: {e}")
        except Exception as e:
            raise YouTubeAPIError(f"Failed to initialize YouTube client: {e}")
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between API requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 1.0 / self.REQUESTS_PER_SECOND
        
        if time_since_last_request < min_interval:
            sleep_time = min_interval - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _check_quota(self, operation: str) -> None:
        """
        Check if we have enough quota for the operation.
        
        Args:
            operation: API operation name
            
        Raises:
            RateLimitError: If quota would be exceeded
        """
        cost = self.QUOTA_COSTS.get(operation, 1)
        if self.quota_used + cost > self.quota_limit:
            raise RateLimitError(
                f"Quota limit would be exceeded. Used: {self.quota_used}, "
                f"Limit: {self.quota_limit}, Operation cost: {cost}"
            )
    
    def _update_quota(self, operation: str) -> None:
        """Update quota usage after successful API call."""
        cost = self.QUOTA_COSTS.get(operation, 1)
        self.quota_used += cost
        logger.debug(f"Quota updated: +{cost}, total: {self.quota_used}/{self.quota_limit}")
    
    def _retry_with_backoff(self, func, *args, max_retries: int = 3, **kwargs) -> Any:
        """
        Execute function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retry attempts
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result
            
        Raises:
            RateLimitError: If rate limit is exceeded
            AuthenticationError: If authentication fails
            YouTubeAPIError: If all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                self._enforce_rate_limit()
                return func(*args, **kwargs)
                
            except HttpError as e:
                last_exception = e
                error_code = e.resp.status
                error_reason = e.error_details[0].get('reason', '') if e.error_details else ''
                
                logger.warning(f"HTTP error {error_code} on attempt {attempt + 1}: {error_reason}")
                
                # Handle specific error codes
                if error_code == 403:
                    if 'quotaExceeded' in error_reason or 'dailyLimitExceeded' in error_reason:
                        raise RateLimitError(f"YouTube API quota exceeded: {error_reason}")
                    elif 'keyInvalid' in error_reason or 'keyExpired' in error_reason:
                        raise AuthenticationError(f"YouTube API key invalid: {error_reason}")
                elif error_code == 401:
                    raise AuthenticationError(f"YouTube API authentication failed: {error_reason}")
                elif error_code == 429:
                    # Rate limit exceeded, wait longer
                    if attempt < max_retries:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.info(f"Rate limited, waiting {wait_time:.2f} seconds before retry")
                        time.sleep(wait_time)
                        continue
                
                # For other 4xx errors, don't retry
                if 400 <= error_code < 500 and error_code not in [429, 403]:
                    raise YouTubeAPIError(f"Client error {error_code}: {error_reason}")
                
                # For 5xx errors, retry with backoff
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Server error {error_code}, retrying in {wait_time:.2f} seconds")
                    time.sleep(wait_time)
                    continue
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")
                
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {wait_time:.2f} seconds")
                    time.sleep(wait_time)
                    continue
        
        # All retries failed
        raise YouTubeAPIError(f"All retry attempts failed. Last error: {last_exception}")
    
    def search_videos(self, filters: SearchFilters) -> List[Dict[str, Any]]:
        """
        Search for YouTube videos with specified filters.
        
        Args:
            filters: Search filters including keywords, date range, etc.
            
        Returns:
            List of video search results with basic metadata
            
        Raises:
            RateLimitError: If API quota is exceeded
            YouTubeAPIError: If search fails
        """
        self._check_quota('search')
        
        # Build search query
        query = ' '.join(filters.keywords)
        
        # Calculate published after date
        published_after = datetime.now() - timedelta(days=filters.days_back)
        published_after_str = published_after.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        logger.info(f"Searching for videos: query='{query}', days_back={filters.days_back}")
        
        def _search():
            request = self.youtube.search().list(
                part='snippet',
                q=query,
                type='video',
                order=filters.order,
                maxResults=min(filters.max_results, 50),  # API limit is 50
                publishedAfter=published_after_str,
                videoDuration=filters.video_duration,
                videoDefinition=filters.video_definition,
                relevanceLanguage='en',  # Focus on English content
                safeSearch='moderate'
            )
            return request.execute()
        
        try:
            response = self._retry_with_backoff(_search)
            self._update_quota('search')
            
            videos = response.get('items', [])
            logger.info(f"Found {len(videos)} videos in search results")
            
            return videos
            
        except Exception as e:
            logger.error(f"Video search failed: {e}")
            raise
    
    def get_video_details(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get detailed information for multiple videos with comprehensive metadata.
        
        Implements detailed metadata retrieval including:
        - View count, like count, comment count, upload date
        - Video status and availability information
        - Content details (duration, definition, etc.)
        - Handles missing or restricted video data gracefully
        
        Args:
            video_ids: List of YouTube video IDs
            
        Returns:
            List of detailed video information with complete metadata
            
        Raises:
            RateLimitError: If API quota is exceeded
            YouTubeAPIError: If details fetching fails
        """
        if not video_ids:
            logger.debug("No video IDs provided for details fetching")
            return []
        
        self._check_quota('videos')
        
        # API allows up to 50 video IDs per request
        batch_size = 50
        all_videos = []
        failed_video_ids = []
        
        logger.info(f"Fetching detailed metadata for {len(video_ids)} videos")
        
        for i in range(0, len(video_ids), batch_size):
            batch_ids = video_ids[i:i + batch_size]
            video_ids_str = ','.join(batch_ids)
            
            logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch_ids)} videos")
            
            def _get_details():
                request = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails,status',
                    id=video_ids_str
                )
                return request.execute()
            
            try:
                response = self._retry_with_backoff(_get_details)
                self._update_quota('videos')
                
                batch_videos = response.get('items', [])
                
                # Process each video and handle missing/restricted data
                for video in batch_videos:
                    processed_video = self._process_video_details(video)
                    if processed_video:
                        all_videos.append(processed_video)
                    else:
                        failed_video_ids.append(video.get('id', 'unknown'))
                
                # Check for videos that weren't returned (deleted, private, etc.)
                returned_ids = {video['id'] for video in batch_videos}
                missing_ids = set(batch_ids) - returned_ids
                
                if missing_ids:
                    logger.warning(f"Videos not found or restricted: {list(missing_ids)}")
                    failed_video_ids.extend(missing_ids)
                
            except Exception as e:
                logger.error(f"Failed to fetch video details for batch: {e}")
                failed_video_ids.extend(batch_ids)
                # Continue with other batches
                continue
        
        success_count = len(all_videos)
        failure_count = len(failed_video_ids)
        
        logger.info(f"Video details retrieval complete: {success_count} successful, {failure_count} failed")
        
        if failed_video_ids:
            logger.debug(f"Failed video IDs: {failed_video_ids[:10]}{'...' if len(failed_video_ids) > 10 else ''}")
        
        return all_videos
    
    def _process_video_details(self, video: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process individual video details and handle missing/restricted data gracefully.
        
        Args:
            video: Raw video data from YouTube API
            
        Returns:
            Processed video data or None if video should be skipped
        """
        try:
            video_id = video.get('id')
            if not video_id:
                logger.warning("Video missing ID, skipping")
                return None
            
            snippet = video.get('snippet', {})
            statistics = video.get('statistics', {})
            status = video.get('status', {})
            content_details = video.get('contentDetails', {})
            
            # Check if video is available for processing
            if not self._is_video_available(status, snippet):
                logger.debug(f"Video {video_id} not available for processing")
                return None
            
            # Extract and validate metadata with graceful handling of missing data
            processed_video = {
                'id': video_id,
                'snippet': {
                    'title': snippet.get('title', 'Unknown Title'),
                    'channelId': snippet.get('channelId', ''),
                    'channelTitle': snippet.get('channelTitle', 'Unknown Channel'),
                    'publishedAt': snippet.get('publishedAt', ''),
                    'description': snippet.get('description', ''),
                    'thumbnails': snippet.get('thumbnails', {}),
                    'tags': snippet.get('tags', []),
                    'categoryId': snippet.get('categoryId', ''),
                    'defaultLanguage': snippet.get('defaultLanguage', ''),
                    'defaultAudioLanguage': snippet.get('defaultAudioLanguage', '')
                },
                'statistics': {
                    'viewCount': self._safe_int_parse(statistics.get('viewCount', '0')),
                    'likeCount': self._safe_int_parse(statistics.get('likeCount', '0')),
                    'commentCount': self._safe_int_parse(statistics.get('commentCount', '0')),
                    # Note: dislikeCount is no longer available in YouTube API
                    'favoriteCount': self._safe_int_parse(statistics.get('favoriteCount', '0'))
                },
                'contentDetails': {
                    'duration': content_details.get('duration', ''),
                    'dimension': content_details.get('dimension', ''),
                    'definition': content_details.get('definition', ''),
                    'caption': content_details.get('caption', 'false'),
                    'licensedContent': content_details.get('licensedContent', False),
                    'projection': content_details.get('projection', 'rectangular')
                },
                'status': {
                    'uploadStatus': status.get('uploadStatus', ''),
                    'privacyStatus': status.get('privacyStatus', ''),
                    'license': status.get('license', ''),
                    'embeddable': status.get('embeddable', True),
                    'publicStatsViewable': status.get('publicStatsViewable', True)
                }
            }
            
            # Add engagement metrics
            processed_video['engagementMetrics'] = self._calculate_engagement_metrics(processed_video)
            
            return processed_video
            
        except Exception as e:
            logger.warning(f"Error processing video details for {video.get('id', 'unknown')}: {e}")
            return None
    
    def _is_video_available(self, status: Dict[str, Any], snippet: Dict[str, Any]) -> bool:
        """
        Check if video is available for processing based on status and snippet data.
        
        Args:
            status: Video status information
            snippet: Video snippet information
            
        Returns:
            True if video is available for processing
        """
        # Check upload status
        upload_status = status.get('uploadStatus', '')
        if upload_status not in ['processed', '']:
            logger.debug(f"Video not processed: upload_status={upload_status}")
            return False
        
        # Check privacy status
        privacy_status = status.get('privacyStatus', '')
        if privacy_status in ['private', 'privacyStatusUnspecified']:
            logger.debug(f"Video not public: privacy_status={privacy_status}")
            return False
        
        # Check if basic metadata is available
        if not snippet.get('title') or snippet.get('title') == '[Private video]':
            logger.debug("Video title indicates private or deleted video")
            return False
        
        if not snippet.get('publishedAt'):
            logger.debug("Video missing publish date")
            return False
        
        return True
    
    def _safe_int_parse(self, value: str) -> int:
        """
        Safely parse string to integer with fallback to 0.
        
        Args:
            value: String value to parse
            
        Returns:
            Parsed integer or 0 if parsing fails
        """
        try:
            return int(value) if value else 0
        except (ValueError, TypeError):
            return 0
    
    def _calculate_engagement_metrics(self, video: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate engagement metrics for a video.
        
        Implements engagement metrics calculation including:
        - Like-to-view ratio
        - Comment-to-view ratio  
        - Overall engagement rate
        - Engagement quality score
        
        Args:
            video: Processed video data
            
        Returns:
            Dictionary of engagement metrics
        """
        statistics = video.get('statistics', {})
        view_count = statistics.get('viewCount', 0)
        like_count = statistics.get('likeCount', 0)
        comment_count = statistics.get('commentCount', 0)
        
        # Avoid division by zero
        if view_count == 0:
            return {
                'likeToViewRatio': 0.0,
                'commentToViewRatio': 0.0,
                'engagementRate': 0.0,
                'engagementScore': 0.0,
                'likeRatio': 0.0,
                'viewToSubscriberRatio': 0.0  # Will be calculated when channel info is available
            }
        
        # Calculate basic engagement ratios
        like_to_view_ratio = like_count / view_count
        comment_to_view_ratio = comment_count / view_count
        engagement_rate = (like_count + comment_count) / view_count
        
        # Calculate like ratio (estimate since dislikes not available)
        # Use engagement patterns to estimate like ratio
        # Higher engagement typically correlates with higher like ratio
        estimated_like_ratio = min(0.85 + (engagement_rate * 15), 1.0)
        
        # Calculate engagement quality score (0-100)
        # Combines multiple factors for overall engagement assessment
        engagement_score = min(
            (like_to_view_ratio * 2000) +  # Like ratio component (up to 20 points)
            (comment_to_view_ratio * 5000) +  # Comment ratio component (up to 50 points)
            (estimated_like_ratio * 30),  # Like ratio component (up to 30 points)
            100.0
        )
        
        return {
            'likeToViewRatio': like_to_view_ratio,
            'commentToViewRatio': comment_to_view_ratio,
            'engagementRate': engagement_rate,
            'engagementScore': engagement_score,
            'likeRatio': estimated_like_ratio,
            'viewToSubscriberRatio': 0.0  # Will be updated when channel info is available
        }
    
    def get_channel_info(self, channel_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get channel information for multiple channels with subscriber data for engagement metrics.
        
        Retrieves channel information needed for calculating view-to-subscriber ratios
        and other channel-based engagement metrics.
        
        Args:
            channel_ids: List of YouTube channel IDs
            
        Returns:
            Dictionary mapping channel ID to processed channel info
            
        Raises:
            RateLimitError: If API quota is exceeded
            YouTubeAPIError: If channel info fetching fails
        """
        if not channel_ids:
            logger.debug("No channel IDs provided for channel info fetching")
            return {}
        
        self._check_quota('channels')
        
        # Remove duplicates while preserving order
        unique_channel_ids = list(dict.fromkeys(channel_ids))
        
        # API allows up to 50 channel IDs per request
        batch_size = 50
        all_channels = {}
        failed_channel_ids = []
        
        logger.info(f"Fetching channel information for {len(unique_channel_ids)} channels")
        
        for i in range(0, len(unique_channel_ids), batch_size):
            batch_ids = unique_channel_ids[i:i + batch_size]
            channel_ids_str = ','.join(batch_ids)
            
            logger.debug(f"Processing channel batch {i//batch_size + 1}: {len(batch_ids)} channels")
            
            def _get_channels():
                request = self.youtube.channels().list(
                    part='snippet,statistics,status',
                    id=channel_ids_str
                )
                return request.execute()
            
            try:
                response = self._retry_with_backoff(_get_channels)
                self._update_quota('channels')
                
                batch_channels = response.get('items', [])
                
                # Process each channel and handle missing data
                for channel in batch_channels:
                    processed_channel = self._process_channel_info(channel)
                    if processed_channel:
                        channel_id = channel['id']
                        all_channels[channel_id] = processed_channel
                    else:
                        failed_channel_ids.append(channel.get('id', 'unknown'))
                
                # Check for channels that weren't returned
                returned_ids = {channel['id'] for channel in batch_channels}
                missing_ids = set(batch_ids) - returned_ids
                
                if missing_ids:
                    logger.warning(f"Channels not found: {list(missing_ids)}")
                    failed_channel_ids.extend(missing_ids)
                    
            except Exception as e:
                logger.error(f"Failed to fetch channel info for batch: {e}")
                failed_channel_ids.extend(batch_ids)
                # Continue with other batches
                continue
        
        success_count = len(all_channels)
        failure_count = len(failed_channel_ids)
        
        logger.info(f"Channel info retrieval complete: {success_count} successful, {failure_count} failed")
        
        if failed_channel_ids:
            logger.debug(f"Failed channel IDs: {failed_channel_ids[:5]}{'...' if len(failed_channel_ids) > 5 else ''}")
        
        return all_channels
    
    def _process_channel_info(self, channel: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process individual channel information and handle missing data gracefully.
        
        Args:
            channel: Raw channel data from YouTube API
            
        Returns:
            Processed channel data or None if channel should be skipped
        """
        try:
            channel_id = channel.get('id')
            if not channel_id:
                logger.warning("Channel missing ID, skipping")
                return None
            
            snippet = channel.get('snippet', {})
            statistics = channel.get('statistics', {})
            status = channel.get('status', {})
            
            # Process channel data with graceful handling of missing fields
            processed_channel = {
                'id': channel_id,
                'snippet': {
                    'title': snippet.get('title', 'Unknown Channel'),
                    'description': snippet.get('description', ''),
                    'customUrl': snippet.get('customUrl', ''),
                    'publishedAt': snippet.get('publishedAt', ''),
                    'thumbnails': snippet.get('thumbnails', {}),
                    'country': snippet.get('country', ''),
                    'defaultLanguage': snippet.get('defaultLanguage', '')
                },
                'statistics': {
                    'viewCount': self._safe_int_parse(statistics.get('viewCount', '0')),
                    'subscriberCount': self._safe_int_parse(statistics.get('subscriberCount', '0')),
                    'videoCount': self._safe_int_parse(statistics.get('videoCount', '0')),
                    'hiddenSubscriberCount': statistics.get('hiddenSubscriberCount', False)
                },
                'status': {
                    'privacyStatus': status.get('privacyStatus', ''),
                    'isLinked': status.get('isLinked', False),
                    'longUploadsStatus': status.get('longUploadsStatus', '')
                }
            }
            
            return processed_channel
            
        except Exception as e:
            logger.warning(f"Error processing channel info for {channel.get('id', 'unknown')}: {e}")
            return None
    
    def discover_videos(self, keywords: List[str], max_videos: int = 10, days_back: int = 7) -> List[VideoData]:
        """
        Discover videos with keyword-based search and trending evaluation.
        
        This is the main video discovery function used by the workflow.
        It implements 7-day date filtering for weekly podcast focus and
        filters videos by minimum view count and age requirements.
        
        Args:
            keywords: List of search keywords for video discovery
            max_videos: Maximum number of videos to return
            days_back: Number of days back to search (default 7 for weekly focus)
            
        Returns:
            List of VideoData objects meeting discovery criteria
            
        Raises:
            RateLimitError: If API quota is exceeded
            YouTubeAPIError: If discovery fails
        """
        logger.info(f"Starting video discovery with keywords: {keywords}")
        logger.info(f"Search parameters: max_videos={max_videos}, days_back={days_back}")
        
        # Create search filters for discovery
        filters = SearchFilters(
            keywords=keywords,
            days_back=days_back,
            max_results=min(max_videos * 2, 50),  # Search for more to allow filtering
            min_views=1000,  # Minimum 1000 views as per requirements
            order="relevance"  # Start with relevance, can be adjusted for trending
        )
        
        try:
            # Step 1: Search for videos with keyword-based discovery
            search_results = self.search_videos(filters)
            
            if not search_results:
                logger.warning("No videos found in search results")
                return []
            
            logger.info(f"Found {len(search_results)} videos in initial search")
            
            # Step 2: Extract video IDs and channel IDs
            video_ids = [item['id']['videoId'] for item in search_results]
            channel_ids = [item['snippet']['channelId'] for item in search_results]
            
            # Step 3: Get detailed video information
            video_details = self.get_video_details(video_ids)
            
            # Step 4: Get channel information for engagement analysis
            channel_info = self.get_channel_info(channel_ids)
            
            # Step 5: Convert to VideoData objects and apply discovery filters
            discovered_videos = []
            
            for video in video_details:
                try:
                    video_data = self._convert_to_video_data(video, channel_info)
                    
                    # Apply discovery criteria
                    if self._meets_discovery_criteria(video_data, days_back):
                        discovered_videos.append(video_data)
                        logger.debug(f"Added video: {video_data.title} (views: {video_data.view_count})")
                    else:
                        logger.debug(f"Filtered out: {video_data.title} (views: {video_data.view_count})")
                        
                except Exception as e:
                    logger.warning(f"Failed to process video {video.get('id', 'unknown')}: {e}")
                    continue
            
            # Step 6: Evaluate trending status and sort by trending score
            video_scores = []
            for video in discovered_videos:
                trending_score = self._calculate_trending_score(video)
                video_scores.append((video, trending_score))
            
            # Sort by trending score (highest first)
            video_scores.sort(key=lambda x: x[1], reverse=True)
            discovered_videos = [video for video, score in video_scores]
            
            # Step 7: Return top videos up to max_videos limit
            final_videos = discovered_videos[:max_videos]
            
            logger.info(f"Discovered {len(final_videos)} videos meeting all criteria")
            
            # Log summary of discovered videos with their trending scores
            for i, video in enumerate(final_videos, 1):
                # Recalculate trending score for logging (could be optimized by storing)
                trending_score = self._calculate_trending_score(video)
                logger.info(f"#{i}: {video.title} (views: {video.view_count}, "
                           f"engagement: {video.get_engagement_rate():.4f}, "
                           f"trending_score: {trending_score:.2f})")
            
            return final_videos
            
        except Exception as e:
            logger.error(f"Video discovery failed: {e}")
            raise YouTubeAPIError(f"Failed to discover videos: {e}")
    
    def discover_trending_videos(self, filters: SearchFilters) -> List[VideoData]:
        """
        Discover trending videos based on search filters and engagement metrics.
        
        This method combines search results with detailed video information
        to identify trending content based on view counts, engagement rates,
        and upload recency.
        
        Args:
            filters: Search filters for video discovery
            
        Returns:
            List of VideoData objects for trending videos
            
        Raises:
            RateLimitError: If API quota is exceeded
            YouTubeAPIError: If discovery fails
        """
        logger.info("Starting trending video discovery")
        
        try:
            # Step 1: Search for videos
            search_results = self.search_videos(filters)
            
            if not search_results:
                logger.warning("No videos found in search results")
                return []
            
            # Step 2: Extract video IDs and channel IDs
            video_ids = [item['id']['videoId'] for item in search_results]
            channel_ids = [item['snippet']['channelId'] for item in search_results]
            
            # Step 3: Get detailed video information
            video_details = self.get_video_details(video_ids)
            
            # Step 4: Get channel information for subscriber counts
            channel_info = self.get_channel_info(channel_ids)
            
            # Step 5: Convert to VideoData objects and filter by criteria
            trending_videos = []
            
            for video in video_details:
                try:
                    video_data = self._convert_to_video_data(video, channel_info)
                    
                    # Apply trending filters
                    if self._is_trending_video(video_data, filters):
                        trending_videos.append(video_data)
                        logger.debug(f"Added trending video: {video_data.title}")
                    else:
                        logger.debug(f"Filtered out video: {video_data.title}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process video {video.get('id', 'unknown')}: {e}")
                    continue
            
            # Step 6: Sort by trending indicators
            trending_videos.sort(key=self._calculate_trending_score, reverse=True)
            
            logger.info(f"Discovered {len(trending_videos)} trending videos")
            return trending_videos
            
        except Exception as e:
            logger.error(f"Trending video discovery failed: {e}")
            raise YouTubeAPIError(f"Failed to discover trending videos: {e}")
    
    def _convert_to_video_data(self, video: Dict[str, Any], channel_info: Dict[str, Dict[str, Any]]) -> VideoData:
        """
        Convert enhanced YouTube API video response to VideoData object.
        
        Uses the enhanced video details with engagement metrics and handles
        missing or restricted video data gracefully.
        
        Args:
            video: Enhanced video data from YouTube API (processed by _process_video_details)
            channel_info: Channel information mapping
            
        Returns:
            VideoData object with complete metadata
        """
        video_id = video['id']
        snippet = video['snippet']
        statistics = video.get('statistics', {})
        engagement_metrics = video.get('engagementMetrics', {})
        
        # Get channel info for view-to-subscriber ratio calculation
        channel_id = snippet['channelId']
        channel_data = channel_info.get(channel_id, {})
        channel_snippet = channel_data.get('snippet', {})
        channel_statistics = channel_data.get('statistics', {})
        
        # Parse statistics with defaults (handle both processed and raw API data)
        view_count = self._safe_int_parse(str(statistics.get('viewCount', 0)))
        like_count = self._safe_int_parse(str(statistics.get('likeCount', 0)))
        comment_count = self._safe_int_parse(str(statistics.get('commentCount', 0)))
        
        # Calculate view-to-subscriber ratio if subscriber count is available
        subscriber_count = self._safe_int_parse(str(channel_statistics.get('subscriberCount', 0)))
        if subscriber_count > 0 and view_count > 0:
            view_to_subscriber_ratio = view_count / subscriber_count
            # Update engagement metrics with calculated ratio
            engagement_metrics['viewToSubscriberRatio'] = view_to_subscriber_ratio
        else:
            engagement_metrics['viewToSubscriberRatio'] = 0.0
        
        # Parse upload date
        upload_date = snippet['publishedAt']
        
        # Create VideoData object with enhanced metadata
        video_data = VideoData(
            video_id=video_id,
            title=snippet['title'],
            channel=channel_snippet.get('title', snippet['channelTitle']),
            view_count=view_count,
            like_count=like_count,
            comment_count=comment_count,
            upload_date=upload_date,
            transcript=None,  # Will be fetched separately
            quality_score=None,  # Will be calculated later
            key_topics=[]  # Will be extracted from transcript
        )
        
        # Store engagement metrics as additional attributes for quality evaluation
        # These will be used by the quality evaluation node
        video_data.__dict__.update({
            'engagement_metrics': engagement_metrics,
            'channel_subscriber_count': subscriber_count,
            'content_details': video.get('contentDetails', {}),
            'video_status': video.get('status', {})
        })
        
        return video_data
    
    def _meets_discovery_criteria(self, video: VideoData, days_back: int = 7) -> bool:
        """
        Check if a video meets the discovery criteria for the weekly podcast focus.
        
        Implements the filtering requirements:
        - Minimum view count (1000+ views)
        - Age requirements (within specified days_back)
        - Basic engagement threshold
        
        Args:
            video: VideoData object to evaluate
            days_back: Maximum age in days (default 7 for weekly focus)
            
        Returns:
            True if video meets all discovery criteria
        """
        # Requirement 1.4: Filter videos by minimum view count (1000+ views)
        if video.view_count < 1000:
            logger.debug(f"Video filtered: {video.title} - insufficient views ({video.view_count})")
            return False
        
        # Requirement 1.3: 7-day date filtering for weekly podcast focus
        if not video.is_recent(days_back):
            logger.debug(f"Video filtered: {video.title} - too old (uploaded: {video.upload_date})")
            return False
        
        # Basic engagement check - ensure video has some interaction
        engagement_rate = video.get_engagement_rate()
        min_engagement_rate = 0.0001  # Very low threshold (0.01%) for discovery
        
        if engagement_rate < min_engagement_rate:
            logger.debug(f"Video filtered: {video.title} - no engagement ({engagement_rate:.6f})")
            return False
        
        # All criteria met
        return True
    
    def _is_trending_video(self, video: VideoData, filters: SearchFilters) -> bool:
        """
        Determine if a video meets trending criteria.
        
        Args:
            video: VideoData object to evaluate
            filters: Search filters with minimum requirements
            
        Returns:
            True if video is considered trending
        """
        # Check minimum view count
        if video.view_count < filters.min_views:
            return False
        
        # Check if video is recent enough
        if not video.is_recent(filters.days_back):
            return False
        
        # Check engagement rate (likes + comments relative to views)
        engagement_rate = video.get_engagement_rate()
        min_engagement_rate = 0.001  # 0.1% minimum engagement
        
        if engagement_rate < min_engagement_rate:
            return False
        
        # Additional trending indicators could be added here
        # (e.g., view velocity, subscriber ratio, etc.)
        
        return True
    
    def _calculate_trending_score(self, video: VideoData) -> float:
        """
        Calculate a trending score for ranking videos based on multiple factors.
        
        Implements trending status evaluation considering:
        - View count growth indicators
        - Engagement rate (likes + comments relative to views)
        - Upload recency (weekly podcast focus)
        - Like ratio quality
        
        Args:
            video: VideoData object to score
            
        Returns:
            Trending score (higher is better)
        """
        # 1. View count score - indicates popularity and reach
        # Use logarithmic scaling to handle wide range of view counts
        import math
        if video.view_count > 0:
            view_score = min(math.log10(video.view_count) - 3, 10.0)  # log10(1000) = 3, cap at 10
        else:
            view_score = 0
        
        # 2. Engagement rate score - key trending indicator
        engagement_rate = video.get_engagement_rate()
        # Scale engagement rate (typical good videos have 0.01-0.05 engagement)
        engagement_score = min(engagement_rate * 200, 8.0)  # Cap at 8 points
        
        # 3. Recency score - prioritize recent content for weekly focus
        try:
            upload_dt = datetime.fromisoformat(video.upload_date.replace('Z', '+00:00'))
            hours_old = (datetime.now().replace(tzinfo=upload_dt.tzinfo) - upload_dt).total_seconds() / 3600
            
            # Higher score for more recent videos (within 7 days)
            if hours_old <= 24:  # Last 24 hours
                recency_score = 6.0
            elif hours_old <= 72:  # Last 3 days
                recency_score = 5.0
            elif hours_old <= 168:  # Last 7 days (weekly focus)
                recency_score = 4.0
            else:
                recency_score = max(0, 3.0 - (hours_old - 168) / 168)  # Decreasing after 7 days
        except:
            recency_score = 0
        
        # 4. Like ratio score - content quality indicator
        like_ratio = video.get_like_ratio()
        like_score = like_ratio * 3.0  # Up to 3 points for high like ratio
        
        # 5. View velocity estimation (views per hour since upload)
        try:
            upload_dt = datetime.fromisoformat(video.upload_date.replace('Z', '+00:00'))
            hours_since_upload = max(1, (datetime.now().replace(tzinfo=upload_dt.tzinfo) - upload_dt).total_seconds() / 3600)
            views_per_hour = video.view_count / hours_since_upload
            
            # Normalize view velocity (good trending videos get 100+ views/hour)
            velocity_score = min(views_per_hour / 100, 5.0)  # Cap at 5 points
        except:
            velocity_score = 0
        
        # 6. Comment engagement bonus - indicates active discussion
        if video.view_count > 0:
            comment_rate = video.comment_count / video.view_count
            comment_bonus = min(comment_rate * 500, 2.0)  # Up to 2 bonus points
        else:
            comment_bonus = 0
        
        # Calculate total trending score
        total_score = (view_score + engagement_score + recency_score + 
                      like_score + velocity_score + comment_bonus)
        
        logger.debug(f"Trending score for '{video.title}': {total_score:.2f} "
                    f"(views: {view_score:.1f}, engagement: {engagement_score:.1f}, "
                    f"recency: {recency_score:.1f}, likes: {like_score:.1f}, "
                    f"velocity: {velocity_score:.1f}, comments: {comment_bonus:.1f})")
        
        return total_score
    
    def fetch_video_details_batch(self, video_data_list: List[VideoData]) -> List[VideoData]:
        """
        Fetch detailed metadata for a batch of VideoData objects.
        
        This is the main method used by the workflow for task 3.3 implementation.
        It enhances VideoData objects with detailed metadata including:
        - Complete engagement metrics (like-to-dislike ratio, view-to-subscriber ratio)
        - Content details and video status information
        - Graceful handling of missing or restricted video data
        
        Args:
            video_data_list: List of VideoData objects to enhance with details
            
        Returns:
            List of VideoData objects with enhanced metadata
            
        Raises:
            RateLimitError: If API quota is exceeded
            YouTubeAPIError: If details fetching fails
        """
        if not video_data_list:
            logger.debug("No videos provided for details fetching")
            return []
        
        logger.info(f"Fetching detailed metadata for {len(video_data_list)} videos")
        
        # Extract video IDs and channel IDs
        video_ids = [video.video_id for video in video_data_list]
        channel_ids = [getattr(video, 'channel_id', None) for video in video_data_list]
        
        # For videos that don't have channel_id stored, we'll need to get it from the API
        # This is a fallback for videos that were created without full channel info
        video_id_to_video_data = {video.video_id: video for video in video_data_list}
        
        try:
            # Step 1: Get detailed video information
            detailed_videos = self.get_video_details(video_ids)
            
            # Step 2: Extract channel IDs from detailed video data
            all_channel_ids = []
            video_id_to_channel_id = {}
            
            for video in detailed_videos:
                video_id = video['id']
                channel_id = video['snippet']['channelId']
                video_id_to_channel_id[video_id] = channel_id
                all_channel_ids.append(channel_id)
            
            # Step 3: Get channel information for engagement metrics
            channel_info = self.get_channel_info(all_channel_ids)
            
            # Step 4: Update VideoData objects with enhanced details
            enhanced_videos = []
            
            for video in detailed_videos:
                video_id = video['id']
                original_video_data = video_id_to_video_data.get(video_id)
                
                if not original_video_data:
                    logger.warning(f"No original VideoData found for video {video_id}")
                    continue
                
                try:
                    # Convert to enhanced VideoData using the detailed information
                    enhanced_video_data = self._convert_to_video_data(video, channel_info)
                    
                    # Preserve any existing data that shouldn't be overwritten
                    if original_video_data.transcript:
                        enhanced_video_data.transcript = original_video_data.transcript
                    if original_video_data.quality_score is not None:
                        enhanced_video_data.quality_score = original_video_data.quality_score
                    if original_video_data.key_topics:
                        enhanced_video_data.key_topics = original_video_data.key_topics
                    
                    enhanced_videos.append(enhanced_video_data)
                    
                    logger.debug(f"Enhanced video details for: {enhanced_video_data.title}")
                    
                except Exception as e:
                    logger.warning(f"Failed to enhance video {video_id}: {e}")
                    # Keep original video data if enhancement fails
                    enhanced_videos.append(original_video_data)
            
            # Step 5: Handle videos that weren't found in API response
            enhanced_video_ids = {video.video_id for video in enhanced_videos}
            missing_videos = [video for video in video_data_list if video.video_id not in enhanced_video_ids]
            
            if missing_videos:
                logger.warning(f"Could not enhance {len(missing_videos)} videos (deleted, private, or restricted)")
                for video in missing_videos:
                    logger.debug(f"Missing video: {video.title} ({video.video_id})")
                
                # Add missing videos back to results (they may still be useful)
                enhanced_videos.extend(missing_videos)
            
            success_count = len(enhanced_videos) - len(missing_videos)
            logger.info(f"Video details enhancement complete: {success_count} enhanced, {len(missing_videos)} unchanged")
            
            return enhanced_videos
            
        except Exception as e:
            logger.error(f"Failed to fetch video details batch: {e}")
            # Return original videos if enhancement fails completely
            logger.warning("Returning original video data due to enhancement failure")
            return video_data_list

    def get_quota_usage(self) -> Tuple[int, int]:
        """
        Get current quota usage.
        
        Returns:
            Tuple of (used_quota, quota_limit)
        """
        return self.quota_used, self.quota_limit
    
    def reset_quota(self) -> None:
        """Reset quota usage counter (for new day)."""
        self.quota_used = 0
        logger.info("Quota usage reset")
    
    def set_quota_limit(self, limit: int) -> None:
        """
        Set custom quota limit.
        
        Args:
            limit: New quota limit
        """
        self.quota_limit = limit
        logger.info(f"Quota limit set to {limit}")