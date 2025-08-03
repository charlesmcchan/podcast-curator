"""
Tests for the transcript processor module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    YouTubeRequestFailed,
    CouldNotRetrieveTranscript
)

from src.nanook_curator.transcript_processor import TranscriptProcessor, fetch_transcripts_node
from src.nanook_curator.models import VideoData, CuratorState


class TestTranscriptProcessor:
    """Test cases for TranscriptProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TranscriptProcessor()
        
    def test_init_default_languages(self):
        """Test processor initialization with default languages."""
        processor = TranscriptProcessor()
        assert processor.preferred_languages == ['en', 'en-US', 'en-GB', 'en-CA']
        assert processor.fallback_languages == ['en-auto', 'auto']
        
    def test_init_custom_languages(self):
        """Test processor initialization with custom languages."""
        custom_langs = ['es', 'fr']
        processor = TranscriptProcessor(preferred_languages=custom_langs)
        assert processor.preferred_languages == custom_langs
        
    def test_clean_text_basic(self):
        """Test basic text cleaning functionality."""
        raw_text = "  Hello world  this is a test  "
        cleaned = self.processor._clean_text(raw_text)
        assert cleaned == "Hello world this is a test"
        
    def test_clean_text_artifacts(self):
        """Test removal of transcript artifacts."""
        raw_text = "[Music] Speaker 1: Hello world [Applause] um this is a test"
        cleaned = self.processor._clean_text(raw_text)
        assert "[Music]" not in cleaned
        assert "[Applause]" not in cleaned
        # The speaker label should be removed from the beginning of lines
        assert not cleaned.startswith("Speaker 1:")
        # Check that the main content is preserved
        assert "Hello world" in cleaned
        assert "this is a test" in cleaned
        
    def test_clean_text_timestamps(self):
        """Test removal of timestamps."""
        raw_text = "Hello world 12:34 this is a test 1:23:45"
        cleaned = self.processor._clean_text(raw_text)
        assert "12:34" not in cleaned
        assert "1:23:45" not in cleaned
        
    def test_clean_transcript_empty(self):
        """Test cleaning empty transcript."""
        result = self.processor._clean_transcript([])
        assert result == ""
        
    def test_clean_transcript_valid(self):
        """Test cleaning valid transcript data."""
        raw_transcript = [
            {'text': 'Hello world', 'start': 0.0, 'duration': 2.0},
            {'text': 'This is a test', 'start': 2.0, 'duration': 3.0}
        ]
        result = self.processor._clean_transcript(raw_transcript)
        assert "Hello world" in result
        assert "This is a test" in result
        
    @patch('src.nanook_curator.transcript_processor.YouTubeTranscriptApi.list')
    def test_fetch_transcript_success_manual(self, mock_list_transcripts):
        """Test successful transcript fetching with manual transcript."""
        # Mock transcript list and transcript
        mock_transcript = Mock()
        mock_transcript.fetch.return_value = [
            {'text': 'Hello world', 'start': 0.0, 'duration': 2.0}
        ]
        
        mock_transcript_list = Mock()
        mock_transcript_list.find_manually_created_transcript.return_value = mock_transcript
        mock_list_transcripts.return_value = mock_transcript_list
        
        result = self.processor.fetch_transcript('test_video_id')
        
        assert result is not None
        assert "Hello world" in result
        mock_list_transcripts.assert_called_once_with('test_video_id')
        
    @patch('src.nanook_curator.transcript_processor.YouTubeTranscriptApi.list')
    def test_fetch_transcript_success_auto_generated(self, mock_list_transcripts):
        """Test successful transcript fetching with auto-generated transcript."""
        # Mock transcript list and transcript
        mock_transcript = Mock()
        mock_transcript.fetch.return_value = [
            {'text': 'Auto generated content', 'start': 0.0, 'duration': 2.0}
        ]
        
        mock_transcript_list = Mock()
        # Manual transcript not found, but auto-generated is available
        mock_transcript_list.find_manually_created_transcript.side_effect = NoTranscriptFound
        mock_transcript_list.find_generated_transcript.return_value = mock_transcript
        mock_list_transcripts.return_value = mock_transcript_list
        
        result = self.processor.fetch_transcript('test_video_id')
        
        assert result is not None
        assert "Auto generated content" in result
        
    @patch('src.nanook_curator.transcript_processor.YouTubeTranscriptApi.list')
    def test_fetch_transcript_transcripts_disabled(self, mock_list_transcripts):
        """Test handling of disabled transcripts."""
        mock_list_transcripts.side_effect = TranscriptsDisabled('test_video_id')
        
        result = self.processor.fetch_transcript('test_video_id')
        
        assert result is None
        
    @patch('src.nanook_curator.transcript_processor.YouTubeTranscriptApi.list')
    def test_fetch_transcript_video_unavailable(self, mock_list_transcripts):
        """Test handling of unavailable video."""
        mock_list_transcripts.side_effect = VideoUnavailable('test_video_id')
        
        result = self.processor.fetch_transcript('test_video_id')
        
        assert result is None
        
    @patch('src.nanook_curator.transcript_processor.YouTubeTranscriptApi.list')
    def test_fetch_transcript_request_failed(self, mock_list_transcripts):
        """Test handling of YouTube request failures."""
        mock_list_transcripts.side_effect = YouTubeRequestFailed('test_video_id', 'HTTP Error')
        
        result = self.processor.fetch_transcript('test_video_id')
        
        assert result is None
        
    @patch('src.nanook_curator.transcript_processor.YouTubeTranscriptApi.list')
    def test_fetch_transcript_could_not_retrieve(self, mock_list_transcripts):
        """Test handling when transcript could not be retrieved."""
        mock_list_transcripts.side_effect = CouldNotRetrieveTranscript('test_video_id')
        
        result = self.processor.fetch_transcript('test_video_id')
        
        assert result is None
        
    def test_process_videos_transcripts_empty_state(self):
        """Test processing transcripts with empty video list."""
        state = CuratorState(search_keywords=['test'])
        
        result_state = self.processor.process_videos_transcripts(state)
        
        assert result_state == state
        assert len(result_state.discovered_videos) == 0
        
    @patch.object(TranscriptProcessor, 'fetch_transcript')
    def test_process_videos_transcripts_success(self, mock_fetch):
        """Test successful transcript processing for multiple videos."""
        # Setup mock
        mock_fetch.side_effect = ['Transcript 1', 'Transcript 2', None]  # Third video has no transcript
        
        # Create test videos
        videos = [
            VideoData(
                video_id='dQw4w9WgXcQ',
                title='Video 1',
                channel='Channel 1',
                view_count=1000,
                like_count=100,
                comment_count=10,
                upload_date='2024-01-01T00:00:00Z'
            ),
            VideoData(
                video_id='jNQXAC9IVRw',
                title='Video 2',
                channel='Channel 2',
                view_count=2000,
                like_count=200,
                comment_count=20,
                upload_date='2024-01-02T00:00:00Z'
            ),
            VideoData(
                video_id='9bZkp7q19f0',
                title='Video 3',
                channel='Channel 3',
                view_count=3000,
                like_count=300,
                comment_count=30,
                upload_date='2024-01-03T00:00:00Z'
            )
        ]
        
        state = CuratorState(search_keywords=['test'], discovered_videos=videos)
        
        result_state = self.processor.process_videos_transcripts(state)
        
        # Check results
        assert result_state.discovered_videos[0].transcript == 'Transcript 1'
        assert result_state.discovered_videos[1].transcript == 'Transcript 2'
        assert result_state.discovered_videos[2].transcript is None
        
        # Check metadata
        assert result_state.generation_metadata['transcripts_processed'] == 3
        assert result_state.generation_metadata['transcripts_successful'] == 2
        assert result_state.generation_metadata['transcripts_failed'] == 1
        assert result_state.generation_metadata['transcript_success_rate'] == 2/3
        
    def test_get_transcript_statistics_empty(self):
        """Test statistics calculation with empty video list."""
        stats = self.processor.get_transcript_statistics([])
        
        expected = {
            'total_videos': 0,
            'with_transcripts': 0,
            'without_transcripts': 0,
            'availability_rate': 0.0,
            'average_length': 0,
            'total_words': 0
        }
        
        for key, value in expected.items():
            assert stats[key] == value
            
    def test_get_transcript_statistics_mixed(self):
        """Test statistics calculation with mixed transcript availability."""
        videos = [
            VideoData(
                video_id='dQw4w9WgXcQ',
                title='Video 1',
                channel='Channel 1',
                view_count=1000,
                like_count=100,
                comment_count=10,
                upload_date='2024-01-01T00:00:00Z',
                transcript='This is a test transcript with ten words exactly here'
            ),
            VideoData(
                video_id='jNQXAC9IVRw',
                title='Video 2',
                channel='Channel 2',
                view_count=2000,
                like_count=200,
                comment_count=20,
                upload_date='2024-01-02T00:00:00Z',
                transcript='Short transcript'
            ),
            VideoData(
                video_id='9bZkp7q19f0',
                title='Video 3',
                channel='Channel 3',
                view_count=3000,
                like_count=300,
                comment_count=30,
                upload_date='2024-01-03T00:00:00Z'
                # No transcript
            )
        ]
        
        stats = self.processor.get_transcript_statistics(videos)
        
        assert stats['total_videos'] == 3
        assert stats['with_transcripts'] == 2
        assert stats['without_transcripts'] == 1
        assert stats['availability_rate'] == 2/3
        assert stats['total_words'] == 12  # 10 + 2 words
        assert stats['average_word_count'] == 6  # 12 / 2


class TestFetchTranscriptsNode:
    """Test cases for the fetch_transcripts_node function."""
    
    @patch('src.nanook_curator.transcript_processor.TranscriptProcessor')
    def test_fetch_transcripts_node_success(self, mock_processor_class):
        """Test successful execution of fetch_transcripts_node."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        # Create test state
        state = CuratorState(search_keywords=['test'])
        mock_processor.process_videos_transcripts.return_value = state
        mock_processor.get_transcript_statistics.return_value = {
            'availability_rate': 0.8,
            'with_transcripts': 4,
            'total_videos': 5
        }
        
        result_state = fetch_transcripts_node(state)
        
        assert result_state == state
        mock_processor.process_videos_transcripts.assert_called_once_with(state)
        mock_processor.get_transcript_statistics.assert_called_once()
        
    @patch('src.nanook_curator.transcript_processor.TranscriptProcessor')
    def test_fetch_transcripts_node_error(self, mock_processor_class):
        """Test error handling in fetch_transcripts_node."""
        # Setup mock to raise exception
        mock_processor_class.side_effect = Exception("Test error")
        
        state = CuratorState(search_keywords=['test'])
        
        result_state = fetch_transcripts_node(state)
        
        assert len(result_state.errors) > 0
        assert "Critical error in transcript fetching node" in result_state.errors[0]