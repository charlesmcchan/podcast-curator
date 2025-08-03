"""
Transcript processing module for the nanook-curator system.

This module handles fetching, cleaning, and parsing YouTube video transcripts
using the youtube-transcript-api library with robust error handling and fallback mechanisms.
"""

import logging
import re
from typing import List, Optional, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    NotTranslatable,
    TranslationLanguageNotAvailable,
    CookiePathInvalid,
    FailedToCreateConsentCookie,
    YouTubeRequestFailed,
    CouldNotRetrieveTranscript
)

from .models import VideoData, CuratorState

# Configure logging
logger = logging.getLogger(__name__)


class TranscriptProcessor:
    """
    Handles YouTube transcript fetching and processing with comprehensive error handling.
    
    This class provides methods to fetch transcripts from YouTube videos, clean and parse
    the text content, and handle various error conditions gracefully.
    """
    
    def __init__(self, preferred_languages: List[str] = None):
        """
        Initialize the transcript processor.
        
        Args:
            preferred_languages: List of preferred language codes (e.g., ['en', 'en-US'])
                                Defaults to English variants if not specified.
        """
        self.preferred_languages = preferred_languages or ['en', 'en-US', 'en-GB', 'en-CA']
        self.fallback_languages = ['en-auto', 'auto']  # Auto-generated transcripts as fallback
        
    def fetch_transcript(self, video_id: str) -> Optional[str]:
        """
        Fetch transcript for a single video with comprehensive error handling.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Cleaned transcript text or None if unavailable
        """
        try:
            # First, try to get manually created transcripts in preferred languages
            transcript_list = YouTubeTranscriptApi.list(video_id)
            
            # Try preferred languages first (manually created)
            for lang in self.preferred_languages:
                try:
                    transcript = transcript_list.find_manually_created_transcript([lang])
                    raw_transcript = transcript.fetch()
                    cleaned_text = self._clean_transcript(raw_transcript)
                    logger.info(f"Successfully fetched manual transcript for {video_id} in {lang}")
                    return cleaned_text
                except NoTranscriptFound:
                    continue
                except Exception as e:
                    logger.warning(f"Error fetching manual transcript for {video_id} in {lang}: {e}")
                    continue
            
            # If no manual transcripts found, try auto-generated transcripts
            for lang in self.preferred_languages + self.fallback_languages:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    raw_transcript = transcript.fetch()
                    cleaned_text = self._clean_transcript(raw_transcript)
                    logger.info(f"Successfully fetched auto-generated transcript for {video_id} in {lang}")
                    return cleaned_text
                except NoTranscriptFound:
                    continue
                except Exception as e:
                    logger.warning(f"Error fetching auto-generated transcript for {video_id} in {lang}: {e}")
                    continue
            
            # If still no transcript found, try translation from available transcripts
            try:
                available_transcripts = list(transcript_list)
                if available_transcripts:
                    # Try to translate the first available transcript to English
                    first_transcript = available_transcripts[0]
                    if first_transcript.is_translatable:
                        translated = first_transcript.translate('en')
                        raw_transcript = translated.fetch()
                        cleaned_text = self._clean_transcript(raw_transcript)
                        logger.info(f"Successfully fetched translated transcript for {video_id} from {first_transcript.language_code}")
                        return cleaned_text
            except (NotTranslatable, TranslationLanguageNotAvailable) as e:
                logger.warning(f"Translation not available for {video_id}: {e}")
            except Exception as e:
                logger.warning(f"Error translating transcript for {video_id}: {e}")
            
            logger.warning(f"No suitable transcript found for video {video_id}")
            return None
            
        except TranscriptsDisabled:
            logger.warning(f"Transcripts are disabled for video {video_id}")
            return None
        except VideoUnavailable:
            logger.warning(f"Video {video_id} is unavailable")
            return None
        except YouTubeRequestFailed:
            logger.error(f"YouTube request failed when fetching transcript for {video_id}")
            return None
        except CouldNotRetrieveTranscript:
            logger.warning(f"Could not retrieve transcript for video {video_id}")
            return None
        except (CookiePathInvalid, FailedToCreateConsentCookie) as e:
            logger.error(f"Cookie-related error for video {video_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching transcript for {video_id}: {e}")
            return None
    
    def _clean_transcript(self, raw_transcript: List[Dict[str, Any]]) -> str:
        """
        Clean and parse raw transcript data into readable text.
        
        Args:
            raw_transcript: Raw transcript data from YouTube API
            
        Returns:
            Cleaned transcript text
        """
        if not raw_transcript:
            return ""
        
        # Extract text from transcript segments
        text_segments = []
        for segment in raw_transcript:
            text = segment.get('text', '').strip()
            if text:
                text_segments.append(text)
        
        # Join segments and clean the text
        full_text = ' '.join(text_segments)
        
        # Clean the text
        cleaned_text = self._clean_text(full_text)
        
        return cleaned_text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean transcript text by removing artifacts and normalizing formatting.
        
        Args:
            text: Raw transcript text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove common transcript artifacts
        # Remove music notation like [Music], [Applause], etc.
        text = re.sub(r'\[[\w\s]+\]', '', text)
        
        # Remove speaker labels like "Speaker 1:", "John:", etc.
        text = re.sub(r'^[A-Za-z\s\d]+:\s*', '', text, flags=re.MULTILINE)
        
        # Remove timestamps and time markers
        text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?', '', text)
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove common filler words and artifacts at the beginning/end
        text = re.sub(r'^(um|uh|so|well|okay|alright)\s+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+(um|uh|you know|like)\s*$', '', text, flags=re.IGNORECASE)
        
        # Ensure proper sentence capitalization
        sentences = text.split('. ')
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Capitalize first letter of each sentence
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                cleaned_sentences.append(sentence)
        
        return '. '.join(cleaned_sentences)
    
    def process_videos_transcripts(self, state: CuratorState) -> CuratorState:
        """
        Process transcripts for all discovered videos in the state.
        
        This method updates the VideoData objects in state.discovered_videos
        with transcript content where available.
        
        Args:
            state: Current curator state with discovered videos
            
        Returns:
            Updated state with transcript data
        """
        if not state.discovered_videos:
            logger.warning("No discovered videos to process transcripts for")
            return state
        
        logger.info(f"Processing transcripts for {len(state.discovered_videos)} videos")
        
        successful_transcripts = 0
        failed_transcripts = 0
        
        for video in state.discovered_videos:
            try:
                transcript = self.fetch_transcript(video.video_id)
                if transcript:
                    video.transcript = transcript
                    successful_transcripts += 1
                    logger.debug(f"Successfully processed transcript for {video.video_id}: {len(transcript)} characters")
                else:
                    video.transcript = None
                    failed_transcripts += 1
                    logger.debug(f"No transcript available for {video.video_id}")
                    
            except Exception as e:
                video.transcript = None
                failed_transcripts += 1
                error_msg = f"Error processing transcript for {video.video_id}: {e}"
                logger.error(error_msg)
                state.add_error(error_msg, "transcript_processor")
        
        # Update metadata
        state.update_generation_metadata(
            transcripts_processed=len(state.discovered_videos),
            transcripts_successful=successful_transcripts,
            transcripts_failed=failed_transcripts,
            transcript_success_rate=successful_transcripts / len(state.discovered_videos) if state.discovered_videos else 0
        )
        
        logger.info(f"Transcript processing complete: {successful_transcripts} successful, {failed_transcripts} failed")
        
        return state
    
    def get_transcript_statistics(self, videos: List[VideoData]) -> Dict[str, Any]:
        """
        Get statistics about transcript availability and quality.
        
        Args:
            videos: List of VideoData objects
            
        Returns:
            Dictionary with transcript statistics
        """
        total_videos = len(videos)
        if total_videos == 0:
            return {
                'total_videos': 0,
                'with_transcripts': 0,
                'without_transcripts': 0,
                'availability_rate': 0.0,
                'average_length': 0,
                'total_words': 0
            }
        
        with_transcripts = [v for v in videos if v.transcript]
        without_transcripts = [v for v in videos if not v.transcript]
        
        # Calculate transcript lengths
        transcript_lengths = [len(v.transcript) for v in with_transcripts]
        word_counts = [len(v.transcript.split()) for v in with_transcripts]
        
        return {
            'total_videos': total_videos,
            'with_transcripts': len(with_transcripts),
            'without_transcripts': len(without_transcripts),
            'availability_rate': len(with_transcripts) / total_videos,
            'average_length': sum(transcript_lengths) / len(transcript_lengths) if transcript_lengths else 0,
            'average_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
            'total_words': sum(word_counts),
            'min_length': min(transcript_lengths) if transcript_lengths else 0,
            'max_length': max(transcript_lengths) if transcript_lengths else 0
        }


def fetch_transcripts_node(state: CuratorState) -> CuratorState:
    """
    LangGraph node function for fetching transcripts.
    
    This function is designed to be used as a node in the LangGraph workflow
    and processes transcripts for all discovered videos.
    
    Args:
        state: Current curator state
        
    Returns:
        Updated state with transcript data
    """
    logger.info("Starting transcript fetching node")
    
    try:
        processor = TranscriptProcessor()
        updated_state = processor.process_videos_transcripts(state)
        
        # Log summary statistics
        stats = processor.get_transcript_statistics(updated_state.discovered_videos)
        logger.info(f"Transcript fetching complete: {stats['availability_rate']:.1%} success rate "
                   f"({stats['with_transcripts']}/{stats['total_videos']} videos)")
        
        return updated_state
        
    except Exception as e:
        error_msg = f"Critical error in transcript fetching node: {e}"
        logger.error(error_msg)
        state.add_error(error_msg, "fetch_transcripts_node")
        return state