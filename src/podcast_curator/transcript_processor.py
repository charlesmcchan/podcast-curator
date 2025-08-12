"""
Transcript processing module for the podcast-curator system.

This module handles fetching, cleaning, and parsing YouTube video transcripts
using the youtube-transcript-api library with robust error handling and fallback mechanisms.
"""

import logging
import re
import time
import random
from typing import List, Optional, Dict, Any
from collections import Counter
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
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
from .config import get_config, Configuration

# Configure logging
logger = logging.getLogger(__name__)


class TranscriptProcessor:
    """
    Handles YouTube transcript fetching and processing with comprehensive error handling.
    
    This class provides methods to fetch transcripts from YouTube videos, clean and parse
    the text content, and handle various error conditions gracefully.
    """
    
    def __init__(self, preferred_languages: List[str] = None, config: Optional[Configuration] = None):
        """
        Initialize the transcript processor.
        
        Args:
            preferred_languages: List of preferred language codes (e.g., ['en', 'en-US'])
                                Defaults to English variants if not specified.
            config: Optional configuration instance. If not provided, attempts to get global config.
        """
        self.preferred_languages = preferred_languages or ['en', 'en-US', 'en-GB', 'en-CA']
        self.fallback_languages = ['en-auto', 'auto']  # Auto-generated transcripts as fallback
        self.config = config or get_config()
        
        # User agents to rotate through to appear less bot-like
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        ]
        self.current_user_agent_index = 0
        
        # Initialize YouTube API with proxy config from configuration
        if self.config.proxy_username and self.config.proxy_password:
            self.youtube_api = YouTubeTranscriptApi(
                proxy_config=WebshareProxyConfig(
                    proxy_username=self.config.proxy_username,
                    proxy_password=self.config.proxy_password,
                )
            )
        else:
            # Use without proxy if credentials not provided
            self.youtube_api = YouTubeTranscriptApi()

    def _rotate_user_agent(self):
        """Rotate to the next user agent to appear less bot-like."""
        self.current_user_agent_index = (self.current_user_agent_index + 1) % len(self.user_agents)
        # Note: YouTubeTranscriptApi doesn't directly support user agent changes
        # This is mainly for future extensibility if we need to add custom headers
        return self.user_agents[self.current_user_agent_index]
        
    def fetch_transcript(self, video_id: str, max_retries: int = 3) -> Optional[str]:
        """
        Fetch transcript for a single video with comprehensive error handling and retry logic.
        
        Args:
            video_id: YouTube video ID
            max_retries: Maximum number of retries for IP blocking
            
        Returns:
            Cleaned transcript text or None if unavailable
        """
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff with jitter
                    delay = (2 ** attempt) + random.uniform(0.5, 2.0)
                    # Rotate user agent on retry (for future extensibility)
                    user_agent = self._rotate_user_agent()
                    logger.debug(f"Retry {attempt} for {video_id}, waiting {delay:.1f}s with new user agent")
                    time.sleep(delay)
                
                return self._fetch_transcript_attempt(video_id)
                
            except (YouTubeRequestFailed, CouldNotRetrieveTranscript) as e:
                if "blocking" in str(e).lower() or "ip" in str(e).lower():
                    if attempt < max_retries:
                        logger.warning(f"IP blocking detected for {video_id}, retrying ({attempt+1}/{max_retries})")
                        continue
                    else:
                        logger.error(f"Failed to fetch transcript for {video_id} after {max_retries} retries: IP blocked")
                        return None
                else:
                    # Non-blocking error, don't retry
                    logger.warning(f"Non-retryable error for {video_id}: {e}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error fetching transcript for {video_id}: {e}")
                return None
        
        return None
    
    def _fetch_transcript_attempt(self, video_id: str) -> Optional[str]:
        """
        Single attempt to fetch transcript without retry logic.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Cleaned transcript text or None if unavailable
        """
        try:
            # First, try to get manually created transcripts in preferred languages
            transcript_list = self.youtube_api.list(video_id)
            
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
        except YouTubeRequestFailed as e:
            # Re-raise for retry logic to handle
            raise e
        except CouldNotRetrieveTranscript as e:
            # Re-raise for retry logic to handle
            raise e
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
            # Handle both old dict format and new FetchedTranscriptSnippet objects
            if hasattr(segment, 'text'):
                text = segment.text.strip()
            else:
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
    
    def extract_key_topics(self, transcript: str) -> List[str]:
        """
        Extract key topics from transcript text using keyword frequency and AI-related terms.
        
        Args:
            transcript: Cleaned transcript text
            
        Returns:
            List of key topics/keywords found in the transcript
        """
        if not transcript:
            return []
        
        # Get configurable keywords from user config
        search_keywords = set(keyword.lower() for keyword in self.config.default_search_keywords)
        
        # Convert to lowercase for matching
        text_lower = transcript.lower()
        
        # Find keywords from user configuration
        found_topics = set()
        for keyword in search_keywords:
            if keyword in text_lower:
                found_topics.add(keyword)
        
        # Extract frequent meaningful words (nouns, adjectives, technical terms)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', transcript.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him',
            'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way',
            'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too',
            'use', 'that', 'with', 'have', 'this', 'will', 'your', 'from',
            'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time',
            'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make',
            'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were',
            'what', 'would', 'about', 'after', 'before', 'other', 'right',
            'think', 'where', 'being', 'every', 'first', 'going', 'look',
            'made', 'most', 'people', 'should', 'these', 'things', 'through',
            'work', 'years', 'actually', 'really', 'something', 'because',
            'there', 'their', 'could', 'said', 'each', 'which', 'doing',
            'into', 'only', 'also', 'back', 'call', 'came', 'same', 'find',
            'great', 'little', 'might', 'never', 'still', 'those', 'under',
            'while', 'another', 'around', 'between', 'different', 'important',
            'kind', 'need', 'part', 'place', 'small', 'system', 'world',
            'yeah', 'okay', 'alright', 'basically', 'obviously', 'definitely'
        }
        
        # Count word frequencies
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        word_counts = Counter(filtered_words)
        
        # Get most frequent words that might be topics
        frequent_words = [word for word, count in word_counts.most_common(20) if count >= 2]
        
        # Add frequent words that seem technical or relevant
        for word in frequent_words:
            if any(tech_term in word for tech_term in ['tech', 'data', 'model', 'system', 'network', 'algorithm']):
                found_topics.add(word)
        
        # Convert to sorted list and limit to top topics
        topics = sorted(list(found_topics))[:15]  # Limit to 15 most relevant topics
        
        return topics
    
    def analyze_content_quality(self, transcript: str) -> Dict[str, float]:
        """
        Analyze content quality based on coherence and information density.
        
        Args:
            transcript: Cleaned transcript text
            
        Returns:
            Dictionary with quality metrics (0-100 scale)
        """
        if not transcript:
            return {
                'coherence_score': 0.0,
                'information_density': 0.0,
                'technical_depth': 0.0,
                'overall_quality': 0.0
            }
        
        # Calculate coherence score
        coherence_score = self._calculate_coherence(transcript)
        
        # Calculate information density
        information_density = self._calculate_information_density(transcript)
        
        # Calculate technical depth
        technical_depth = self._calculate_technical_depth(transcript)
        
        # Calculate overall quality score
        overall_quality = (coherence_score * 0.4 + information_density * 0.3 + technical_depth * 0.3)
        
        return {
            'coherence_score': coherence_score,
            'information_density': information_density,
            'technical_depth': technical_depth,
            'overall_quality': overall_quality
        }
    
    def _calculate_coherence(self, transcript: str) -> float:
        """
        Calculate transcript coherence based on sentence structure and flow.
        
        Args:
            transcript: Cleaned transcript text
            
        Returns:
            Coherence score (0-100)
        """
        sentences = [s.strip() for s in transcript.split('.') if s.strip()]
        if len(sentences) < 2:
            return 50.0  # Neutral score for very short content
        
        coherence_indicators = 0
        total_checks = 0
        
        # Check for transition words and phrases
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'meanwhile', 'similarly', 'likewise', 'in contrast',
            'on the other hand', 'for example', 'for instance', 'specifically',
            'in particular', 'as a result', 'in conclusion', 'to summarize',
            'first', 'second', 'third', 'finally', 'next', 'then', 'also',
            'furthermore', 'besides', 'in addition', 'what\'s more'
        }
        
        text_lower = transcript.lower()
        transition_count = sum(1 for word in transition_words if word in text_lower)
        transition_score = min(transition_count / len(sentences) * 100, 100)
        
        # Check average sentence length (not too short, not too long)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        length_score = 100 - abs(avg_sentence_length - 15) * 2  # Optimal around 15 words
        length_score = max(0, min(100, length_score))
        
        # Check for repetitive patterns (lower score for high repetition)
        words = transcript.lower().split()
        if len(words) > 10:
            unique_words = len(set(words))
            repetition_score = (unique_words / len(words)) * 100
        else:
            repetition_score = 50.0
        
        # Combine scores
        coherence_score = (transition_score * 0.4 + length_score * 0.3 + repetition_score * 0.3)
        
        return max(0, min(100, coherence_score))
    
    def _calculate_information_density(self, transcript: str) -> float:
        """
        Calculate information density based on unique concepts and technical terms.
        
        Args:
            transcript: Cleaned transcript text
            
        Returns:
            Information density score (0-100)
        """
        words = transcript.lower().split()
        if len(words) < 10:
            return 30.0  # Low score for very short content
        
        # Count unique meaningful words (excluding stop words)
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him',
            'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way',
            'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too',
            'use', 'that', 'with', 'have', 'this', 'will', 'your', 'from',
            'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time',
            'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make',
            'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'
        }
        
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 3]
        unique_meaningful = len(set(meaningful_words))
        
        # Calculate lexical diversity
        lexical_diversity = unique_meaningful / len(meaningful_words) if meaningful_words else 0
        
        # Count technical/domain-specific terms
        technical_terms = {
            'algorithm', 'model', 'training', 'neural', 'network', 'learning',
            'artificial', 'intelligence', 'machine', 'deep', 'data', 'analysis',
            'processing', 'computer', 'vision', 'language', 'natural', 'api',
            'framework', 'library', 'python', 'tensorflow', 'pytorch', 'research',
            'paper', 'study', 'experiment', 'results', 'performance', 'accuracy',
            'optimization', 'parameters', 'hyperparameters', 'architecture',
            'transformer', 'attention', 'embedding', 'tokenization', 'inference'
        }
        
        technical_count = sum(1 for word in meaningful_words if word in technical_terms)
        technical_density = (technical_count / len(meaningful_words)) * 100 if meaningful_words else 0
        
        # Combine metrics
        density_score = (lexical_diversity * 60 + min(technical_density, 40))
        
        return max(0, min(100, density_score))
    
    def _calculate_technical_depth(self, transcript: str) -> float:
        """
        Calculate technical depth based on presence of technical concepts and explanations.
        
        Args:
            transcript: Cleaned transcript text
            
        Returns:
            Technical depth score (0-100)
        """
        text_lower = transcript.lower()
        
        # Advanced technical terms
        advanced_terms = {
            'transformer', 'attention mechanism', 'backpropagation', 'gradient descent',
            'convolutional', 'recurrent', 'lstm', 'gru', 'bert', 'gpt', 'fine-tuning',
            'hyperparameter', 'regularization', 'dropout', 'batch normalization',
            'activation function', 'loss function', 'optimizer', 'learning rate',
            'overfitting', 'underfitting', 'cross-validation', 'ensemble',
            'reinforcement learning', 'supervised learning', 'unsupervised learning',
            'semi-supervised', 'transfer learning', 'few-shot learning', 'zero-shot',
            'embedding', 'vector space', 'dimensionality reduction', 'clustering',
            'classification', 'regression', 'neural architecture search',
            'autoencoder', 'generative adversarial', 'diffusion model', 'variational'
        }
        
        # Count advanced terms
        advanced_count = sum(1 for term in advanced_terms if term in text_lower)
        
        # Look for explanatory patterns
        explanation_patterns = [
            r'this means that',
            r'in other words',
            r'for example',
            r'specifically',
            r'the reason is',
            r'because of',
            r'this is important because',
            r'the key insight',
            r'what this shows',
            r'the implication'
        ]
        
        explanation_count = sum(1 for pattern in explanation_patterns 
                              if re.search(pattern, text_lower))
        
        # Look for numerical/quantitative information
        numbers_pattern = r'\b\d+(?:\.\d+)?(?:%|percent|million|billion|thousand)?\b'
        numerical_mentions = len(re.findall(numbers_pattern, text_lower))
        
        # Calculate technical depth score
        word_count = len(transcript.split())
        if word_count == 0:
            return 0.0
        
        # Normalize scores
        advanced_score = min((advanced_count / word_count) * 1000, 40)  # Max 40 points
        explanation_score = min((explanation_count / word_count) * 2000, 30)  # Max 30 points
        numerical_score = min((numerical_mentions / word_count) * 1000, 30)  # Max 30 points
        
        technical_depth = advanced_score + explanation_score + numerical_score
        
        return max(0, min(100, technical_depth))
    
    def detect_technical_accuracy_indicators(self, transcript: str) -> Dict[str, Any]:
        """
        Detect indicators of technical accuracy in the transcript.
        
        Args:
            transcript: Cleaned transcript text
            
        Returns:
            Dictionary with technical accuracy indicators
        """
        if not transcript:
            return {
                'has_citations': False,
                'mentions_sources': False,
                'uses_precise_terminology': False,
                'shows_uncertainty_awareness': False,
                'provides_context': False,
                'accuracy_score': 0.0
            }
        
        text_lower = transcript.lower()
        
        # Check for citations and references
        citation_patterns = [
            r'according to',
            r'research shows',
            r'study found',
            r'paper published',
            r'researchers at',
            r'university of',
            r'journal of',
            r'arxiv',
            r'published in',
            r'peer reviewed'
        ]
        
        has_citations = any(re.search(pattern, text_lower) for pattern in citation_patterns)
        
        # Check for source mentions
        source_patterns = [
            r'openai',
            r'google',
            r'microsoft',
            r'anthropic',
            r'meta',
            r'facebook',
            r'deepmind',
            r'nvidia',
            r'hugging face',
            r'stanford',
            r'mit',
            r'berkeley',
            r'carnegie mellon'
        ]
        
        mentions_sources = any(pattern in text_lower for pattern in source_patterns)
        
        # Check for precise terminology usage
        precise_terms = {
            'specifically', 'precisely', 'exactly', 'approximately', 'roughly',
            'about', 'around', 'nearly', 'close to', 'in the range of',
            'between', 'from', 'to', 'up to', 'as much as', 'at least'
        }
        
        uses_precise_terminology = any(term in text_lower for term in precise_terms)
        
        # Check for uncertainty awareness
        uncertainty_phrases = [
            'might be', 'could be', 'possibly', 'potentially', 'likely',
            'probably', 'seems to', 'appears to', 'suggests that',
            'indicates that', 'preliminary', 'early results', 'initial findings',
            'more research needed', 'further investigation', 'not yet clear',
            'remains to be seen', 'unclear', 'uncertain'
        ]
        
        shows_uncertainty_awareness = any(phrase in text_lower for phrase in uncertainty_phrases)
        
        # Check for context provision
        context_phrases = [
            'background', 'context', 'historically', 'previously', 'in the past',
            'traditionally', 'compared to', 'in contrast', 'unlike', 'similar to',
            'building on', 'based on', 'following', 'as a result of'
        ]
        
        provides_context = any(phrase in text_lower for phrase in context_phrases)
        
        # Calculate overall accuracy score
        indicators = [
            has_citations,
            mentions_sources,
            uses_precise_terminology,
            shows_uncertainty_awareness,
            provides_context
        ]
        
        accuracy_score = (sum(indicators) / len(indicators)) * 100
        
        return {
            'has_citations': has_citations,
            'mentions_sources': mentions_sources,
            'uses_precise_terminology': uses_precise_terminology,
            'shows_uncertainty_awareness': shows_uncertainty_awareness,
            'provides_context': provides_context,
            'accuracy_score': accuracy_score
        }
    
    def extract_main_points_and_details(self, transcript: str) -> Dict[str, List[str]]:
        """
        Extract main points and technical details from the transcript.
        
        Args:
            transcript: Cleaned transcript text
            
        Returns:
            Dictionary with main points and technical details
        """
        if not transcript:
            return {
                'main_points': [],
                'technical_details': [],
                'key_insights': [],
                'actionable_items': []
            }
        
        sentences = [s.strip() for s in transcript.split('.') if s.strip() and len(s.strip()) > 10]
        
        main_points = []
        technical_details = []
        key_insights = []
        actionable_items = []
        
        # Patterns for identifying main points
        main_point_indicators = [
            r'^(the main|the key|the important|the primary|the central)',
            r'(most important|key point|main idea|central concept)',
            r'^(first|second|third|finally|in conclusion)',
            r'(what this means|the takeaway|the bottom line)'
        ]
        
        # Patterns for technical details
        technical_indicators = [
            r'(algorithm|model|architecture|framework|implementation)',
            r'(parameters|hyperparameters|configuration|settings)',
            r'(training|inference|optimization|performance)',
            r'(accuracy|precision|recall|f1|score|metric)',
            r'(\d+(?:\.\d+)?(?:%|percent|million|billion))'
        ]
        
        # Patterns for insights
        insight_indicators = [
            r'(this shows|this demonstrates|this reveals|this suggests)',
            r'(the insight|the discovery|the finding|the result)',
            r'(surprisingly|interestingly|notably|remarkably)',
            r'(breakthrough|innovation|advancement|improvement)'
        ]
        
        # Patterns for actionable items
        action_indicators = [
            r'(you can|you should|you need to|you might want to)',
            r'(to implement|to use|to apply|to try)',
            r'(recommendation|suggestion|advice|tip)',
            r'(next step|action item|todo|follow up)'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for main points
            if any(re.search(pattern, sentence_lower) for pattern in main_point_indicators):
                main_points.append(sentence)
            
            # Check for technical details
            elif any(re.search(pattern, sentence_lower) for pattern in technical_indicators):
                technical_details.append(sentence)
            
            # Check for insights
            elif any(re.search(pattern, sentence_lower) for pattern in insight_indicators):
                key_insights.append(sentence)
            
            # Check for actionable items
            elif any(re.search(pattern, sentence_lower) for pattern in action_indicators):
                actionable_items.append(sentence)
            
            # If sentence is long and contains important keywords, consider it a main point
            elif len(sentence.split()) > 15 and any(keyword in sentence_lower 
                                                   for keyword in ['artificial intelligence', 'machine learning', 'ai', 'model', 'algorithm']):
                main_points.append(sentence)
        
        # Limit results to most relevant items
        return {
            'main_points': main_points[:5],  # Top 5 main points
            'technical_details': technical_details[:8],  # Top 8 technical details
            'key_insights': key_insights[:4],  # Top 4 insights
            'actionable_items': actionable_items[:3]  # Top 3 actionable items
        }
    
    def analyze_transcript(self, video: VideoData) -> VideoData:
        """
        Perform comprehensive transcript analysis on a single video.
        
        Args:
            video: VideoData object with transcript
            
        Returns:
            Updated VideoData object with analysis results
        """
        if not video.transcript:
            logger.warning(f"No transcript available for analysis: {video.video_id}")
            return video
        
        try:
            # Extract key topics
            video.key_topics = self.extract_key_topics(video.transcript)
            
            # Analyze content quality
            quality_metrics = self.analyze_content_quality(video.transcript)
            
            # Detect technical accuracy indicators
            accuracy_indicators = self.detect_technical_accuracy_indicators(video.transcript)
            
            # Extract main points and details
            content_structure = self.extract_main_points_and_details(video.transcript)
            
            # Store analysis results in video object (using extra fields allowed by model)
            video.content_quality = quality_metrics
            video.accuracy_indicators = accuracy_indicators
            video.content_structure = content_structure
            
            # Calculate a combined content analysis score
            content_score = (
                quality_metrics['overall_quality'] * 0.6 +
                accuracy_indicators['accuracy_score'] * 0.4
            )
            
            video.content_analysis_score = content_score
            
            logger.info(f"Transcript analysis complete for {video.video_id}: "
                       f"topics={len(video.key_topics)}, quality={quality_metrics['overall_quality']:.1f}, "
                       f"accuracy={accuracy_indicators['accuracy_score']:.1f}")
            
        except Exception as e:
            logger.error(f"Error analyzing transcript for {video.video_id}: {e}")
            # Set default values on error
            video.key_topics = []
            video.content_quality = {'overall_quality': 0.0}
            video.accuracy_indicators = {'accuracy_score': 0.0}
            video.content_structure = {'main_points': [], 'technical_details': []}
            video.content_analysis_score = 0.0
        
        return video
    
    def process_videos_transcripts(self, state: CuratorState) -> CuratorState:
        """
        Process transcripts for all discovered videos in the state.
        
        This method updates the VideoData objects in state.discovered_videos
        with transcript content and performs comprehensive transcript analysis.
        
        Args:
            state: Current curator state with discovered videos
            
        Returns:
            Updated state with transcript data and analysis
        """
        if not state.discovered_videos:
            logger.warning("No discovered videos to process transcripts for")
            return state
        
        logger.info(f"Processing transcripts and analysis for {len(state.discovered_videos)} videos")
        
        successful_transcripts = 0
        failed_transcripts = 0
        successful_analysis = 0
        
        for video in state.discovered_videos:
            try:
                # Fetch transcript
                transcript = self.fetch_transcript(video.video_id)
                if transcript:
                    video.transcript = transcript
                    successful_transcripts += 1
                    logger.debug(f"Successfully fetched transcript for {video.video_id}: {len(transcript)} characters")
                    
                    # Perform transcript analysis
                    try:
                        video = self.analyze_transcript(video)
                        successful_analysis += 1
                        logger.debug(f"Successfully analyzed transcript for {video.video_id}")
                    except Exception as analysis_error:
                        error_msg = f"Error analyzing transcript for {video.video_id}: {analysis_error}"
                        logger.error(error_msg)
                        state.add_error(error_msg, "transcript_analysis")
                        # Continue processing other videos even if analysis fails
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
            transcript_success_rate=successful_transcripts / len(state.discovered_videos) if state.discovered_videos else 0,
            transcript_analysis_successful=successful_analysis,
            transcript_analysis_rate=successful_analysis / successful_transcripts if successful_transcripts > 0 else 0
        )
        
        logger.info(f"Transcript processing complete: {successful_transcripts} transcripts fetched, "
                   f"{successful_analysis} analyzed, {failed_transcripts} failed")
        
        return state
    
    def get_transcript_statistics(self, videos: List[VideoData]) -> Dict[str, Any]:
        """
        Get statistics about transcript availability, quality, and analysis results.
        
        Args:
            videos: List of VideoData objects
            
        Returns:
            Dictionary with comprehensive transcript statistics
        """
        total_videos = len(videos)
        if total_videos == 0:
            return {
                'total_videos': 0,
                'with_transcripts': 0,
                'without_transcripts': 0,
                'availability_rate': 0.0,
                'average_length': 0,
                'total_words': 0,
                'with_analysis': 0,
                'analysis_rate': 0.0,
                'average_topics': 0,
                'average_quality_score': 0.0,
                'average_accuracy_score': 0.0
            }
        
        with_transcripts = [v for v in videos if v.transcript]
        without_transcripts = [v for v in videos if not v.transcript]
        with_analysis = [v for v in with_transcripts if hasattr(v, 'content_analysis_score')]
        
        # Calculate transcript lengths
        transcript_lengths = [len(v.transcript) for v in with_transcripts]
        word_counts = [len(v.transcript.split()) for v in with_transcripts]
        
        # Calculate analysis statistics
        topic_counts = [len(v.key_topics) for v in with_analysis if hasattr(v, 'key_topics')]
        quality_scores = [getattr(v, 'content_analysis_score', 0) for v in with_analysis]
        accuracy_scores = []
        
        for v in with_analysis:
            if hasattr(v, 'accuracy_indicators') and isinstance(v.accuracy_indicators, dict):
                accuracy_scores.append(v.accuracy_indicators.get('accuracy_score', 0))
        
        return {
            'total_videos': total_videos,
            'with_transcripts': len(with_transcripts),
            'without_transcripts': len(without_transcripts),
            'availability_rate': len(with_transcripts) / total_videos,
            'average_length': sum(transcript_lengths) / len(transcript_lengths) if transcript_lengths else 0,
            'average_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
            'total_words': sum(word_counts),
            'min_length': min(transcript_lengths) if transcript_lengths else 0,
            'max_length': max(transcript_lengths) if transcript_lengths else 0,
            'with_analysis': len(with_analysis),
            'analysis_rate': len(with_analysis) / len(with_transcripts) if with_transcripts else 0,
            'average_topics': sum(topic_counts) / len(topic_counts) if topic_counts else 0,
            'average_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'average_accuracy_score': sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
        }


def fetch_transcripts_node(state: CuratorState) -> CuratorState:
    """
    LangGraph node function for fetching and analyzing transcripts.
    
    This function is designed to be used as a node in the LangGraph workflow
    and processes transcripts for all discovered videos, including comprehensive
    content analysis for quality evaluation.
    
    Args:
        state: Current curator state
        
    Returns:
        Updated state with transcript data and analysis results
    """
    logger.info("Starting transcript fetching and analysis node")
    
    try:
        processor = TranscriptProcessor()
        updated_state = processor.process_videos_transcripts(state)
        
        # Log summary statistics
        stats = processor.get_transcript_statistics(updated_state.discovered_videos)
        logger.info(f"Transcript processing complete: {stats['availability_rate']:.1%} fetch rate "
                   f"({stats['with_transcripts']}/{stats['total_videos']} videos), "
                   f"{stats['analysis_rate']:.1%} analysis rate "
                   f"({stats['with_analysis']}/{stats['with_transcripts']} analyzed)")
        
        if stats['with_analysis'] > 0:
            logger.info(f"Analysis summary: avg topics={stats['average_topics']:.1f}, "
                       f"avg quality={stats['average_quality_score']:.1f}, "
                       f"avg accuracy={stats['average_accuracy_score']:.1f}")
        
        return updated_state
        
    except Exception as e:
        error_msg = f"Critical error in transcript fetching and analysis node: {e}"
        logger.error(error_msg)
        state.add_error(error_msg, "fetch_transcripts_node")
        return state