"""
OpenAI integration for podcast script generation.

This module provides OpenAI API client functionality with authentication,
prompt templates, error handling, and response validation for generating
podcast scripts from curated video content.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field, ValidationError

from .models import VideoData, CuratorState
from .config import get_config

logger = logging.getLogger(__name__)


class ScriptGenerationRequest(BaseModel):
    """Request model for script generation."""
    model_config = {"arbitrary_types_allowed": True}
    
    videos: List[VideoData] = Field(..., description="Videos to use for script generation")
    target_word_count_min: int = Field(default=750, ge=100, description="Minimum word count")
    target_word_count_max: int = Field(default=1500, ge=500, description="Maximum word count")
    language: str = Field(default="en", description="Language code for script")
    style: str = Field(default="conversational", description="Script style")


class ScriptGenerationResponse(BaseModel):
    """Response model for script generation."""
    script: str = Field(..., description="Generated podcast script")
    word_count: int = Field(..., description="Actual word count")
    estimated_duration_minutes: float = Field(..., description="Estimated speaking duration")
    source_videos: List[str] = Field(..., description="Video IDs used as sources")
    generation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")


class OpenAIScriptGenerator:
    """
    OpenAI-powered podcast script generator.
    
    Handles authentication, prompt templating, API calls with retry logic,
    and response validation for generating podcast scripts from video content.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI script generator.
        
        Args:
            api_key: Optional OpenAI API key. If not provided, uses config.
        """
        config = get_config()
        self.api_key = api_key or config.openai_api_key
        self.client = OpenAI(api_key=self.api_key)
        
        # Configuration
        self.model = "gpt-4o-mini"  # Cost-effective model for script generation
        self.max_retries = 3
        self.retry_delay = 1.0  # Initial delay in seconds
        self.max_tokens = 4000  # Sufficient for 1500-word scripts
        self.temperature = 0.7  # Balance creativity and consistency
        
        logger.info(f"Initialized OpenAI script generator with model: {self.model}")
    
    def generate_script(self, request: ScriptGenerationRequest) -> ScriptGenerationResponse:
        """
        Generate a podcast script from video content.
        
        Args:
            request: Script generation request with videos and parameters
            
        Returns:
            ScriptGenerationResponse with generated script and metadata
            
        Raises:
            ValueError: If request validation fails
            RuntimeError: If script generation fails after retries
        """
        if not request.videos:
            raise ValueError("At least one video is required for script generation")
        
        # Validate videos have required content
        valid_videos = [v for v in request.videos if v.transcript and v.transcript.strip()]
        if not valid_videos:
            raise ValueError("At least one video must have a transcript for script generation")
        
        logger.info(f"Generating script from {len(valid_videos)} videos")
        
        # Generate the script with retry logic
        script_content = self._generate_with_retry(request, valid_videos)
        
        # Validate and process the response
        response = self._process_script_response(script_content, valid_videos, request)
        
        logger.info(f"Generated script: {response.word_count} words, "
                   f"{response.estimated_duration_minutes:.1f} minutes")
        
        return response
    
    def _generate_with_retry(self, request: ScriptGenerationRequest, videos: List[VideoData]) -> str:
        """
        Generate script with exponential backoff retry logic.
        
        Args:
            request: Script generation request
            videos: Valid videos with transcripts
            
        Returns:
            Generated script content
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Script generation attempt {attempt + 1}/{self.max_retries}")
                
                # Create the prompt
                prompt = self._create_script_prompt(request, videos)
                
                # Make the API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt(request)
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                    frequency_penalty=0.1,
                    presence_penalty=0.1
                )
                
                # Extract the script content
                script_content = response.choices[0].message.content
                if not script_content or not script_content.strip():
                    raise ValueError("OpenAI returned empty script content")
                
                return script_content.strip()
                
            except Exception as e:
                last_error = e
                logger.warning(f"Script generation attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} script generation attempts failed")
        
        raise RuntimeError(f"Script generation failed after {self.max_retries} attempts. "
                          f"Last error: {str(last_error)}")
    
    def _get_system_prompt(self, request: ScriptGenerationRequest) -> str:
        """
        Get the system prompt for script generation.
        
        Args:
            request: Script generation request
            
        Returns:
            System prompt string
        """
        return f"""You are an expert podcast script writer specializing in AI and technology content. 
Your task is to create engaging, informative podcast scripts that synthesize information from multiple YouTube videos.

REQUIREMENTS:
- Target length: {request.target_word_count_min}-{request.target_word_count_max} words
- Style: {request.style} and engaging
- Language: {request.language}
- Audience: Tech-savvy professionals interested in AI developments

STRUCTURE:
1. Hook/Introduction (10-15% of content)
2. Main content with smooth transitions between topics (70-80% of content)
3. Key takeaways and conclusion (10-15% of content)

GUIDELINES:
- Create a cohesive narrative that flows naturally between video topics
- Include specific insights and technical details from the source videos
- Use conversational tone suitable for audio consumption
- Add smooth transitions between different topics/sources
- Cite sources naturally within the narrative (e.g., "According to [Channel Name]...")
- Focus on the most valuable and actionable insights
- Avoid repetitive information across sources
- Make complex topics accessible without oversimplifying

OUTPUT FORMAT:
Provide only the script content without additional formatting, headers, or metadata."""
    
    def _create_script_prompt(self, request: ScriptGenerationRequest, videos: List[VideoData]) -> str:
        """
        Create the user prompt with video content.
        
        Args:
            request: Script generation request
            videos: Videos to include in the script
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"Create a {request.target_word_count_min}-{request.target_word_count_max} word podcast script from the following video content:",
            ""
        ]
        
        # Add video content
        for i, video in enumerate(videos, 1):
            prompt_parts.extend([
                f"VIDEO {i}: {video.title}",
                f"Channel: {video.channel}",
                f"Views: {video.view_count:,}",
                f"Quality Score: {video.quality_score:.1f}/100" if video.quality_score else "Quality Score: N/A",
                f"Key Topics: {', '.join(video.key_topics)}" if video.key_topics else "Key Topics: N/A",
                "",
                "TRANSCRIPT:",
                self._truncate_transcript(video.transcript, max_words=800),  # Limit transcript length
                "",
                "---",
                ""
            ])
        
        prompt_parts.extend([
            "INSTRUCTIONS:",
            "- Synthesize the most valuable insights from these videos into a cohesive narrative",
            "- Focus on recent developments, practical applications, and key takeaways",
            "- Maintain a conversational tone suitable for podcast listening",
            "- Include natural source attribution throughout the script",
            "- Ensure smooth transitions between topics from different videos",
            f"- Target exactly {request.target_word_count_min}-{request.target_word_count_max} words",
            "",
            "Generate the podcast script now:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _truncate_transcript(self, transcript: str, max_words: int = 800) -> str:
        """
        Truncate transcript to fit within token limits while preserving meaning.
        
        Args:
            transcript: Full transcript text
            max_words: Maximum number of words to include
            
        Returns:
            Truncated transcript
        """
        if not transcript:
            return ""
        
        words = transcript.split()
        if len(words) <= max_words:
            return transcript
        
        # Take first portion and add ellipsis
        truncated = " ".join(words[:max_words])
        return f"{truncated}... [transcript truncated for length]"
    
    def _process_script_response(
        self, 
        script_content: str, 
        videos: List[VideoData], 
        request: ScriptGenerationRequest
    ) -> ScriptGenerationResponse:
        """
        Process and validate the generated script response.
        
        Args:
            script_content: Raw script content from OpenAI
            videos: Source videos used
            request: Original request parameters
            
        Returns:
            Validated ScriptGenerationResponse
            
        Raises:
            ValueError: If script validation fails
        """
        # Clean up the script content
        script_content = script_content.strip()
        
        # Calculate word count
        word_count = len(script_content.split())
        
        # Estimate duration (average speaking rate: 150-160 words per minute)
        estimated_duration = word_count / 155.0  # Use 155 WPM as average
        
        # Validate word count is reasonable
        if word_count < 100:
            raise ValueError(f"Generated script too short: {word_count} words")
        
        if word_count > request.target_word_count_max * 1.5:
            logger.warning(f"Generated script longer than expected: {word_count} words")
        
        # Create response
        response = ScriptGenerationResponse(
            script=script_content,
            word_count=word_count,
            estimated_duration_minutes=estimated_duration,
            source_videos=[v.video_id for v in videos],
            generation_metadata={
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "target_word_count_min": request.target_word_count_min,
                "target_word_count_max": request.target_word_count_max,
                "language": request.language,
                "style": request.style,
                "source_video_count": len(videos),
                "total_source_views": sum(v.view_count for v in videos),
                "avg_source_quality": sum(v.quality_score or 0 for v in videos) / len(videos)
            }
        )
        
        return response
    
    def validate_api_connection(self) -> bool:
        """
        Validate OpenAI API connection and authentication.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Make a minimal API call to test connection
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            return bool(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI API validation failed: {str(e)}")
            return False


def generate_podcast_script(
    videos: List[VideoData],
    target_word_count_min: int = 750,
    target_word_count_max: int = 1500,
    language: str = "en",
    style: str = "conversational"
) -> ScriptGenerationResponse:
    """
    Convenience function to generate a podcast script from videos.
    
    Args:
        videos: List of videos to use for script generation
        target_word_count_min: Minimum word count for script
        target_word_count_max: Maximum word count for script
        language: Language code for script
        style: Script style
        
    Returns:
        ScriptGenerationResponse with generated script
        
    Raises:
        ValueError: If input validation fails
        RuntimeError: If script generation fails
    """
    generator = OpenAIScriptGenerator()
    
    request = ScriptGenerationRequest(
        videos=videos,
        target_word_count_min=target_word_count_min,
        target_word_count_max=target_word_count_max,
        language=language,
        style=style
    )
    
    return generator.generate_script(request)


def update_curator_state_with_script(
    state: CuratorState,
    script_response: ScriptGenerationResponse
) -> CuratorState:
    """
    Update CuratorState with generated script and metadata.
    
    Args:
        state: Current curator state
        script_response: Generated script response
        
    Returns:
        Updated curator state
    """
    state.podcast_script = script_response.script
    
    # Update generation metadata
    state.update_generation_metadata(
        script_word_count=script_response.word_count,
        estimated_duration_minutes=script_response.estimated_duration_minutes,
        source_video_ids=script_response.source_videos,
        generation_model=script_response.generation_metadata.get("model"),
        generation_temperature=script_response.generation_metadata.get("temperature"),
        source_video_count=script_response.generation_metadata.get("source_video_count"),
        total_source_views=script_response.generation_metadata.get("total_source_views"),
        avg_source_quality=script_response.generation_metadata.get("avg_source_quality")
    )
    
    return state