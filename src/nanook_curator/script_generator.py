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
        
        # Select top 3-5 ranked videos as per requirement 4.1
        top_videos = self._select_top_ranked_videos(valid_videos)
        
        logger.info(f"Generating script from {len(top_videos)} top-ranked videos "
                   f"(selected from {len(valid_videos)} valid videos)")
        
        # Generate the script with retry logic
        script_content = self._generate_with_retry(request, top_videos)
        
        # Validate and process the response
        response = self._process_script_response(script_content, top_videos, request)
        
        logger.info(f"Generated script: {response.word_count} words, "
                   f"{response.estimated_duration_minutes:.1f} minutes, "
                   f"synthesis score: {response.generation_metadata.get('synthesis_quality', {}).get('structure_score', 0):.2f}")
        
        return response
    
    def _select_top_ranked_videos(self, videos: List[VideoData]) -> List[VideoData]:
        """
        Select the top 3-5 highest-ranked videos for script generation.
        
        Args:
            videos: List of valid videos with transcripts
            
        Returns:
            List of top 3-5 videos sorted by quality score
        """
        # Sort videos by quality score (highest first)
        sorted_videos = sorted(
            [v for v in videos if v.quality_score is not None],
            key=lambda x: x.quality_score,
            reverse=True
        )
        
        # If no quality scores, use all videos up to 5
        if not sorted_videos:
            logger.warning("No videos have quality scores, using first 5 videos")
            return videos[:5]
        
        # Select top 3-5 videos based on quality distribution
        if len(sorted_videos) >= 5:
            # Use top 5 if we have enough high-quality videos
            top_videos = sorted_videos[:5]
        elif len(sorted_videos) >= 3:
            # Use top 3-4 if we have at least 3 quality videos
            top_videos = sorted_videos[:min(4, len(sorted_videos))]
        else:
            # Use all available videos if less than 3
            logger.warning(f"Only {len(sorted_videos)} videos with quality scores available")
            top_videos = sorted_videos
        
        logger.info(f"Selected {len(top_videos)} videos with quality scores: "
                   f"{[f'{v.quality_score:.1f}' for v in top_videos]}")
        
        return top_videos
    
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
Your task is to create engaging, informative podcast scripts that synthesize information from multiple YouTube videos into a cohesive narrative.

REQUIREMENTS:
- Target length: {request.target_word_count_min}-{request.target_word_count_max} words (5-10 minutes speaking time)
- Style: {request.style} and engaging for audio consumption
- Language: {request.language}
- Audience: Tech-savvy professionals interested in AI developments

SCRIPT STRUCTURE (MANDATORY):
1. HOOK/INTRODUCTION (10-15% of content):
   - Start with an engaging hook that captures attention
   - Briefly preview the main topics to be covered
   - Set the context for why these developments matter

2. MAIN CONTENT (70-80% of content):
   - Synthesize insights from the top 3-5 ranked videos
   - Create smooth transitions between different video topics
   - Maintain narrative flow while covering distinct insights
   - Include specific technical details and practical implications
   - Cite original video sources naturally in the narrative

3. KEY TAKEAWAYS & CONCLUSION (10-15% of content):
   - Summarize the most important insights
   - Connect the dots between different topics
   - End with actionable takeaways or future implications

NARRATIVE SYNTHESIS GUIDELINES:
- Create a cohesive story arc that connects insights from multiple sources
- Use transitional phrases to move smoothly between video topics
- Avoid simply summarizing each video sequentially
- Instead, weave insights together thematically
- Include specific quotes or key points from each source video
- Maintain conversational tone suitable for podcast listening
- Cite sources naturally (e.g., "As highlighted by [Channel Name]...", "According to the analysis from [Channel]...")

QUALITY REQUIREMENTS:
- Focus on the most valuable and actionable insights from each video
- Avoid repetitive information across sources
- Make complex topics accessible without oversimplifying
- Ensure technical accuracy while maintaining engagement
- Create content that justifies the 5-10 minute listening time

OUTPUT FORMAT:
Provide only the script content without additional formatting, headers, or metadata. The script should be ready for immediate podcast recording."""
    
    def _create_script_prompt(self, request: ScriptGenerationRequest, videos: List[VideoData]) -> str:
        """
        Create the user prompt with video content for script synthesis.
        
        Args:
            request: Script generation request
            videos: Videos to include in the script (top 3-5 ranked)
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"Create a {request.target_word_count_min}-{request.target_word_count_max} word podcast script by synthesizing insights from the following {len(videos)} top-ranked videos:",
            ""
        ]
        
        # Add video content with enhanced metadata for better synthesis
        for i, video in enumerate(videos, 1):
            prompt_parts.extend([
                f"=== SOURCE VIDEO {i} ===",
                f"Title: {video.title}",
                f"Channel: {video.channel}",
                f"Views: {video.view_count:,}",
                f"Quality Score: {video.quality_score:.1f}/100" if video.quality_score else "Quality Score: N/A",
                f"Key Topics: {', '.join(video.key_topics)}" if video.key_topics else "Key Topics: N/A",
                "",
                "TRANSCRIPT CONTENT:",
                self._truncate_transcript(video.transcript, max_words=800),
                "",
                "---",
                ""
            ])
        
        prompt_parts.extend([
            "SYNTHESIS INSTRUCTIONS:",
            "",
            "1. NARRATIVE STRUCTURE:",
            "   - Create a compelling introduction that hooks the listener",
            "   - Develop main content that weaves insights from all videos together",
            "   - Conclude with key takeaways and actionable insights",
            "",
            "2. CONTENT SYNTHESIS:",
            "   - Don't summarize each video separately",
            "   - Instead, identify common themes and complementary insights",
            "   - Create a unified narrative that connects different perspectives",
            "   - Highlight unique insights from each source",
            "",
            "3. SOURCE ATTRIBUTION:",
            "   - Naturally cite each video source within the narrative flow",
            "   - Use varied attribution phrases (e.g., 'According to [Channel]...', 'As highlighted in [Title]...', '[Channel] points out that...')",
            "   - Ensure all source videos are referenced in the final script",
            "",
            "4. TRANSITIONS & FLOW:",
            "   - Use smooth transitions between topics from different videos",
            "   - Connect ideas logically rather than jumping between sources",
            "   - Maintain conversational flow suitable for audio consumption",
            "",
            "5. TECHNICAL REQUIREMENTS:",
            f"   - Target exactly {request.target_word_count_min}-{request.target_word_count_max} words",
            "   - Maintain engaging, conversational tone throughout",
            "   - Include specific technical details and practical implications",
            "   - Focus on the most valuable insights that justify the listening time",
            "",
            "Generate the cohesive podcast script now, ensuring it meets all synthesis and structuring requirements:"
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
        
        # Calculate initial word count and duration
        initial_word_count = len(script_content.split())
        initial_duration = initial_word_count / 155.0  # Use 155 WPM as average
        
        # Apply automatic length management if script exceeds 10-minute target
        max_duration_minutes = 10.0
        max_word_count = int(max_duration_minutes * 155)  # ~1550 words for 10 minutes
        
        trimming_applied = False
        if initial_duration > max_duration_minutes:
            logger.info(f"Script exceeds 10-minute target ({initial_duration:.1f} minutes, {initial_word_count} words). Applying automatic trimming.")
            script_content = self._apply_automatic_length_management(
                script_content, 
                max_word_count, 
                request.target_word_count_max
            )
            trimming_applied = True
        
        # Calculate final word count and duration
        final_word_count = len(script_content.split())
        final_duration = final_word_count / 155.0
        
        # Validate word count is reasonable
        if final_word_count < 100:
            raise ValueError(f"Generated script too short: {final_word_count} words")
        
        if final_word_count > request.target_word_count_max * 1.5 and not trimming_applied:
            logger.warning(f"Generated script longer than expected: {final_word_count} words")
        
        # Validate script synthesis quality
        synthesis_quality = self._validate_script_synthesis(script_content, videos)
        
        # Create response with length management metadata
        generation_metadata = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "target_word_count_min": request.target_word_count_min,
            "target_word_count_max": request.target_word_count_max,
            "language": request.language,
            "style": request.style,
            "source_video_count": len(videos),
            "total_source_views": sum(v.view_count for v in videos),
            "avg_source_quality": sum(v.quality_score or 0 for v in videos) / len(videos),
            "synthesis_quality": synthesis_quality,
            "length_management": {
                "initial_word_count": initial_word_count,
                "initial_duration_minutes": initial_duration,
                "final_word_count": final_word_count,
                "final_duration_minutes": final_duration,
                "trimming_applied": trimming_applied,
                "words_trimmed": initial_word_count - final_word_count if trimming_applied else 0
            }
        }
        
        response = ScriptGenerationResponse(
            script=script_content,
            word_count=final_word_count,
            estimated_duration_minutes=final_duration,
            source_videos=[v.video_id for v in videos],
            generation_metadata=generation_metadata
        )
        
        if trimming_applied:
            logger.info(f"Length management completed: {initial_word_count} → {final_word_count} words "
                       f"({initial_duration:.1f} → {final_duration:.1f} minutes)")
        
        return response
    
    def _validate_script_synthesis(self, script_content: str, videos: List[VideoData]) -> Dict[str, Any]:
        """
        Validate that the script meets synthesis and structuring requirements.
        
        Args:
            script_content: Generated script content
            videos: Source videos used
            
        Returns:
            Dictionary with synthesis quality metrics
        """
        synthesis_metrics = {
            "has_introduction": False,
            "has_conclusion": False,
            "source_attribution_count": 0,
            "sources_referenced": [],
            "transition_indicators": 0,
            "structure_score": 0.0
        }
        
        script_lower = script_content.lower()
        
        # Check for introduction indicators
        intro_indicators = [
            "welcome", "today", "we're", "let's", "first", "starting", 
            "diving into", "exploring", "looking at"
        ]
        if any(indicator in script_lower[:200] for indicator in intro_indicators):
            synthesis_metrics["has_introduction"] = True
        
        # Check for conclusion indicators
        conclusion_indicators = [
            "takeaway", "conclusion", "summary", "wrap up", "finally", 
            "in summary", "to conclude", "key points", "remember"
        ]
        if any(indicator in script_lower[-300:] for indicator in conclusion_indicators):
            synthesis_metrics["has_conclusion"] = True
        
        # Count source attributions
        for video in videos:
            channel_mentions = script_lower.count(video.channel.lower())
            if channel_mentions > 0:
                synthesis_metrics["source_attribution_count"] += channel_mentions
                synthesis_metrics["sources_referenced"].append(video.channel)
        
        # Count transition indicators
        transition_phrases = [
            "according to", "as highlighted", "points out", "meanwhile", 
            "furthermore", "additionally", "on the other hand", "however",
            "building on", "this connects to", "similarly", "in contrast"
        ]
        for phrase in transition_phrases:
            synthesis_metrics["transition_indicators"] += script_lower.count(phrase)
        
        # Calculate overall structure score
        structure_score = 0.0
        if synthesis_metrics["has_introduction"]:
            structure_score += 0.3
        if synthesis_metrics["has_conclusion"]:
            structure_score += 0.3
        if synthesis_metrics["source_attribution_count"] >= len(videos):
            structure_score += 0.2
        if synthesis_metrics["transition_indicators"] >= 3:
            structure_score += 0.2
        
        synthesis_metrics["structure_score"] = structure_score
        
        # Log synthesis quality
        logger.info(f"Script synthesis quality - Structure score: {structure_score:.2f}, "
                   f"Sources referenced: {len(synthesis_metrics['sources_referenced'])}/{len(videos)}, "
                   f"Transitions: {synthesis_metrics['transition_indicators']}")
        
        return synthesis_metrics
    
    def _apply_automatic_length_management(
        self, 
        script_content: str, 
        max_word_count: int, 
        target_max: int
    ) -> str:
        """
        Apply automatic length management to trim script while maintaining coherence.
        
        Uses importance-based section prioritization to identify and trim less critical
        content while preserving the script's structure and key insights.
        
        Args:
            script_content: Original script content
            max_word_count: Maximum allowed word count (10-minute target)
            target_max: Target maximum from request (for fallback)
            
        Returns:
            Trimmed script content maintaining coherence
        """
        current_word_count = len(script_content.split())
        
        # If already within target, no trimming needed
        if current_word_count <= max_word_count:
            return script_content
        
        logger.info(f"Applying length management: {current_word_count} → {max_word_count} words target")
        
        # Split script into sections for analysis
        sections = self._identify_script_sections(script_content)
        
        # Calculate importance scores for each section
        section_importance = self._calculate_section_importance(sections)
        
        # Apply progressive trimming strategies
        trimmed_script = self._progressive_trimming(
            sections, 
            section_importance, 
            max_word_count,
            target_max
        )
        
        # Ensure coherence is maintained after trimming
        final_script = self._ensure_script_coherence(trimmed_script)
        
        return final_script
    
    def _identify_script_sections(self, script_content: str) -> List[Dict[str, Any]]:
        """
        Identify and categorize script sections for importance-based trimming.
        
        Args:
            script_content: Full script content
            
        Returns:
            List of section dictionaries with content and metadata
        """
        paragraphs = [p.strip() for p in script_content.split('\n\n') if p.strip()]
        sections = []
        
        for i, paragraph in enumerate(paragraphs):
            section_type = self._classify_section_type(paragraph, i, len(paragraphs))
            word_count = len(paragraph.split())
            
            sections.append({
                "content": paragraph,
                "type": section_type,
                "word_count": word_count,
                "position": i,
                "total_sections": len(paragraphs)
            })
        
        return sections
    
    def _classify_section_type(self, paragraph: str, position: int, total_sections: int) -> str:
        """
        Classify the type of script section for importance scoring.
        
        Args:
            paragraph: Paragraph content
            position: Position in script (0-based)
            total_sections: Total number of sections
            
        Returns:
            Section type classification
        """
        paragraph_lower = paragraph.lower()
        
        # Introduction section (first 20% of script)
        if position < total_sections * 0.2:
            if any(word in paragraph_lower for word in [
                "welcome", "today", "we're", "let's", "diving into", "starting"
            ]):
                return "introduction"
            return "early_content"
        
        # Conclusion section (last 20% of script)
        elif position >= total_sections * 0.8:
            if any(word in paragraph_lower for word in [
                "takeaway", "conclusion", "summary", "wrap up", "finally", 
                "in summary", "to conclude", "key points", "remember"
            ]):
                return "conclusion"
            return "late_content"
        
        # Main content sections
        else:
            # High-value content indicators
            if any(indicator in paragraph_lower for indicator in [
                "according to", "as highlighted", "key insight", "important", 
                "breakthrough", "significant", "critical", "essential"
            ]):
                return "key_insight"
            
            # Transition/connecting content
            elif any(indicator in paragraph_lower for indicator in [
                "meanwhile", "furthermore", "additionally", "however", 
                "on the other hand", "building on", "this connects"
            ]):
                return "transition"
            
            # Supporting details
            elif any(indicator in paragraph_lower for indicator in [
                "for example", "specifically", "in particular", "details", 
                "such as", "including"
            ]):
                return "supporting_detail"
            
            return "main_content"
    
    def _calculate_section_importance(self, sections: List[Dict[str, Any]]) -> Dict[int, float]:
        """
        Calculate importance scores for each section based on type and content.
        
        Args:
            sections: List of section dictionaries
            
        Returns:
            Dictionary mapping section index to importance score (0.0-1.0)
        """
        importance_scores = {}
        
        # Base importance by section type
        type_importance = {
            "introduction": 0.9,      # High - sets context
            "conclusion": 0.9,        # High - provides closure
            "key_insight": 0.8,       # High - core content
            "main_content": 0.6,      # Medium - general content
            "transition": 0.4,        # Lower - can be condensed
            "supporting_detail": 0.3, # Lower - can be trimmed
            "early_content": 0.5,     # Medium - context setting
            "late_content": 0.5       # Medium - wrapping up
        }
        
        for i, section in enumerate(sections):
            base_score = type_importance.get(section["type"], 0.5)
            
            # Adjust based on content quality indicators
            content_lower = section["content"].lower()
            
            # Boost for source attribution
            if any(phrase in content_lower for phrase in [
                "according to", "as highlighted", "points out", "research shows"
            ]):
                base_score += 0.1
            
            # Boost for technical specificity
            if any(term in content_lower for term in [
                "algorithm", "model", "api", "framework", "architecture", 
                "implementation", "performance", "efficiency"
            ]):
                base_score += 0.1
            
            # Boost for actionable insights
            if any(phrase in content_lower for phrase in [
                "you can", "this means", "practical", "actionable", 
                "how to", "enables", "allows"
            ]):
                base_score += 0.1
            
            # Penalty for redundant phrases
            if any(phrase in content_lower for phrase in [
                "as mentioned", "like we said", "again", "once more", "repeat"
            ]):
                base_score -= 0.1
            
            # Ensure score stays within bounds
            importance_scores[i] = max(0.0, min(1.0, base_score))
        
        return importance_scores
    
    def _progressive_trimming(
        self, 
        sections: List[Dict[str, Any]], 
        importance_scores: Dict[int, float], 
        max_word_count: int,
        target_max: int
    ) -> str:
        """
        Apply progressive trimming strategies to reach target word count.
        
        Args:
            sections: Script sections with metadata
            importance_scores: Importance score for each section
            max_word_count: Hard maximum (10-minute target)
            target_max: Soft target maximum
            
        Returns:
            Trimmed script content
        """
        current_word_count = sum(section["word_count"] for section in sections)
        
        # Strategy 1: Remove lowest importance sections entirely
        if current_word_count > max_word_count:
            sections = self._remove_low_importance_sections(
                sections, importance_scores, max_word_count
            )
            current_word_count = sum(section["word_count"] for section in sections)
        
        # Strategy 2: Trim within sections by removing less important sentences
        if current_word_count > max_word_count:
            sections = self._trim_within_sections(
                sections, importance_scores, max_word_count
            )
            current_word_count = sum(section["word_count"] for section in sections)
        
        # Strategy 3: Aggressive trimming if still over limit
        if current_word_count > max_word_count:
            sections = self._aggressive_trimming(
                sections, importance_scores, max_word_count
            )
        
        # Reconstruct script
        trimmed_content = "\n\n".join(section["content"] for section in sections if section["content"].strip())
        
        return trimmed_content
    
    def _remove_low_importance_sections(
        self, 
        sections: List[Dict[str, Any]], 
        importance_scores: Dict[int, float], 
        target_word_count: int
    ) -> List[Dict[str, Any]]:
        """
        Remove entire sections with lowest importance scores.
        
        Args:
            sections: Script sections
            importance_scores: Importance scores
            target_word_count: Target word count
            
        Returns:
            Filtered sections list
        """
        # Sort sections by importance (lowest first)
        sections_by_importance = sorted(
            enumerate(sections), 
            key=lambda x: importance_scores.get(x[0], 0.5)
        )
        
        current_word_count = sum(section["word_count"] for section in sections)
        filtered_sections = sections.copy()
        
        for original_idx, section in sections_by_importance:
            if current_word_count <= target_word_count:
                break
            
            # Don't remove introduction or conclusion sections
            if section["type"] in ["introduction", "conclusion"]:
                continue
            
            # Don't remove if it would break narrative flow
            if self._is_critical_for_flow(section, sections):
                continue
            
            # Remove this section
            filtered_sections = [s for i, s in enumerate(filtered_sections) if i != original_idx]
            current_word_count -= section["word_count"]
            
            logger.debug(f"Removed {section['type']} section ({section['word_count']} words)")
        
        return filtered_sections
    
    def _trim_within_sections(
        self, 
        sections: List[Dict[str, Any]], 
        importance_scores: Dict[int, float], 
        target_word_count: int
    ) -> List[Dict[str, Any]]:
        """
        Trim content within sections by removing less important sentences.
        
        Args:
            sections: Script sections
            importance_scores: Section importance scores
            target_word_count: Target word count
            
        Returns:
            Sections with trimmed content
        """
        current_word_count = sum(section["word_count"] for section in sections)
        words_to_remove = current_word_count - target_word_count
        
        if words_to_remove <= 0:
            return sections
        
        # Focus on sections with lower importance scores for trimming
        trimmable_sections = [
            (i, section) for i, section in enumerate(sections)
            if importance_scores.get(i, 0.5) < 0.7 and section["word_count"] > 50
        ]
        
        for section_idx, section in trimmable_sections:
            if words_to_remove <= 0:
                break
            
            # Trim sentences from this section
            sentences = [s.strip() for s in section["content"].split('.') if s.strip()]
            if len(sentences) <= 2:  # Keep at least 2 sentences
                continue
            
            # Remove less important sentences (typically middle ones)
            sentences_to_keep = max(2, len(sentences) - min(2, len(sentences) // 3))
            
            # Keep first and last sentences, trim from middle
            if len(sentences) > sentences_to_keep:
                kept_sentences = (
                    sentences[:1] + 
                    sentences[-(sentences_to_keep-1):] if sentences_to_keep > 1 else sentences[:1]
                )
                
                new_content = '. '.join(kept_sentences) + '.'
                old_word_count = section["word_count"]
                new_word_count = len(new_content.split())
                
                sections[section_idx]["content"] = new_content
                sections[section_idx]["word_count"] = new_word_count
                
                words_removed = old_word_count - new_word_count
                words_to_remove -= words_removed
                
                logger.debug(f"Trimmed {words_removed} words from {section['type']} section")
        
        return sections
    
    def _aggressive_trimming(
        self, 
        sections: List[Dict[str, Any]], 
        importance_scores: Dict[int, float], 
        target_word_count: int
    ) -> List[Dict[str, Any]]:
        """
        Apply aggressive trimming as last resort while preserving core structure.
        
        Args:
            sections: Script sections
            importance_scores: Section importance scores
            target_word_count: Target word count
            
        Returns:
            Aggressively trimmed sections
        """
        current_word_count = sum(section["word_count"] for section in sections)
        
        if current_word_count <= target_word_count:
            return sections
        
        reduction_ratio = min(0.8, target_word_count / current_word_count)  # Cap reduction ratio
        
        logger.warning(f"Applying aggressive trimming with {reduction_ratio:.2f} reduction ratio")
        
        for i, section in enumerate(sections):
            # Preserve introduction and conclusion with minimal trimming
            if section["type"] in ["introduction", "conclusion"]:
                target_section_words = max(30, int(section["word_count"] * 0.9))
            else:
                target_section_words = max(20, int(section["word_count"] * reduction_ratio))
            
            if section["word_count"] > target_section_words:
                # Aggressive sentence trimming
                sentences = [s.strip() for s in section["content"].split('.') if s.strip()]
                
                if len(sentences) <= 1:
                    # Single sentence - just keep it
                    continue
                
                sentences_to_keep = max(1, min(len(sentences), int(len(sentences) * reduction_ratio)))
                
                # Keep most important sentences (first and last have priority)
                if sentences_to_keep == 1:
                    kept_sentences = [sentences[0]]
                elif sentences_to_keep == 2 and len(sentences) >= 2:
                    kept_sentences = [sentences[0], sentences[-1]]
                else:
                    # Keep first, last, and some middle sentences
                    if len(sentences) >= 3:
                        middle_count = max(0, sentences_to_keep - 2)
                        middle_sentences = sentences[1:-1][:middle_count]
                        kept_sentences = [sentences[0]] + middle_sentences + [sentences[-1]]
                    else:
                        kept_sentences = sentences[:sentences_to_keep]
                
                new_content = '. '.join(kept_sentences) + '.'
                sections[i]["content"] = new_content
                sections[i]["word_count"] = len(new_content.split())
        
        return sections
    
    def _is_critical_for_flow(self, section: Dict[str, Any], all_sections: List[Dict[str, Any]]) -> bool:
        """
        Determine if a section is critical for maintaining narrative flow.
        
        Args:
            section: Section to evaluate
            all_sections: All script sections for context
            
        Returns:
            True if section is critical for flow
        """
        # Transition sections are often critical for flow
        if section["type"] == "transition":
            return True
        
        # Sections with source attribution are important
        content_lower = section["content"].lower()
        if any(phrase in content_lower for phrase in [
            "according to", "as highlighted", "research shows", "study found"
        ]):
            return True
        
        # First and last content sections are typically important
        content_sections = [s for s in all_sections if s["type"] not in ["introduction", "conclusion"]]
        if section in content_sections[:1] or section in content_sections[-1:]:
            return True
        
        return False
    
    def _ensure_script_coherence(self, script_content: str) -> str:
        """
        Ensure script maintains coherence after trimming by fixing transitions.
        
        Args:
            script_content: Trimmed script content
            
        Returns:
            Script with improved coherence
        """
        paragraphs = [p.strip() for p in script_content.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= 1:
            return script_content
        
        # Fix abrupt transitions between paragraphs
        improved_paragraphs = [paragraphs[0]]  # Keep first paragraph as-is
        
        for i in range(1, len(paragraphs)):
            current_para = paragraphs[i]
            previous_para = improved_paragraphs[-1]
            
            # Add transition if needed
            if self._needs_transition(previous_para, current_para):
                transition = self._generate_transition(previous_para, current_para)
                if transition:
                    # Add transition to beginning of current paragraph
                    current_para = f"{transition} {current_para}"
            
            improved_paragraphs.append(current_para)
        
        return '\n\n'.join(improved_paragraphs)
    
    def _needs_transition(self, previous_para: str, current_para: str) -> bool:
        """
        Determine if a transition is needed between paragraphs.
        
        Args:
            previous_para: Previous paragraph content
            current_para: Current paragraph content
            
        Returns:
            True if transition is needed
        """
        # Check if current paragraph already starts with a transition
        current_lower = current_para.lower()
        existing_transitions = [
            "meanwhile", "furthermore", "additionally", "however", "but",
            "on the other hand", "building on", "this connects", "similarly",
            "in contrast", "next", "another", "also"
        ]
        
        if any(current_lower.startswith(trans) for trans in existing_transitions):
            return False
        
        # Check for topic shifts that might need transitions
        # This is a simplified heuristic - in practice, you might use more sophisticated NLP
        return True  # Conservative approach - add transitions where they might help
    
    def _generate_transition(self, previous_para: str, current_para: str) -> str:
        """
        Generate an appropriate transition phrase between paragraphs.
        
        Args:
            previous_para: Previous paragraph content
            current_para: Current paragraph content
            
        Returns:
            Transition phrase or empty string
        """
        # Simple transition generation based on content patterns
        current_lower = current_para.lower()
        
        # If current paragraph mentions a source, use "Meanwhile" or "Additionally"
        if any(phrase in current_lower for phrase in ["according to", "as highlighted", "research"]):
            return "Meanwhile,"
        
        # If current paragraph seems to build on previous, use "Furthermore"
        if any(word in current_lower for word in ["also", "builds", "extends", "expands"]):
            return "Furthermore,"
        
        # If current paragraph contrasts, use "However"
        if any(word in current_lower for word in ["but", "however", "different", "contrast"]):
            return "However,"
        
        # Default transition for continuing the narrative
        return "Additionally,"
    
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