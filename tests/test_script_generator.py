"""
Tests for OpenAI script generation functionality.

This module tests the OpenAI integration including API client functionality,
prompt generation, error handling, retry logic, and response validation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from nanook_curator.script_generator import (
    OpenAIScriptGenerator,
    ScriptGenerationRequest,
    ScriptGenerationResponse,
    generate_podcast_script,
    update_curator_state_with_script
)
from nanook_curator.models import VideoData, CuratorState


@pytest.fixture
def sample_videos():
    """Create sample video data for testing."""
    return [
        VideoData(
            video_id="dQw4w9WgXcQ",  # Valid 11-character YouTube ID
            title="AI Tools Revolution",
            channel="Tech Channel",
            view_count=10000,
            like_count=800,
            comment_count=50,
            upload_date="2024-01-15T10:00:00Z",
            transcript="This video discusses the latest AI tools and their impact on productivity. "
                      "We explore GPT-4, Claude, and other language models that are changing how we work. "
                      "The key insight is that AI tools are becoming more accessible and powerful.",
            quality_score=85.0,
            key_topics=["AI tools", "productivity", "GPT-4", "language models"]
        ),
        VideoData(
            video_id="jNQXAC9IVRw",  # Valid 11-character YouTube ID
            title="Machine Learning Trends 2024",
            channel="ML Insights",
            view_count=15000,
            like_count=1200,
            comment_count=75,
            upload_date="2024-01-16T14:30:00Z",
            transcript="Machine learning is evolving rapidly in 2024. We see advances in multimodal AI, "
                      "better training efficiency, and new architectures. The trend toward smaller, "
                      "more efficient models is particularly interesting for deployment.",
            quality_score=92.0,
            key_topics=["machine learning", "multimodal AI", "efficiency", "deployment"]
        )
    ]


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    mock_message = ChatCompletionMessage(
        role="assistant",
        content="""Welcome to today's AI update! We're diving into the latest developments that are reshaping how we work with artificial intelligence.

First up, let's talk about the AI tools revolution. According to Tech Channel, we're seeing unprecedented accessibility in powerful AI systems. GPT-4 and Claude are leading the charge, making sophisticated language processing available to everyday users. The key insight here is that these tools aren't just getting more powerful - they're becoming more intuitive and integrated into our daily workflows.

But that's not the whole story. ML Insights brings us fascinating trends from 2024 that show where the field is heading. We're witnessing a shift toward multimodal AI systems that can process text, images, and audio simultaneously. This isn't just about adding features - it's about creating AI that understands context the way humans do.

Perhaps most importantly, there's a growing emphasis on efficiency. While everyone was focused on making models bigger and more capable, researchers have been quietly working on making them smaller and more deployable. This trend toward efficient models means AI capabilities can run on your phone, your laptop, or embedded in everyday devices.

The takeaway? We're moving from AI as a specialized tool to AI as an integrated part of how we work and create. Whether you're using these tools for writing, analysis, or creative projects, the barrier to entry keeps getting lower while the capabilities keep expanding.

That's your AI update for today. The revolution isn't coming - it's here, and it's more accessible than ever."""
    )
    
    mock_choice = Choice(
        index=0,
        message=mock_message,
        finish_reason="stop"
    )
    
    mock_response = Mock(spec=ChatCompletion)
    mock_response.choices = [mock_choice]
    
    return mock_response


@pytest.fixture
def script_generator():
    """Create a script generator instance for testing."""
    with patch('nanook_curator.script_generator.get_config') as mock_config:
        mock_config.return_value.openai_api_key = "sk-test-key-123"
        generator = OpenAIScriptGenerator()
        return generator


class TestScriptGenerationRequest:
    """Test ScriptGenerationRequest model validation."""
    
    def test_valid_request(self, sample_videos):
        """Test creating a valid script generation request."""
        request = ScriptGenerationRequest(
            videos=sample_videos,
            target_word_count_min=750,
            target_word_count_max=1500,
            language="en",
            style="conversational"
        )
        
        assert request.videos == sample_videos
        assert request.target_word_count_min == 750
        assert request.target_word_count_max == 1500
        assert request.language == "en"
        assert request.style == "conversational"
    
    def test_default_values(self, sample_videos):
        """Test default values in script generation request."""
        request = ScriptGenerationRequest(videos=sample_videos)
        
        assert request.target_word_count_min == 750
        assert request.target_word_count_max == 1500
        assert request.language == "en"
        assert request.style == "conversational"
    
    def test_validation_errors(self):
        """Test validation errors for invalid requests."""
        with pytest.raises(ValueError):
            ScriptGenerationRequest(
                videos=[],  # Empty videos list should be handled by generator
                target_word_count_min=-100  # Invalid word count
            )


class TestScriptGenerationResponse:
    """Test ScriptGenerationResponse model validation."""
    
    def test_valid_response(self):
        """Test creating a valid script generation response."""
        response = ScriptGenerationResponse(
            script="Test script content",
            word_count=250,
            estimated_duration_minutes=1.6,
            source_videos=["video1", "video2"],
            generation_metadata={"model": "gpt-4o-mini"}
        )
        
        assert response.script == "Test script content"
        assert response.word_count == 250
        assert response.estimated_duration_minutes == 1.6
        assert response.source_videos == ["video1", "video2"]
        assert response.generation_metadata["model"] == "gpt-4o-mini"


class TestOpenAIScriptGenerator:
    """Test OpenAI script generator functionality."""
    
    def test_initialization(self):
        """Test script generator initialization."""
        with patch('nanook_curator.script_generator.get_config') as mock_config:
            mock_config.return_value.openai_api_key = "sk-test-key-123"
            
            generator = OpenAIScriptGenerator()
            
            assert generator.api_key == "sk-test-key-123"
            assert generator.model == "gpt-4o-mini"
            assert generator.max_retries == 3
            assert generator.temperature == 0.7
    
    def test_initialization_with_custom_key(self):
        """Test script generator initialization with custom API key."""
        with patch('nanook_curator.script_generator.get_config') as mock_config:
            mock_config.return_value.openai_api_key = "sk-config-key"
            
            generator = OpenAIScriptGenerator(api_key="sk-custom-key")
            
            assert generator.api_key == "sk-custom-key"
    
    @patch('nanook_curator.script_generator.OpenAI')
    def test_generate_script_success(self, mock_openai_class, sample_videos, mock_openai_response):
        """Test successful script generation."""
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response
        
        with patch('nanook_curator.script_generator.get_config') as mock_config:
            mock_config.return_value.openai_api_key = "sk-test-key"
            
            generator = OpenAIScriptGenerator()
            request = ScriptGenerationRequest(videos=sample_videos)
            
            response = generator.generate_script(request)
            
            assert isinstance(response, ScriptGenerationResponse)
            assert response.script.startswith("Welcome to today's AI update!")
            assert response.word_count > 0
            assert response.estimated_duration_minutes > 0
            assert len(response.source_videos) == 2
            assert "gpt-4o-mini" in response.generation_metadata.get("model", "")
    
    def test_generate_script_no_videos(self, script_generator):
        """Test script generation with no videos."""
        request = ScriptGenerationRequest(videos=[])
        
        with pytest.raises(ValueError, match="At least one video is required"):
            script_generator.generate_script(request)
    
    def test_generate_script_no_transcripts(self, script_generator):
        """Test script generation with videos that have no transcripts."""
        videos_no_transcript = [
            VideoData(
                video_id="dQw4w9WgXcQ",  # Valid 11-character YouTube ID
                title="Test Video",
                channel="Test Channel",
                view_count=1000,
                like_count=100,
                comment_count=10,
                upload_date="2024-01-15T10:00:00Z",
                transcript=None  # No transcript
            )
        ]
        
        request = ScriptGenerationRequest(videos=videos_no_transcript)
        
        with pytest.raises(ValueError, match="At least one video must have a transcript"):
            script_generator.generate_script(request)
    
    @patch('nanook_curator.script_generator.OpenAI')
    def test_generate_script_api_failure_with_retry(self, mock_openai_class, sample_videos, mock_openai_response):
        """Test script generation with API failure and retry logic."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # First two calls fail, third succeeds
        mock_client.chat.completions.create.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            mock_openai_response
        ]
        
        with patch('nanook_curator.script_generator.get_config') as mock_config:
            mock_config.return_value.openai_api_key = "sk-test-key"
            
            with patch('time.sleep'):  # Speed up test by mocking sleep
                generator = OpenAIScriptGenerator()
                request = ScriptGenerationRequest(videos=sample_videos)
                
                response = generator.generate_script(request)
                
                assert isinstance(response, ScriptGenerationResponse)
                assert mock_client.chat.completions.create.call_count == 3
    
    @patch('nanook_curator.script_generator.OpenAI')
    def test_generate_script_all_retries_fail(self, mock_openai_class, sample_videos):
        """Test script generation when all retry attempts fail."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Persistent API Error")
        
        with patch('nanook_curator.script_generator.get_config') as mock_config:
            mock_config.return_value.openai_api_key = "sk-test-key"
            
            with patch('time.sleep'):  # Speed up test
                generator = OpenAIScriptGenerator()
                request = ScriptGenerationRequest(videos=sample_videos)
                
                with pytest.raises(RuntimeError, match="Script generation failed after 3 attempts"):
                    generator.generate_script(request)
                
                assert mock_client.chat.completions.create.call_count == 3
    
    def test_truncate_transcript(self, script_generator):
        """Test transcript truncation functionality."""
        long_transcript = " ".join(["word"] * 1000)  # 1000 words
        
        truncated = script_generator._truncate_transcript(long_transcript, max_words=500)
        
        # Should be 500 words + 4 words in "[transcript truncated for length]"
        assert len(truncated.split()) <= 504
        assert "truncated for length" in truncated
    
    def test_truncate_transcript_short(self, script_generator):
        """Test transcript truncation with short transcript."""
        short_transcript = "This is a short transcript"
        
        truncated = script_generator._truncate_transcript(short_transcript, max_words=500)
        
        assert truncated == short_transcript
        assert "truncated" not in truncated
    
    def test_get_system_prompt(self, script_generator):
        """Test system prompt generation."""
        request = ScriptGenerationRequest(
            videos=[],  # Empty for this test
            target_word_count_min=750,
            target_word_count_max=1500,
            language="en",
            style="conversational"
        )
        
        prompt = script_generator._get_system_prompt(request)
        
        assert "750-1500 words" in prompt
        assert "conversational" in prompt
        assert "en" in prompt
        assert "podcast script writer" in prompt.lower()
    
    def test_create_script_prompt(self, script_generator, sample_videos):
        """Test script prompt creation with video content."""
        request = ScriptGenerationRequest(videos=sample_videos)
        
        prompt = script_generator._create_script_prompt(request, sample_videos)
        
        assert "AI Tools Revolution" in prompt
        assert "Machine Learning Trends 2024" in prompt
        assert "Tech Channel" in prompt
        assert "ML Insights" in prompt
        assert "TRANSCRIPT CONTENT:" in prompt  # Updated to match new format
        assert "750-1500 word" in prompt
        assert "SYNTHESIS INSTRUCTIONS:" in prompt  # Check for new synthesis instructions
    
    @patch('nanook_curator.script_generator.OpenAI')
    def test_validate_api_connection_success(self, mock_openai_class):
        """Test successful API connection validation."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('nanook_curator.script_generator.get_config') as mock_config:
            mock_config.return_value.openai_api_key = "sk-test-key"
            
            generator = OpenAIScriptGenerator()
            result = generator.validate_api_connection()
            
            assert result is True
    
    @patch('nanook_curator.script_generator.OpenAI')
    def test_validate_api_connection_failure(self, mock_openai_class):
        """Test API connection validation failure."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        with patch('nanook_curator.script_generator.get_config') as mock_config:
            mock_config.return_value.openai_api_key = "sk-test-key"
            
            generator = OpenAIScriptGenerator()
            result = generator.validate_api_connection()
            
            assert result is False


class TestConvenienceFunctions:
    """Test convenience functions for script generation."""
    
    @patch('nanook_curator.script_generator.OpenAIScriptGenerator')
    def test_generate_podcast_script(self, mock_generator_class, sample_videos):
        """Test the convenience function for generating podcast scripts."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        mock_response = ScriptGenerationResponse(
            script="Test script",
            word_count=250,
            estimated_duration_minutes=1.6,
            source_videos=["video1", "video2"],
            generation_metadata={}
        )
        mock_generator.generate_script.return_value = mock_response
        
        result = generate_podcast_script(
            videos=sample_videos,
            target_word_count_min=800,
            target_word_count_max=1200
        )
        
        assert result == mock_response
        mock_generator.generate_script.assert_called_once()
        
        # Check the request parameters
        call_args = mock_generator.generate_script.call_args[0][0]
        assert call_args.target_word_count_min == 800
        assert call_args.target_word_count_max == 1200
        assert call_args.videos == sample_videos
    
    def test_update_curator_state_with_script(self):
        """Test updating curator state with generated script."""
        state = CuratorState(search_keywords=["AI", "test"])
        
        script_response = ScriptGenerationResponse(
            script="Generated podcast script content",
            word_count=300,
            estimated_duration_minutes=2.0,
            source_videos=["video1", "video2"],
            generation_metadata={
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "source_video_count": 2,
                "total_source_views": 25000,
                "avg_source_quality": 88.5
            }
        )
        
        updated_state = update_curator_state_with_script(state, script_response)
        
        assert updated_state.podcast_script == "Generated podcast script content"
        assert updated_state.generation_metadata["script_word_count"] == 300
        assert updated_state.generation_metadata["estimated_duration_minutes"] == 2.0
        assert updated_state.generation_metadata["source_video_ids"] == ["video1", "video2"]
        assert updated_state.generation_metadata["generation_model"] == "gpt-4o-mini"
        assert updated_state.generation_metadata["source_video_count"] == 2
        assert updated_state.generation_metadata["total_source_views"] == 25000
        assert updated_state.generation_metadata["avg_source_quality"] == 88.5
        assert "last_updated" in updated_state.generation_metadata


class TestScriptSynthesis:
    """Test script synthesis and structuring functionality."""
    
    def test_select_top_ranked_videos(self, script_generator):
        """Test selection of top 3-5 ranked videos."""
        videos = [
            VideoData(
                video_id="dQw4w9WgXc1", title="Video 1", channel="Channel 1",
                view_count=1000, like_count=100, comment_count=10,
                upload_date="2024-01-15T10:00:00Z", transcript="Test transcript",
                quality_score=95.0
            ),
            VideoData(
                video_id="dQw4w9WgXc2", title="Video 2", channel="Channel 2",
                view_count=2000, like_count=200, comment_count=20,
                upload_date="2024-01-16T10:00:00Z", transcript="Test transcript",
                quality_score=85.0
            ),
            VideoData(
                video_id="dQw4w9WgXc3", title="Video 3", channel="Channel 3",
                view_count=3000, like_count=300, comment_count=30,
                upload_date="2024-01-17T10:00:00Z", transcript="Test transcript",
                quality_score=75.0
            ),
            VideoData(
                video_id="dQw4w9WgXc4", title="Video 4", channel="Channel 4",
                view_count=4000, like_count=400, comment_count=40,
                upload_date="2024-01-18T10:00:00Z", transcript="Test transcript",
                quality_score=65.0
            )
        ]
        
        top_videos = script_generator._select_top_ranked_videos(videos)
        
        assert len(top_videos) == 4  # Should select top 4 videos
        assert top_videos[0].quality_score == 95.0  # Highest quality first
        assert top_videos[1].quality_score == 85.0
        assert top_videos[2].quality_score == 75.0
        assert top_videos[3].quality_score == 65.0
    
    def test_select_top_ranked_videos_no_quality_scores(self, script_generator):
        """Test video selection when no quality scores are available."""
        videos = [
            VideoData(
                video_id="dQw4w9WgXc1", title="Video 1", channel="Channel 1",
                view_count=1000, like_count=100, comment_count=10,
                upload_date="2024-01-15T10:00:00Z", transcript="Test transcript"
            ),
            VideoData(
                video_id="dQw4w9WgXc2", title="Video 2", channel="Channel 2",
                view_count=2000, like_count=200, comment_count=20,
                upload_date="2024-01-16T10:00:00Z", transcript="Test transcript"
            )
        ]
        
        with patch('nanook_curator.script_generator.logger') as mock_logger:
            top_videos = script_generator._select_top_ranked_videos(videos)
            
            assert len(top_videos) == 2
            mock_logger.warning.assert_called_once()
            assert "No videos have quality scores" in mock_logger.warning.call_args[0][0]
    
    def test_validate_script_synthesis(self, script_generator, sample_videos):
        """Test script synthesis validation."""
        script_with_good_structure = """
        Welcome to today's AI update! We're diving into the latest developments.
        
        According to Tech Channel, AI tools are becoming more accessible. 
        Meanwhile, ML Insights points out that efficiency is key.
        Furthermore, these developments connect to broader trends.
        
        In summary, these are the key takeaways from our analysis.
        """
        
        synthesis_quality = script_generator._validate_script_synthesis(
            script_with_good_structure, sample_videos
        )
        
        assert synthesis_quality["has_introduction"] is True
        assert synthesis_quality["has_conclusion"] is True
        assert synthesis_quality["source_attribution_count"] >= 2
        assert synthesis_quality["transition_indicators"] >= 3
        assert synthesis_quality["structure_score"] > 0.8


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_empty_script_response(self, script_generator, sample_videos):
        """Test handling of empty script response from OpenAI."""
        with patch.object(script_generator, '_generate_with_retry') as mock_generate:
            mock_generate.return_value = ""  # Empty response
            
            request = ScriptGenerationRequest(videos=sample_videos)
            
            with pytest.raises(ValueError, match="Generated script too short"):
                script_generator.generate_script(request)
    
    def test_very_short_script_response(self, script_generator, sample_videos):
        """Test handling of very short script response."""
        with patch.object(script_generator, '_generate_with_retry') as mock_generate:
            mock_generate.return_value = "Short script"  # Very short response
            
            request = ScriptGenerationRequest(videos=sample_videos)
            
            with pytest.raises(ValueError, match="Generated script too short"):
                script_generator.generate_script(request)
    
    def test_very_long_script_response(self, script_generator, sample_videos):
        """Test handling of very long script response."""
        with patch.object(script_generator, '_generate_with_retry') as mock_generate:
            # Create a very long script (3000 words)
            long_script = " ".join(["word"] * 3000)
            mock_generate.return_value = long_script
            
            request = ScriptGenerationRequest(
                videos=sample_videos,
                target_word_count_max=1500
            )
            
            # Should not raise error but should log warning
            with patch('nanook_curator.script_generator.logger') as mock_logger:
                response = script_generator.generate_script(request)
                
                assert response.word_count == 3000
                # Check that warning was called (may be called multiple times due to video selection)
                warning_calls = [call for call in mock_logger.warning.call_args_list 
                               if "longer than expected" in str(call)]
                assert len(warning_calls) >= 1
                assert "longer than expected" in str(warning_calls[0])


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response with proper structure."""
    mock_message = Mock()
    mock_message.content = """Welcome to today's AI update! We're diving into the latest developments that are reshaping how we work with artificial intelligence.

First up, let's talk about the AI tools revolution. According to Tech Channel, we're seeing unprecedented accessibility in powerful AI systems. GPT-4 and Claude are leading the charge, making sophisticated language processing available to everyday users. The key insight here is that these tools aren't just getting more powerful - they're becoming more intuitive and integrated into our daily workflows.

But that's not the whole story. ML Insights brings us fascinating trends from 2024 that show where the field is heading. We're witnessing a shift toward multimodal AI systems that can process text, images, and audio simultaneously. This isn't just about adding features - it's about creating AI that understands context the way humans do.

Perhaps most importantly, there's a growing emphasis on efficiency. While everyone was focused on making models bigger and more capable, researchers have been quietly working on making them smaller and more deployable. This trend toward efficient models means AI capabilities can run on your phone, your laptop, or embedded in everyday devices.

The takeaway? We're moving from AI as a specialized tool to AI as an integrated part of how we work and create. Whether you're using these tools for writing, analysis, or creative projects, the barrier to entry keeps getting lower while the capabilities keep expanding.

That's your AI update for today. The revolution isn't coming - it's here, and it's more accessible than ever."""
    
    mock_choice = Mock()
    mock_choice.message = mock_message
    
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    
    return mock_response