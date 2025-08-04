"""
LangGraph workflow orchestration for the nanook-curator system.

This module implements the complete LangGraph workflow with nodes for each processing step,
including video discovery, transcript fetching, quality evaluation, ranking, script generation,
and iterative search refinement with parallel processing capabilities.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from functools import wraps
from datetime import datetime

from langgraph.graph import StateGraph, END

from .models import VideoData, CuratorState
from .config import get_config, Configuration
from .youtube_client import YouTubeClient, SearchFilters
from .transcript_processor import TranscriptProcessor
from .engagement_analyzer import EngagementAnalyzer
from .content_quality_scorer import ContentQualityScorer
from .video_ranking_system import VideoRankingSystem, RankingStrategy
from .script_generator import OpenAIScriptGenerator, ScriptGenerationRequest
from .search_refinement import SearchRefinementEngine
from .error_handling import (
    handle_node_error_with_recovery,
    handle_api_errors,
    handle_transcript_errors,
    validate_processing_state,
    log_processing_metrics,
    RetryConfig
)

# Configure logging
logger = logging.getLogger(__name__)


# Error handling is now imported from error_handling module


@handle_node_error_with_recovery("discover_videos", allow_partial_failure=True)
def discover_videos_node(state: CuratorState) -> CuratorState:
    """
    Discovers trending YouTube videos based on search keywords.
    
    Focuses on videos from the past week for weekly podcast production
    and implements iterative search refinement if insufficient results.
    
    Args:
        state: Current curator state
        
    Returns:
        Updated state with discovered videos
    """
    # Validate state before processing
    if not validate_processing_state(state, "discover_videos"):
        return state
    
    # Log processing metrics
    log_processing_metrics(state, "discover_videos")
    
    config = get_config()
    youtube_client = YouTubeClient(config)
    
    # Use current search terms if available (from refinement), otherwise original keywords
    search_keywords = state.current_search_terms if state.current_search_terms else state.search_keywords
    
    logger.info(f"Discovering videos with keywords: {search_keywords}")
    logger.info(f"Search parameters: max_videos={state.max_videos}, days_back={state.days_back}")
    
    # Perform video discovery with API error handling
    @handle_api_errors("video_discovery", retry_config=RetryConfig(max_attempts=3, base_delay=2.0))
    def discover_videos_with_retry():
        return youtube_client.discover_videos(
            keywords=search_keywords,
            max_videos=state.max_videos,
            days_back=state.days_back
        )
    
    discovered_videos = discover_videos_with_retry()
    if discovered_videos is None:
        discovered_videos = []
    
    # Update state with discovered videos
    state.discovered_videos = discovered_videos
    
    # Update metadata
    state.update_generation_metadata(
        discovery_timestamp=datetime.now().isoformat(),
        search_keywords_used=search_keywords,
        videos_discovered=len(discovered_videos),
        search_attempt=state.search_attempt
    )
    
    logger.info(f"Video discovery complete: {len(discovered_videos)} videos found")
    
    if not discovered_videos:
        state.add_error("No videos found during discovery", "discover_videos")
    
    return state


@handle_node_error_with_recovery("fetch_video_details", allow_partial_failure=True)
def fetch_video_details_node(state: CuratorState) -> CuratorState:
    """
    Fetches detailed metadata for discovered videos.
    
    Runs in parallel with transcript fetching to improve efficiency.
    Updates videos with engagement metrics and detailed information.
    
    Args:
        state: Current curator state
        
    Returns:
        Updated state with detailed video metadata
    """
    # Validate state before processing
    if not validate_processing_state(state, "fetch_video_details"):
        return state
    
    # Log processing metrics
    log_processing_metrics(state, "fetch_video_details")
    
    if not state.discovered_videos:
        logger.warning("No videos available for details fetching")
        return state
    
    config = get_config()
    youtube_client = YouTubeClient(config)
    
    # Extract video IDs for batch processing
    video_ids = [video.video_id for video in state.discovered_videos]
    
    logger.info(f"Fetching detailed metadata for {len(video_ids)} videos")
    
    # Get detailed video information with retry mechanism
    @handle_api_errors("video_details_batch", retry_config=RetryConfig(max_attempts=3, base_delay=1.5))
    def fetch_details_with_retry():
        return youtube_client.get_video_details(video_ids)
    
    detailed_videos = fetch_details_with_retry()
    if detailed_videos is None:
        detailed_videos = []
    
    # Create mapping of video ID to detailed info
    details_map = {video['id']: video for video in detailed_videos}
    
    # Update discovered videos with detailed information (graceful degradation)
    updated_videos = []
    successful_updates = 0
    
    for video in state.discovered_videos:
        try:
            if video.video_id in details_map:
                detailed_info = details_map[video.video_id]
                
                # Safely update video with enhanced metadata
                try:
                    video.view_count = int(detailed_info.get('statistics', {}).get('viewCount', video.view_count))
                    video.like_count = int(detailed_info.get('statistics', {}).get('likeCount', video.like_count))
                    video.comment_count = int(detailed_info.get('statistics', {}).get('commentCount', video.comment_count))
                    successful_updates += 1
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse statistics for video {video.video_id}: {e}")
                
                # Add enhanced engagement metrics (optional)
                video.engagement_metrics = detailed_info.get('engagementMetrics', {})
                
                # Add video status and availability info (optional)
                video.video_status = detailed_info.get('status', {})
                
                # Add content details (optional)
                video.content_details = detailed_info.get('contentDetails', {})
                
                updated_videos.append(video)
            else:
                logger.debug(f"No detailed info found for video {video.video_id}, keeping original data")
                # Keep original video even if details fetch failed
                updated_videos.append(video)
                
        except Exception as e:
            logger.warning(f"Error processing details for video {video.video_id}: {e}")
            # Keep original video data
            updated_videos.append(video)
    
    state.discovered_videos = updated_videos
    
    # Update metadata
    state.update_generation_metadata(
        details_fetch_timestamp=datetime.now().isoformat(),
        videos_with_details=len([v for v in updated_videos if hasattr(v, 'engagement_metrics')]),
        details_fetch_success_rate=len(detailed_videos) / len(video_ids) if video_ids else 0
    )
    
    logger.info(f"Video details fetching complete: {len(detailed_videos)}/{len(video_ids)} successful")
    
    return state


@handle_node_error_with_recovery("fetch_transcripts", allow_partial_failure=True)
def fetch_transcripts_node(state: CuratorState) -> CuratorState:
    """
    Retrieves video transcripts for all discovered videos.
    
    Runs in parallel with details fetching to improve efficiency.
    Handles missing transcripts gracefully and logs availability issues.
    
    Args:
        state: Current curator state
        
    Returns:
        Updated state with transcript data
    """
    # Validate state before processing
    if not validate_processing_state(state, "fetch_transcripts"):
        return state
    
    # Log processing metrics
    log_processing_metrics(state, "fetch_transcripts")
    
    if not state.discovered_videos:
        logger.warning("No videos available for transcript fetching")
        state.add_error("No videos available for transcript fetching", "fetch_transcripts")
        return state
    
    config = get_config()
    transcript_processor = TranscriptProcessor(config=config)
    
    logger.info(f"Fetching transcripts for {len(state.discovered_videos)} videos")
    
    # Fetch transcripts for all videos with individual error handling
    videos_with_transcripts = 0
    transcript_errors = []
    
    for i, video in enumerate(state.discovered_videos):
        try:
            # Add delay between requests to avoid YouTube rate limiting
            if i > 0:  # Skip delay for first video
                delay = min(2.0 + (i * 0.5), 10.0)  # Progressive delay, max 10 seconds
                logger.debug(f"Adding {delay}s delay before fetching transcript for {video.video_id}")
                time.sleep(delay)
            
            # Apply transcript-specific error handling with retry
            @handle_transcript_errors
            @handle_api_errors("transcript_fetch", video_id=video.video_id, retry_config=RetryConfig(max_attempts=2, base_delay=0.5))
            def fetch_with_retry(video_id):
                return transcript_processor.fetch_transcript(video_id)
            
            transcript = fetch_with_retry(video.video_id)
            
            if transcript and transcript.strip():
                video.transcript = transcript
                videos_with_transcripts += 1
                logger.debug(f"Transcript fetched for {video.video_id}: {len(transcript)} characters")
            else:
                logger.debug(f"Empty transcript for {video.video_id}")
                transcript_errors.append(f"Empty transcript for video {video.video_id}")
                
        except Exception as e:
            error_msg = f"Failed to fetch transcript for {video.video_id}: {str(e)}"
            logger.warning(error_msg)
            transcript_errors.append(error_msg)
            # Continue processing other videos - non-blocking error
    
    # Log transcript errors for debugging
    if transcript_errors:
        for error in transcript_errors[:5]:  # Log first 5 errors to avoid spam
            state.add_error(error, "fetch_transcripts")
        
        if len(transcript_errors) > 5:
            state.add_error(f"... and {len(transcript_errors) - 5} more transcript errors", "fetch_transcripts")
    
    # Update metadata with comprehensive transcript information
    state.update_generation_metadata(
        transcript_fetch_timestamp=datetime.now().isoformat(),
        videos_with_transcripts=videos_with_transcripts,
        transcript_availability_rate=videos_with_transcripts / len(state.discovered_videos) if state.discovered_videos else 0,
        transcript_errors_count=len(transcript_errors),
        videos_processed_for_transcripts=len(state.discovered_videos)
    )
    
    logger.info(f"Transcript fetching complete: {videos_with_transcripts}/{len(state.discovered_videos)} videos have transcripts")
    
    if videos_with_transcripts == 0:
        state.add_error("No transcripts available for any discovered videos", "fetch_transcripts")
        logger.error("Critical: No transcripts available for script generation")
    elif videos_with_transcripts < len(state.discovered_videos) * 0.5:
        state.add_error(f"Low transcript availability: only {videos_with_transcripts}/{len(state.discovered_videos)} videos have transcripts", "fetch_transcripts")
        logger.warning(f"Low transcript availability rate: {videos_with_transcripts/len(state.discovered_videos)*100:.1f}%")
    
    return state


@handle_node_error_with_recovery("evaluate_quality", allow_partial_failure=True)
def evaluate_quality_node(state: CuratorState) -> CuratorState:
    """
    Evaluates video quality using engagement metrics and content analysis.
    
    Combines engagement scores with content quality analysis to create
    comprehensive quality assessments for each video.
    
    Args:
        state: Current curator state
        
    Returns:
        Updated state with quality-evaluated videos
    """
    # Validate state before processing
    if not validate_processing_state(state, "evaluate_quality"):
        return state
    
    # Log processing metrics
    log_processing_metrics(state, "evaluate_quality")
    
    if not state.discovered_videos:
        logger.warning("No videos available for quality evaluation")
        state.add_error("No videos available for quality evaluation", "evaluate_quality")
        return state
    
    config = get_config()
    engagement_analyzer = EngagementAnalyzer(config)
    content_scorer = ContentQualityScorer(config)
    transcript_processor = TranscriptProcessor(config=config)
    
    logger.info(f"Evaluating quality for {len(state.discovered_videos)} videos")
    
    processed_videos = []
    quality_scores = []
    evaluation_errors = []
    
    for video in state.discovered_videos:
        try:
            # Analyze engagement metrics with error handling
            try:
                engagement_metrics = engagement_analyzer.analyze_video_engagement(video)
            except Exception as e:
                logger.warning(f"Engagement analysis failed for {video.video_id}: {e}")
                # Create fallback engagement metrics
                engagement_metrics = type('EngagementMetrics', (), {
                    'overall_engagement_score': min(50.0, video.view_count / 1000),
                    'like_ratio': video.get_like_ratio(),
                    'engagement_rate': video.get_engagement_rate()
                })()
                evaluation_errors.append(f"Engagement analysis failed for {video.video_id}")
            
            # Analyze transcript content if available
            if video.transcript and video.transcript.strip():
                try:
                    # Perform transcript analysis with error handling
                    video = transcript_processor.analyze_transcript(video)
                    
                    # Calculate content quality metrics
                    content_metrics = content_scorer.calculate_combined_quality_score(video)
                    
                    # Store content analysis results
                    video.content_quality_analysis = content_metrics
                except Exception as e:
                    logger.warning(f"Content analysis failed for {video.video_id}: {e}")
                    # Fallback content score
                    video.content_analysis_score = 30.0  # Minimal score for having transcript
                    evaluation_errors.append(f"Content analysis failed for {video.video_id}")
            else:
                logger.debug(f"No transcript for quality analysis: {video.video_id}")
                # Create minimal content metrics for videos without transcripts
                video.content_analysis_score = 0.0
            
            # Store engagement analysis results
            video.engagement_analysis = engagement_metrics
            
            # Calculate combined quality score with error handling
            try:
                engagement_score = getattr(engagement_metrics, 'overall_engagement_score', 0.0)
                content_score = getattr(video, 'content_analysis_score', 0.0)
                
                # Weight: 60% engagement, 40% content (since not all videos have transcripts)
                if video.transcript and video.transcript.strip():
                    combined_score = (engagement_score * 0.6) + (content_score * 0.4)
                else:
                    # For videos without transcripts, use engagement score with penalty
                    combined_score = engagement_score * 0.8  # 20% penalty for missing transcript
                
                video.quality_score = max(0.0, min(100.0, combined_score))  # Clamp to 0-100 range
            except Exception as e:
                logger.warning(f"Quality score calculation failed for {video.video_id}: {e}")
                video.quality_score = 25.0  # Minimal fallback score
                evaluation_errors.append(f"Quality score calculation failed for {video.video_id}")
            
            quality_scores.append(video.quality_score)
            processed_videos.append(video)
            
            logger.debug(f"Quality evaluation for {video.video_id}: "
                        f"engagement={getattr(engagement_metrics, 'overall_engagement_score', 0.0):.1f}, "
                        f"content={getattr(video, 'content_analysis_score', 0.0):.1f}, "
                        f"combined={video.quality_score:.1f}")
            
        except Exception as e:
            logger.error(f"Critical quality evaluation failure for video {video.video_id}: {e}")
            # Keep video with minimal quality score for graceful degradation
            video.quality_score = 0.0
            quality_scores.append(0.0)
            processed_videos.append(video)
            evaluation_errors.append(f"Critical evaluation failure for {video.video_id}: {str(e)}")
    
    # Log evaluation errors for debugging
    if evaluation_errors:
        for error in evaluation_errors[:5]:  # Log first 5 errors to avoid spam
            state.add_error(error, "evaluate_quality")
        
        if len(evaluation_errors) > 5:
            state.add_error(f"... and {len(evaluation_errors) - 5} more evaluation errors", "evaluate_quality")
    
    state.processed_videos = processed_videos
    
    # Calculate quality statistics with error handling
    try:
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        videos_above_threshold = len([score for score in quality_scores if score >= state.quality_threshold])
        videos_with_transcripts = len([v for v in processed_videos if v.transcript and v.transcript.strip()])
    except Exception as e:
        logger.error(f"Error calculating quality statistics: {e}")
        avg_quality = 0.0
        videos_above_threshold = 0
        videos_with_transcripts = 0
        state.add_error(f"Quality statistics calculation failed: {str(e)}", "evaluate_quality")
    
    # Update metadata with comprehensive quality information
    state.update_generation_metadata(
        quality_evaluation_timestamp=datetime.now().isoformat(),
        videos_processed=len(processed_videos),
        average_quality_score=avg_quality,
        videos_above_threshold=videos_above_threshold,
        videos_with_transcripts=videos_with_transcripts,
        quality_threshold_used=state.quality_threshold,
        evaluation_errors_count=len(evaluation_errors),
        quality_evaluation_success_rate=(len(processed_videos) - len(evaluation_errors)) / len(processed_videos) if processed_videos else 0
    )
    
    logger.info(f"Quality evaluation complete: avg_score={avg_quality:.1f}, "
               f"{videos_above_threshold}/{len(processed_videos)} above threshold ({state.quality_threshold}%)")
    
    # Add warnings for quality issues
    if avg_quality < state.quality_threshold:
        state.add_error(f"Average quality score ({avg_quality:.1f}) below threshold ({state.quality_threshold})", "evaluate_quality")
        logger.warning(f"Average quality score below threshold: {avg_quality:.1f} < {state.quality_threshold}")
    
    if videos_above_threshold < state.min_quality_videos:
        state.add_error(f"Insufficient quality videos: {videos_above_threshold}/{state.min_quality_videos} required", "evaluate_quality")
        logger.warning(f"Insufficient quality videos for script generation: {videos_above_threshold}/{state.min_quality_videos}")
    
    if len(evaluation_errors) > len(processed_videos) * 0.5:
        state.add_error(f"High evaluation error rate: {len(evaluation_errors)}/{len(processed_videos)} videos had errors", "evaluate_quality")
        logger.error(f"High evaluation error rate: {len(evaluation_errors)}/{len(processed_videos)} videos failed evaluation")
    
    return state


@handle_node_error_with_recovery("rank_videos", allow_partial_failure=True)
def rank_videos_node(state: CuratorState) -> CuratorState:
    """
    Ranks videos by quality score and evaluates if results meet threshold.
    
    Triggers discovery refinement if insufficient quality videos found.
    Selects top 3-5 videos for script generation.
    
    Args:
        state: Current curator state
        
    Returns:
        Updated state with ranked videos and quality assessment
    """
    if not state.processed_videos:
        logger.warning("No processed videos available for ranking")
        return state
    
    config = get_config()
    ranking_system = VideoRankingSystem(config)
    
    logger.info(f"Ranking {len(state.processed_videos)} processed videos")
    
    # Update ranking system with current state values
    ranking_system.quality_threshold = state.quality_threshold
    ranking_system.min_quality_videos = state.min_quality_videos
    
    # Perform comprehensive video ranking
    ranking_result = ranking_system.rank_videos(
        videos=state.processed_videos,
        strategy=RankingStrategy.BALANCED,
        target_count=min(5, len(state.processed_videos))  # Top 3-5 videos
    )
    
    # Update state with ranking results
    state.ranked_videos = ranking_result.ranked_videos
    
    # Update metadata with ranking information
    state.update_generation_metadata(
        ranking_timestamp=datetime.now().isoformat(),
        ranking_strategy=ranking_result.ranking_metadata['strategy'],
        videos_ranked=len(ranking_result.ranked_videos),
        quality_summary=ranking_result.quality_summary,
        threshold_analysis=ranking_result.threshold_analysis,
        meets_quality_requirement=ranking_result.threshold_analysis['meets_minimum_requirement']
    )
    
    # Log ranking results
    logger.info(f"Video ranking complete: {len(ranking_result.ranked_videos)} videos selected")
    logger.info(f"Quality threshold analysis: {ranking_result.threshold_analysis['videos_above_threshold']}"
               f"/{ranking_result.threshold_analysis['total_videos']} videos above {state.quality_threshold}% threshold")
    
    # Add refinement suggestions if quality requirements not met
    if not ranking_result.threshold_analysis['meets_minimum_requirement']:
        for suggestion in ranking_result.refinement_suggestions:
            state.add_error(f"Quality refinement needed: {suggestion}", "rank_videos")
        
        logger.warning(f"Quality requirements not met: need {state.min_quality_videos} videos above "
                      f"{state.quality_threshold}% threshold, got {ranking_result.threshold_analysis['videos_above_threshold']}")
    
    return state


@handle_node_error_with_recovery("generate_script", allow_partial_failure=True)
def generate_script_node(state: CuratorState) -> CuratorState:
    """
    Generates podcast script from top-ranked videos.
    
    Creates cohesive narrative combining insights from top 3-5 videos
    with automatic length management and source attribution.
    
    Args:
        state: Current curator state
        
    Returns:
        Updated state with generated podcast script
    """
    # Validate state before processing
    if not validate_processing_state(state, "generate_script"):
        return state
    
    # Log processing metrics
    log_processing_metrics(state, "generate_script")
    
    if not state.ranked_videos:
        logger.warning("No ranked videos available for script generation")
        state.add_error("No videos available for script generation", "generate_script")
        return state
    
    # Filter videos with transcripts for script generation
    videos_with_transcripts = [v for v in state.ranked_videos if v.transcript and v.transcript.strip()]
    
    if not videos_with_transcripts:
        logger.error("No videos with transcripts available for script generation")
        state.add_error("No videos with transcripts available for script generation", "generate_script")
        return state
    
    config = get_config()
    
    logger.info(f"Generating script from {len(videos_with_transcripts)} videos with transcripts")
    
    # Attempt script generation with retry mechanism and fallback
    script_generated = False
    generation_errors = []
    
    try:
        script_generator = OpenAIScriptGenerator(config.openai_api_key)
        
        # Create script generation request with error handling
        try:
            request = ScriptGenerationRequest(
                videos=videos_with_transcripts,
                target_word_count_min=750,  # 5 minutes minimum
                target_word_count_max=1500,  # 10 minutes maximum
                language="en",
                style="conversational"
            )
        except Exception as e:
            logger.error(f"Failed to create script generation request: {e}")
            generation_errors.append(f"Request creation failed: {str(e)}")
            raise
        
        # Generate the script with API error handling and retry
        @handle_api_errors("script_generation", retry_config=RetryConfig(max_attempts=3, base_delay=2.0, max_delay=30.0))
        def generate_with_retry():
            return script_generator.generate_script(request)
        
        response = generate_with_retry()
        
        if response and response.script and response.script.strip():
            # Update state with generated script
            state.podcast_script = response.script
            script_generated = True
            
            # Validate script quality
            if response.word_count < 500:
                state.add_error(f"Generated script is too short: {response.word_count} words", "generate_script")
                logger.warning(f"Generated script is shorter than expected: {response.word_count} words")
            elif response.word_count > 2000:
                state.add_error(f"Generated script is too long: {response.word_count} words", "generate_script")
                logger.warning(f"Generated script is longer than expected: {response.word_count} words")
            
        else:
            raise ValueError("Empty or invalid script response from generator")
            
    except Exception as e:
        logger.error(f"Script generation failed: {e}")
        generation_errors.append(f"Script generation failed: {str(e)}")
        
        # Attempt fallback script generation using error handling recovery
        logger.info("Attempting fallback script generation")
        try:
            from .error_handling import _create_fallback_script
            fallback_script = _create_fallback_script(videos_with_transcripts)
            state.podcast_script = fallback_script
            script_generated = True
            
            # Create minimal response object for metadata
            response = type('FallbackResponse', (), {
                'script': fallback_script,
                'word_count': len(fallback_script.split()),
                'estimated_duration_minutes': len(fallback_script.split()) / 155.0,
                'source_videos': [v.video_id for v in videos_with_transcripts],
                'generation_metadata': {'fallback_used': True, 'original_error': str(e)}
            })()
            
            state.add_error(f"Used fallback script generation due to: {str(e)}", "generate_script")
            logger.warning("Fallback script generation successful")
            
        except Exception as fallback_error:
            logger.error(f"Fallback script generation also failed: {fallback_error}")
            generation_errors.append(f"Fallback generation failed: {str(fallback_error)}")
            state.add_error(f"Both primary and fallback script generation failed", "generate_script")
            return state
    
    # Log generation errors
    if generation_errors:
        for error in generation_errors:
            state.add_error(error, "generate_script")
    
    # Update state with script if generated
    if script_generated and hasattr(response, 'script'):
    
        # Update metadata with script generation information
        state.update_generation_metadata(
            script_generation_timestamp=datetime.now().isoformat(),
            script_word_count=getattr(response, 'word_count', len(state.podcast_script.split()) if state.podcast_script else 0),
            script_estimated_duration=getattr(response, 'estimated_duration_minutes', 0.0),
            source_videos_used=getattr(response, 'source_videos', [v.video_id for v in videos_with_transcripts]),
            script_generation_metadata=getattr(response, 'generation_metadata', {}),
            script_generation_errors_count=len(generation_errors),
            script_generation_success=script_generated,
            videos_used_for_generation=len(videos_with_transcripts)
        )
        
        logger.info(f"Script generation complete: {getattr(response, 'word_count', 0)} words, "
                   f"{getattr(response, 'estimated_duration_minutes', 0.0):.1f} minutes estimated duration")
        
        # Log synthesis quality if available
        generation_metadata = getattr(response, 'generation_metadata', {})
        synthesis_quality = generation_metadata.get('synthesis_quality', {})
        if synthesis_quality:
            logger.info(f"Script synthesis quality: structure_score={synthesis_quality.get('structure_score', 0):.2f}, "
                       f"sources_referenced={len(synthesis_quality.get('sources_referenced', []))}")
        
        # Log fallback usage if applicable
        if generation_metadata.get('fallback_used'):
            logger.info("Fallback script generation was used due to primary generation failure")
    
    else:
        logger.error("Script generation completely failed - no script available")
        state.add_error("Complete script generation failure", "generate_script")
    
    return state


@handle_node_error_with_recovery("store_results", allow_partial_failure=False)
def store_results_node(state: CuratorState) -> CuratorState:
    """
    Stores generated script and metadata including search refinement history.
    
    Persists results with comprehensive metadata for analysis and future reference.
    
    Args:
        state: Current curator state
        
    Returns:
        Updated state with storage confirmation
    """
    # Validate state before processing
    if not validate_processing_state(state, "store_results"):
        return state
    
    # Log processing metrics
    log_processing_metrics(state, "store_results")
    
    if not state.podcast_script or not state.podcast_script.strip():
        logger.warning("No podcast script available for storage")
        state.add_error("No podcast script available for storage", "store_results")
        return state
    
    logger.info("Storing results and metadata")
    
    # Create comprehensive storage metadata with error handling
    try:
        storage_metadata = {
            'storage_timestamp': datetime.now().isoformat(),
            'workflow_completion_time': datetime.now().isoformat(),
            'total_processing_time': 'calculated_by_external_system',  # Will be calculated by caller
            'final_state_summary': state.get_processing_summary(),
            'script_metadata': {
                'word_count': len(state.podcast_script.split()) if state.podcast_script else 0,
                'estimated_duration': len(state.podcast_script.split()) / 155.0 if state.podcast_script else 0.0,  # 155 WPM average
                'source_video_count': len(state.ranked_videos),
                'quality_videos_used': len([v for v in state.ranked_videos if v.quality_score and v.quality_score >= state.quality_threshold])
            },
            'search_refinement_summary': {
                'total_attempts': state.search_attempt,
                'final_search_terms': state.current_search_terms,
                'original_search_terms': state.search_keywords,
                'videos_discovered': len(state.discovered_videos),
                'videos_processed': len(state.processed_videos),
                'videos_ranked': len(state.ranked_videos)
            },
            'quality_analysis': {
                'threshold_used': state.quality_threshold,
                'minimum_required': state.min_quality_videos,
                'threshold_met': state.has_sufficient_quality_videos(),
                'average_quality': sum(v.quality_score for v in state.ranked_videos if v.quality_score) / len(state.ranked_videos) if state.ranked_videos else 0
            },
            'error_summary': {
                'total_errors': len(state.errors),
                'errors': state.errors[-5:] if state.errors else []  # Last 5 errors
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating storage metadata: {e}")
        state.add_error(f"Storage metadata creation failed: {str(e)}", "store_results")
        # Create minimal metadata as fallback
        storage_metadata = {
            'storage_timestamp': datetime.now().isoformat(),
            'error': f"Metadata creation failed: {str(e)}",
            'script_available': bool(state.podcast_script),
            'total_errors': len(state.errors)
        }
    
    # Update state metadata with storage information
    try:
        state.update_generation_metadata(**storage_metadata)
    except Exception as e:
        logger.error(f"Error updating generation metadata: {e}")
        state.add_error(f"Metadata update failed: {str(e)}", "store_results")
    
    # Save results to files in the configured storage directory
    try:
        config = get_config()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ensure storage directory exists
        storage_path = config.results_storage_path
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Save the podcast script if available
        if state.podcast_script:
            script_file = storage_path / f"script_{timestamp}.txt"
            script_file.write_text(state.podcast_script, encoding='utf-8')
            logger.info(f"Script saved to: {script_file}")
        
        # Save metadata as JSON
        metadata_file = storage_path / f"metadata_{timestamp}.json"
        metadata_content = {
            'timestamp': timestamp,
            'search_keywords': state.search_keywords,
            'search_attempt': state.search_attempt,
            'videos_discovered': len(state.discovered_videos),
            'videos_processed': len(state.processed_videos),
            'videos_ranked': len(state.ranked_videos),
            'quality_threshold': state.quality_threshold,
            'min_quality_videos': state.min_quality_videos,
            'generation_metadata': state.generation_metadata,
            'errors': state.errors,
            'storage_metadata': storage_metadata
        }
        
        import json
        metadata_file.write_text(json.dumps(metadata_content, indent=2, default=str), encoding='utf-8')
        logger.info(f"Metadata saved to: {metadata_file}")
        
        # Save video data as JSON for reference
        video_data_file = storage_path / f"videos_{timestamp}.json"
        videos_data = [
            {
                'video_id': v.video_id,
                'title': v.title,
                'channel': v.channel,
                'upload_date': v.upload_date,
                'view_count': v.view_count,
                'like_count': v.like_count,
                'comment_count': v.comment_count,
                'quality_score': v.quality_score,
                'has_transcript': bool(v.transcript)
            }
            for v in state.ranked_videos
        ]
        
        video_data_file.write_text(json.dumps(videos_data, indent=2, default=str), encoding='utf-8')
        logger.info(f"Video data saved to: {video_data_file}")
        
        script_word_count = storage_metadata.get('script_metadata', {}).get('word_count', 0)
        source_video_count = storage_metadata.get('script_metadata', {}).get('source_video_count', 0)
        
        logger.info(f"Results stored successfully: script with {script_word_count} words, "
                   f"{source_video_count} source videos, saved to {storage_path}")
        
    except Exception as e:
        logger.error(f"Error saving results to file system: {e}")
        state.add_error(f"File storage failed: {str(e)}", "store_results")
        
        # Still log summary even if file saving failed
        try:
            script_word_count = storage_metadata.get('script_metadata', {}).get('word_count', 0)
            source_video_count = storage_metadata.get('script_metadata', {}).get('source_video_count', 0)
            logger.info(f"Results processed (storage failed): script with {script_word_count} words, "
                       f"{source_video_count} source videos")
        except Exception:
            logger.info("Results processed but storage and summary failed")
    
    try:
        # Log final processing summary
        summary = state.get_processing_summary()
        logger.info(f"Workflow complete - Final summary: {summary}")
        
        # Log error summary if there were issues
        if state.errors:
            logger.warning(f"Workflow completed with {len(state.errors)} errors. Recent errors:")
            for error in state.errors[-3:]:  # Show last 3 errors
                logger.warning(f"  - {error}")
                
    except Exception as e:
        logger.error(f"Error logging storage results: {e}")
        state.add_error(f"Storage logging failed: {str(e)}", "store_results")
    
    return state


@handle_node_error_with_recovery("refine_search", allow_partial_failure=True)
def refine_search_node(state: CuratorState) -> CuratorState:
    """
    Refines search parameters based on previous attempt results.
    
    Implements progressive search strategy for better content discovery
    with keyword expansion, timeframe adjustment, and broader term usage.
    
    Args:
        state: Current curator state
        
    Returns:
        Updated state with refined search parameters
    """
    # Validate state before processing
    if not validate_processing_state(state, "refine_search"):
        return state
    
    # Log processing metrics
    log_processing_metrics(state, "refine_search")
    
    logger.info(f"Refining search parameters - attempt {state.search_attempt + 1}/{state.max_search_attempts}")
    
    # Check if we can still refine
    if not state.can_refine_search():
        logger.warning("Maximum search refinement attempts reached")
        state.add_error("Maximum search refinement attempts reached", "refine_search")
        return state
    
    config = get_config()
    
    try:
        refinement_engine = SearchRefinementEngine(config)
        
        # Perform iterative search refinement with error handling
        try:
            updated_state = refinement_engine.refine_search_iteratively(state)
        except Exception as e:
            logger.error(f"Search refinement engine failed: {e}")
            state.add_error(f"Search refinement failed: {str(e)}", "refine_search")
            
            # Fallback: manual search term expansion
            logger.info("Attempting fallback search refinement")
            updated_state = _fallback_search_refinement(state)
        
        # Validate refinement results
        if updated_state.search_attempt <= state.search_attempt:
            logger.warning("Search refinement did not increment attempt counter")
            updated_state.search_attempt = state.search_attempt + 1
        
        # Log refinement results with error handling
        try:
            refinement_summary = refinement_engine.get_refinement_summary(updated_state)
            logger.info(f"Search refinement complete: {refinement_summary['total_attempts']} attempts, "
                       f"{refinement_summary['final_video_count']} videos found")
            
            if refinement_summary['quality_threshold_met']:
                logger.info("Quality threshold requirements met after refinement")
            else:
                logger.warning("Quality threshold requirements still not met after refinement")
                
            # Update metadata with refinement information
            updated_state.update_generation_metadata(
                refinement_timestamp=datetime.now().isoformat(),
                refinement_summary=refinement_summary,
                refinement_success=True
            )
            
        except Exception as e:
            logger.warning(f"Error getting refinement summary: {e}")
            updated_state.add_error(f"Refinement summary failed: {str(e)}", "refine_search")
            
            # Create basic summary
            updated_state.update_generation_metadata(
                refinement_timestamp=datetime.now().isoformat(),
                refinement_attempt=updated_state.search_attempt,
                refinement_success=True
            )
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Critical search refinement failure: {e}")
        state.add_error(f"Critical refinement failure: {str(e)}", "refine_search")
        
        # Return state with incremented attempt to prevent infinite loops
        state.search_attempt += 1
        return state


def _fallback_search_refinement(state: CuratorState) -> CuratorState:
    """
    Fallback search refinement when the main refinement engine fails.
    
    Args:
        state: Current curator state
        
    Returns:
        Updated state with basic search refinement
    """
    logger.info("Applying fallback search refinement")
    
    # Basic keyword expansion
    current_terms = state.current_search_terms or state.search_keywords
    
    # Simple expansion based on attempt number
    if state.search_attempt == 0:
        # Add synonyms
        expanded_terms = current_terms + ["artificial intelligence", "machine learning", "AI tools"]
    elif state.search_attempt == 1:
        # Add broader terms
        expanded_terms = current_terms + ["technology", "innovation", "automation"]
    else:
        # Add very broad terms
        expanded_terms = current_terms + ["tech news", "software", "digital"]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in expanded_terms:
        if term.lower() not in seen:
            seen.add(term.lower())
            unique_terms.append(term)
    
    state.current_search_terms = unique_terms[:10]  # Limit to 10 terms
    state.search_attempt += 1
    
    logger.info(f"Fallback refinement applied: {len(unique_terms)} search terms")
    state.add_error("Used fallback search refinement due to engine failure", "refine_search")
    
    return state

def should_refine_search(state: CuratorState) -> str:
    """
    Determine if search refinement is needed based on discovery results.
    
    Args:
        state: Current curator state
        
    Returns:
        Next node name ("refine_search" or "parallel_processing")
    """
    # Check if we have any videos
    if not state.discovered_videos:
        if state.can_refine_search():
            logger.info("No videos found, proceeding to search refinement")
            return "refine_search"
        else:
            logger.warning("No videos found and max refinement attempts reached")
            return "parallel_processing"  # Continue with empty results
    
    # Check if we have enough videos for quality assessment
    min_videos_for_processing = 3
    if len(state.discovered_videos) < min_videos_for_processing:
        if state.can_refine_search():
            logger.info(f"Only {len(state.discovered_videos)} videos found (need {min_videos_for_processing}), "
                       f"proceeding to search refinement")
            return "refine_search"
        else:
            logger.warning(f"Only {len(state.discovered_videos)} videos found but max refinement attempts reached")
    
    logger.info(f"Sufficient videos found ({len(state.discovered_videos)}), proceeding to parallel processing")
    return "parallel_processing"


def should_refine_after_ranking(state: CuratorState) -> str:
    """
    Determine if search refinement is needed based on ranking results.
    
    Args:
        state: Current curator state
        
    Returns:
        Next node name ("refine_search" or "generate_script")
    """
    # Check if quality threshold is met
    if state.has_sufficient_quality_videos():
        logger.info("Quality threshold met, proceeding to script generation")
        return "generate_script"
    
    # Check if we can still refine
    if state.can_refine_search():
        logger.info(f"Quality threshold not met ({len([v for v in state.ranked_videos if v.quality_score and v.quality_score >= state.quality_threshold])}"
                   f"/{state.min_quality_videos} videos above {state.quality_threshold}%), proceeding to search refinement")
        return "refine_search"
    else:
        logger.warning("Quality threshold not met but max refinement attempts reached, proceeding with available videos")
        return "generate_script"


def create_curator_workflow(config: Optional[Configuration] = None):
    """
    Create the complete LangGraph workflow for the nanook-curator system.
    
    Implements the full workflow with parallel processing, conditional routing,
    and iterative refinement capabilities.
    
    Args:
        config: Optional configuration instance
        
    Returns:
        Compiled LangGraph workflow
    """
    logger.info("Creating nanook-curator LangGraph workflow")
    
    # Create the state graph with CuratorState schema
    workflow = StateGraph(CuratorState)
    
    # Add all processing nodes with error handling decorators already applied
    workflow.add_node("discover_videos", discover_videos_node)
    workflow.add_node("refine_search", refine_search_node)
    workflow.add_node("fetch_details", fetch_video_details_node)
    workflow.add_node("fetch_transcripts", fetch_transcripts_node)
    workflow.add_node("evaluate_quality", evaluate_quality_node)
    workflow.add_node("rank_videos", rank_videos_node)
    workflow.add_node("generate_script", generate_script_node)
    workflow.add_node("store_results", store_results_node)
    
    # Set entry point
    workflow.set_entry_point("discover_videos")
    
    # Conditional edges for discovery success/failure routing
    workflow.add_conditional_edges(
        "discover_videos",
        should_refine_search,
        {
            "refine_search": "refine_search",
            "parallel_processing": "fetch_details"
        }
    )
    
    # Refinement loop - goes back to discovery after search parameter adjustment
    workflow.add_edge("refine_search", "discover_videos")
    
    # Add parallel processing edges for video details and transcript fetching
    # When fetch_details is reached (successful discovery), also trigger fetch_transcripts
    # This creates parallel execution of both nodes
    workflow.add_edge("fetch_details", "fetch_transcripts")
    
    # Both parallel processes feed into quality evaluation
    # LangGraph will wait for both to complete before proceeding to evaluate_quality
    workflow.add_edge("fetch_details", "evaluate_quality") 
    workflow.add_edge("fetch_transcripts", "evaluate_quality")
    
    # Sequential processing after parallel completion
    workflow.add_edge("evaluate_quality", "rank_videos")
    
    # Quality-based conditional routing for refinement loops
    workflow.add_conditional_edges(
        "rank_videos",
        should_refine_after_ranking,
        {
            "refine_search": "refine_search",
            "generate_script": "generate_script"
        }
    )
    
    # Final processing steps
    workflow.add_edge("generate_script", "store_results")
    workflow.add_edge("store_results", END)
    
    # Compile the workflow
    compiled_workflow = workflow.compile()
    
    logger.info("LangGraph workflow created successfully with parallel processing and conditional routing")
    
    return compiled_workflow


def run_curator_workflow(
    search_keywords: List[str],
    max_videos: int = 10,
    days_back: int = 7,
    quality_threshold: float = 70.0,
    config: Optional[Configuration] = None
) -> CuratorState:
    """
    Run the complete curator workflow with specified parameters.
    
    Args:
        search_keywords: Keywords for video discovery
        max_videos: Maximum number of videos to discover
        days_back: Days back to search for videos
        quality_threshold: Minimum quality score threshold
        config: Optional configuration instance
        
    Returns:
        Final curator state with results
        
    Raises:
        ValueError: If required parameters are invalid
        RuntimeError: If workflow execution fails
    """
    if not search_keywords:
        raise ValueError("Search keywords are required")
    
    logger.info(f"Starting curator workflow with keywords: {search_keywords}")
    
    # Create initial state
    initial_state = CuratorState(
        search_keywords=search_keywords,
        max_videos=max_videos,
        days_back=days_back,
        quality_threshold=quality_threshold,
        current_search_terms=search_keywords.copy()
    )
    
    # Create and run workflow
    workflow = create_curator_workflow(config)
    
    try:
        # Execute the workflow
        start_time = datetime.now()
        final_state = workflow.invoke(initial_state)
        end_time = datetime.now()
        
        # Calculate total processing time
        processing_time = (end_time - start_time).total_seconds()
        final_state.update_generation_metadata(
            total_processing_time_seconds=processing_time,
            workflow_start_time=start_time.isoformat(),
            workflow_end_time=end_time.isoformat()
        )
        
        logger.info(f"Curator workflow completed successfully in {processing_time:.2f} seconds")
        
        # Log final results summary
        if final_state.podcast_script:
            script_word_count = len(final_state.podcast_script.split())
            logger.info(f"Generated podcast script: {script_word_count} words from {len(final_state.ranked_videos)} videos")
        else:
            logger.warning("Workflow completed but no podcast script was generated")
        
        return final_state
        
    except Exception as e:
        error_msg = f"Workflow execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


# Utility functions for workflow analysis
def analyze_workflow_performance(state: CuratorState) -> Dict[str, Any]:
    """
    Analyze workflow performance and provide insights.
    
    Args:
        state: Final curator state
        
    Returns:
        Performance analysis dictionary
    """
    metadata = state.generation_metadata
    
    analysis = {
        'execution_summary': {
            'total_time_seconds': metadata.get('total_processing_time_seconds', 0),
            'search_attempts': state.search_attempt,
            'videos_discovered': len(state.discovered_videos),
            'videos_processed': len(state.processed_videos),
            'videos_ranked': len(state.ranked_videos),
            'script_generated': state.podcast_script is not None
        },
        'quality_metrics': {
            'average_quality_score': metadata.get('quality_analysis', {}).get('average_quality', 0),
            'threshold_met': metadata.get('quality_analysis', {}).get('threshold_met', False),
            'videos_above_threshold': metadata.get('videos_above_threshold', 0)
        },
        'efficiency_metrics': {
            'discovery_success_rate': len(state.discovered_videos) / state.max_videos if state.max_videos > 0 else 0,
            'transcript_availability_rate': metadata.get('transcript_availability_rate', 0),
            'processing_success_rate': len(state.processed_videos) / len(state.discovered_videos) if state.discovered_videos else 0
        },
        'error_analysis': {
            'total_errors': len(state.errors),
            'error_rate': len(state.errors) / max(len(state.discovered_videos), 1),
            'critical_errors': [error for error in state.errors if 'failed' in error.lower()]
        },
        'recommendations': _generate_performance_recommendations(state)
    }
    
    return analysis


def _generate_performance_recommendations(state: CuratorState) -> List[str]:
    """Generate performance improvement recommendations."""
    recommendations = []
    metadata = state.generation_metadata
    
    # Search efficiency recommendations
    if state.search_attempt > 2:
        recommendations.append("Consider refining default search keywords to reduce refinement attempts")
    
    # Transcript availability recommendations
    transcript_rate = metadata.get('transcript_availability_rate', 0)
    if transcript_rate < 0.5:
        recommendations.append("Low transcript availability. Consider targeting educational or professional content")
    
    # Quality threshold recommendations
    if not state.has_sufficient_quality_videos() and state.search_attempt >= state.max_search_attempts:
        recommendations.append("Consider lowering quality threshold or expanding search scope")
    
    # Processing efficiency recommendations
    if len(state.errors) > 5:
        recommendations.append("High error rate detected. Review API configuration and error handling")
    
    # Success recommendations
    if state.podcast_script and len(state.ranked_videos) >= 3:
        recommendations.append("Workflow executed successfully. Consider using similar parameters for future runs")
    
    return recommendations