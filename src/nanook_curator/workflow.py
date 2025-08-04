"""
LangGraph workflow orchestration for the nanook-curator system.

This module implements the complete LangGraph workflow with nodes for each processing step,
including video discovery, transcript fetching, quality evaluation, ranking, script generation,
and iterative search refinement with parallel processing capabilities.
"""

import logging
from typing import Dict, Any, List, Optional
from functools import wraps
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph

from .models import VideoData, CuratorState
from .config import get_config, Configuration
from .youtube_client import YouTubeClient, SearchFilters
from .transcript_processor import TranscriptProcessor
from .engagement_analyzer import EngagementAnalyzer
from .content_quality_scorer import ContentQualityScorer
from .video_ranking_system import VideoRankingSystem, RankingStrategy
from .script_generator import OpenAIScriptGenerator, ScriptGenerationRequest
from .search_refinement import SearchRefinementEngine

# Configure logging
logger = logging.getLogger(__name__)


def handle_node_error(node_name: str):
    """
    Decorator for graceful error handling in LangGraph nodes.
    
    Args:
        node_name: Name of the node for error context
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(state: CuratorState) -> CuratorState:
            try:
                logger.info(f"Starting {node_name} node")
                result = func(state)
                logger.info(f"Completed {node_name} node successfully")
                return result
            except Exception as e:
                error_msg = f"{node_name} node failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                state.add_error(error_msg, node_name)
                return state
        return wrapper
    return decorator


@handle_node_error("discover_videos")
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
    config = get_config()
    youtube_client = YouTubeClient(config)
    
    # Use current search terms if available (from refinement), otherwise original keywords
    search_keywords = state.current_search_terms if state.current_search_terms else state.search_keywords
    
    logger.info(f"Discovering videos with keywords: {search_keywords}")
    logger.info(f"Search parameters: max_videos={state.max_videos}, days_back={state.days_back}")
    
    # Perform video discovery
    discovered_videos = youtube_client.discover_videos(
        keywords=search_keywords,
        max_videos=state.max_videos,
        days_back=state.days_back
    )
    
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


@handle_node_error("fetch_video_details")
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
    if not state.discovered_videos:
        logger.warning("No videos available for details fetching")
        return state
    
    config = get_config()
    youtube_client = YouTubeClient(config)
    
    # Extract video IDs for batch processing
    video_ids = [video.video_id for video in state.discovered_videos]
    
    logger.info(f"Fetching detailed metadata for {len(video_ids)} videos")
    
    # Get detailed video information
    detailed_videos = youtube_client.get_video_details(video_ids)
    
    # Create mapping of video ID to detailed info
    details_map = {video['id']: video for video in detailed_videos}
    
    # Update discovered videos with detailed information
    updated_videos = []
    for video in state.discovered_videos:
        if video.video_id in details_map:
            detailed_info = details_map[video.video_id]
            
            # Update video with enhanced metadata
            video.view_count = detailed_info['statistics']['viewCount']
            video.like_count = detailed_info['statistics']['likeCount']
            video.comment_count = detailed_info['statistics']['commentCount']
            
            # Add enhanced engagement metrics
            video.engagement_metrics = detailed_info.get('engagementMetrics', {})
            
            # Add video status and availability info
            video.video_status = detailed_info.get('status', {})
            
            # Add content details
            video.content_details = detailed_info.get('contentDetails', {})
            
            updated_videos.append(video)
        else:
            logger.warning(f"No detailed info found for video {video.video_id}")
            # Keep original video even if details fetch failed
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


@handle_node_error("fetch_transcripts")
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
    if not state.discovered_videos:
        logger.warning("No videos available for transcript fetching")
        return state
    
    config = get_config()
    transcript_processor = TranscriptProcessor(config=config)
    
    logger.info(f"Fetching transcripts for {len(state.discovered_videos)} videos")
    
    # Fetch transcripts for all videos
    videos_with_transcripts = 0
    for video in state.discovered_videos:
        transcript = transcript_processor.fetch_transcript(video.video_id)
        if transcript:
            video.transcript = transcript
            videos_with_transcripts += 1
            logger.debug(f"Transcript fetched for {video.video_id}: {len(transcript)} characters")
        else:
            logger.debug(f"No transcript available for {video.video_id}")
    
    # Update metadata
    state.update_generation_metadata(
        transcript_fetch_timestamp=datetime.now().isoformat(),
        videos_with_transcripts=videos_with_transcripts,
        transcript_availability_rate=videos_with_transcripts / len(state.discovered_videos) if state.discovered_videos else 0
    )
    
    logger.info(f"Transcript fetching complete: {videos_with_transcripts}/{len(state.discovered_videos)} videos have transcripts")
    
    if videos_with_transcripts == 0:
        state.add_error("No transcripts available for any discovered videos", "fetch_transcripts")
    
    return state


@handle_node_error("evaluate_quality")
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
    if not state.discovered_videos:
        logger.warning("No videos available for quality evaluation")
        return state
    
    config = get_config()
    engagement_analyzer = EngagementAnalyzer(config)
    content_scorer = ContentQualityScorer(config)
    transcript_processor = TranscriptProcessor(config=config)
    
    logger.info(f"Evaluating quality for {len(state.discovered_videos)} videos")
    
    processed_videos = []
    quality_scores = []
    
    for video in state.discovered_videos:
        try:
            # Analyze engagement metrics
            engagement_metrics = engagement_analyzer.analyze_video_engagement(video)
            
            # Analyze transcript content if available
            if video.transcript:
                # Perform transcript analysis
                video = transcript_processor.analyze_transcript(video)
                
                # Calculate content quality metrics
                content_metrics = content_scorer.calculate_combined_quality_score(video)
                
                # Store content analysis results
                video.content_quality_analysis = content_metrics
            else:
                logger.debug(f"No transcript for quality analysis: {video.video_id}")
                # Create minimal content metrics for videos without transcripts
                video.content_analysis_score = 0.0
            
            # Store engagement analysis results
            video.engagement_analysis = engagement_metrics
            
            # Calculate combined quality score
            engagement_score = engagement_metrics.overall_engagement_score
            content_score = getattr(video, 'content_analysis_score', 0.0)
            
            # Weight: 60% engagement, 40% content (since not all videos have transcripts)
            if video.transcript:
                combined_score = (engagement_score * 0.6) + (content_score * 0.4)
            else:
                # For videos without transcripts, use engagement score with penalty
                combined_score = engagement_score * 0.8  # 20% penalty for missing transcript
            
            video.quality_score = combined_score
            quality_scores.append(combined_score)
            processed_videos.append(video)
            
            logger.debug(f"Quality evaluation for {video.video_id}: "
                        f"engagement={engagement_score:.1f}, content={content_score:.1f}, "
                        f"combined={combined_score:.1f}")
            
        except Exception as e:
            logger.warning(f"Quality evaluation failed for video {video.video_id}: {e}")
            # Keep video with minimal quality score
            video.quality_score = 0.0
            processed_videos.append(video)
    
    state.processed_videos = processed_videos
    
    # Calculate quality statistics
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    videos_above_threshold = len([score for score in quality_scores if score >= state.quality_threshold])
    
    # Update metadata
    state.update_generation_metadata(
        quality_evaluation_timestamp=datetime.now().isoformat(),
        videos_processed=len(processed_videos),
        average_quality_score=avg_quality,
        videos_above_threshold=videos_above_threshold,
        quality_threshold_used=state.quality_threshold
    )
    
    logger.info(f"Quality evaluation complete: avg_score={avg_quality:.1f}, "
               f"{videos_above_threshold}/{len(processed_videos)} above threshold ({state.quality_threshold}%)")
    
    return state


@handle_node_error("rank_videos")
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


@handle_node_error("generate_script")
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
    script_generator = OpenAIScriptGenerator(config.openai_api_key)
    
    logger.info(f"Generating script from {len(videos_with_transcripts)} videos with transcripts")
    
    # Create script generation request
    request = ScriptGenerationRequest(
        videos=videos_with_transcripts,
        target_word_count_min=750,  # 5 minutes minimum
        target_word_count_max=1500,  # 10 minutes maximum
        language="en",
        style="conversational"
    )
    
    # Generate the script
    response = script_generator.generate_script(request)
    
    # Update state with generated script
    state.podcast_script = response.script
    
    # Update metadata with script generation information
    state.update_generation_metadata(
        script_generation_timestamp=datetime.now().isoformat(),
        script_word_count=response.word_count,
        script_estimated_duration=response.estimated_duration_minutes,
        source_videos_used=response.source_videos,
        script_generation_metadata=response.generation_metadata
    )
    
    logger.info(f"Script generation complete: {response.word_count} words, "
               f"{response.estimated_duration_minutes:.1f} minutes estimated duration")
    
    # Log synthesis quality if available
    synthesis_quality = response.generation_metadata.get('synthesis_quality', {})
    if synthesis_quality:
        logger.info(f"Script synthesis quality: structure_score={synthesis_quality.get('structure_score', 0):.2f}, "
                   f"sources_referenced={len(synthesis_quality.get('sources_referenced', []))}")
    
    return state


@handle_node_error("store_results")
def store_results_node(state: CuratorState) -> CuratorState:
    """
    Stores generated script and metadata including search refinement history.
    
    Persists results with comprehensive metadata for analysis and future reference.
    
    Args:
        state: Current curator state
        
    Returns:
        Updated state with storage confirmation
    """
    if not state.podcast_script:
        logger.warning("No podcast script available for storage")
        return state
    
    logger.info("Storing results and metadata")
    
    # Create comprehensive storage metadata
    storage_metadata = {
        'storage_timestamp': datetime.now().isoformat(),
        'workflow_completion_time': datetime.now().isoformat(),
        'total_processing_time': 'calculated_by_external_system',  # Will be calculated by caller
        'final_state_summary': state.get_processing_summary(),
        'script_metadata': {
            'word_count': len(state.podcast_script.split()),
            'estimated_duration': len(state.podcast_script.split()) / 155.0,  # 155 WPM average
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
    
    # Update state metadata with storage information
    state.update_generation_metadata(**storage_metadata)
    
    # In a real implementation, this would save to a database or file system
    # For now, we just log the successful storage
    logger.info(f"Results stored successfully: script with {storage_metadata['script_metadata']['word_count']} words, "
               f"{storage_metadata['script_metadata']['source_video_count']} source videos")
    
    # Log final processing summary
    summary = state.get_processing_summary()
    logger.info(f"Workflow complete - Final summary: {summary}")
    
    return state


@handle_node_error("refine_search")
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
    logger.info(f"Refining search parameters - attempt {state.search_attempt + 1}/{state.max_search_attempts}")
    
    config = get_config()
    refinement_engine = SearchRefinementEngine(config)
    
    # Perform iterative search refinement
    updated_state = refinement_engine.refine_search_iteratively(state)
    
    # Log refinement results
    refinement_summary = refinement_engine.get_refinement_summary(updated_state)
    logger.info(f"Search refinement complete: {refinement_summary['total_attempts']} attempts, "
               f"{refinement_summary['final_video_count']} videos found")
    
    if refinement_summary['quality_threshold_met']:
        logger.info("Quality threshold requirements met after refinement")
    else:
        logger.warning("Quality threshold requirements still not met after refinement")
    
    return updated_state


# Conditional routing functions
def should_refine_search(state: CuratorState) -> str:
    """
    Determine if search refinement is needed based on discovery results.
    
    Args:
        state: Current curator state
        
    Returns:
        Next node name ("refine_search" or "fetch_details")
    """
    # Check if we have any videos
    if not state.discovered_videos:
        if state.can_refine_search():
            logger.info("No videos found, proceeding to search refinement")
            return "refine_search"
        else:
            logger.warning("No videos found and max refinement attempts reached")
            return "fetch_details"  # Continue with empty results
    
    # Check if we have enough videos for quality assessment
    min_videos_for_processing = 3
    if len(state.discovered_videos) < min_videos_for_processing:
        if state.can_refine_search():
            logger.info(f"Only {len(state.discovered_videos)} videos found (need {min_videos_for_processing}), "
                       f"proceeding to search refinement")
            return "refine_search"
        else:
            logger.warning(f"Only {len(state.discovered_videos)} videos found but max refinement attempts reached")
    
    logger.info(f"Sufficient videos found ({len(state.discovered_videos)}), proceeding to processing")
    return "fetch_details"


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


def create_curator_workflow(config: Optional[Configuration] = None) -> CompiledGraph:
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
    
    # Create the state graph
    workflow = StateGraph(CuratorState)
    
    # Add all processing nodes
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
    
    # Add conditional routing after discovery
    workflow.add_conditional_edges(
        "discover_videos",
        should_refine_search,
        {
            "refine_search": "refine_search",
            "fetch_details": "fetch_details"
        }
    )
    
    # Refinement loop - goes back to discovery
    workflow.add_edge("refine_search", "discover_videos")
    
    # Parallel processing after successful discovery
    workflow.add_edge("fetch_details", "evaluate_quality")
    workflow.add_edge("fetch_transcripts", "evaluate_quality")
    
    # Start both parallel processes from discovery
    workflow.add_edge("discover_videos", "fetch_transcripts")
    
    # Sequential processing after parallel completion
    workflow.add_edge("evaluate_quality", "rank_videos")
    
    # Conditional routing after ranking (quality-based refinement)
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