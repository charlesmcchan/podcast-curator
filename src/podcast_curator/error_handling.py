"""
Comprehensive error handling and recovery mechanisms for the podcast-curator system.

This module provides decorators, retry mechanisms, and error recovery strategies
to ensure robust operation with graceful degradation when individual components fail.
"""

import logging
import time
import random
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
from datetime import datetime

from .models import CuratorState, VideoData

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry mechanisms."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


class ErrorContext:
    """Context information for error handling."""
    
    def __init__(
        self,
        node_name: str,
        operation: str,
        video_id: Optional[str] = None,
        attempt: int = 1,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        self.node_name = node_name
        self.operation = operation
        self.video_id = video_id
        self.attempt = attempt
        self.additional_context = additional_context or {}
        self.timestamp = datetime.now().isoformat()


def calculate_retry_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay for retry attempt using exponential backoff with jitter.
    
    Args:
        attempt: Current attempt number (1-based)
        config: Retry configuration
        
    Returns:
        Delay in seconds
    """
    if attempt <= 1:
        return 0
    
    # Calculate exponential backoff delay
    delay = config.base_delay * (config.exponential_base ** (attempt - 2))
    delay = min(delay, config.max_delay)
    
    # Add jitter to prevent thundering herd
    if config.jitter:
        jitter_range = delay * 0.1  # 10% jitter
        delay += random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)


def retry_with_backoff(
    config: RetryConfig,
    exceptions: tuple = (Exception,),
    context: Optional[ErrorContext] = None
):
    """
    Decorator for retrying operations with exponential backoff.
    
    Args:
        config: Retry configuration
        exceptions: Tuple of exceptions to retry on
        context: Error context for logging
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    if attempt > 1:
                        delay = calculate_retry_delay(attempt, config)
                        if delay > 0:
                            logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt}/{config.max_attempts})")
                            time.sleep(delay)
                    
                    result = func(*args, **kwargs)
                    
                    if attempt > 1:
                        logger.info(f"Retry successful for {func.__name__} on attempt {attempt}")
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    error_msg = f"Attempt {attempt}/{config.max_attempts} failed for {func.__name__}: {str(e)}"
                    
                    if context:
                        error_msg = f"[{context.node_name}:{context.operation}] {error_msg}"
                        if context.video_id:
                            error_msg += f" (video: {context.video_id})"
                    
                    if attempt < config.max_attempts:
                        logger.warning(error_msg)
                    else:
                        logger.error(f"All retry attempts exhausted: {error_msg}")
            
            # All attempts failed, raise the last exception
            raise last_exception
            
        return wrapper
    return decorator


def handle_node_error_with_recovery(
    node_name: str,
    allow_partial_failure: bool = True,
    critical_failure_threshold: float = 0.5
):
    """
    Enhanced decorator for graceful error handling in LangGraph nodes with recovery strategies.
    
    Args:
        node_name: Name of the node for error context
        allow_partial_failure: Whether to continue with partial results
        critical_failure_threshold: Threshold for critical failure (0.0-1.0)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: CuratorState) -> CuratorState:
            start_time = datetime.now()
            
            try:
                logger.info(f"Starting {node_name} node with error recovery")
                result = func(state)
                
                # Log successful completion
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Completed {node_name} node successfully in {duration:.2f}s")
                
                # Add success metadata
                result.update_generation_metadata(**{
                    f"{node_name}_success": True,
                    f"{node_name}_duration": duration,
                    f"{node_name}_completion_time": datetime.now().isoformat()
                })
                
                return result
                
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                error_msg = f"{node_name} node failed after {duration:.2f}s: {str(e)}"
                
                logger.error(error_msg, exc_info=True)
                
                # Add detailed error context to state
                state.add_error(
                    error=error_msg,
                    node_name=node_name
                )
                
                # Add failure metadata
                state.update_generation_metadata(**{
                    f"{node_name}_success": False,
                    f"{node_name}_duration": duration,
                    f"{node_name}_error": str(e),
                    f"{node_name}_error_type": type(e).__name__,
                    f"{node_name}_failure_time": datetime.now().isoformat()
                })
                
                # Implement recovery strategies based on node type
                if allow_partial_failure:
                    logger.info(f"Attempting recovery for {node_name} node")
                    state = _attempt_node_recovery(state, node_name, e)
                
                return state
                
        return wrapper
    return decorator


def _attempt_node_recovery(state: CuratorState, node_name: str, error: Exception) -> CuratorState:
    """
    Attempt recovery strategies based on the failed node.
    
    Args:
        state: Current curator state
        node_name: Name of the failed node
        error: The exception that occurred
        
    Returns:
        State with recovery attempts applied
    """
    recovery_msg = f"Applying recovery strategy for {node_name}"
    logger.info(recovery_msg)
    
    if node_name == "discover_videos":
        # Recovery: Ensure we have some basic state even if discovery fails
        if not state.discovered_videos:
            state.add_error("Video discovery failed completely, no videos available", node_name)
        
    elif node_name == "fetch_video_details":
        # Recovery: Continue with videos that have basic info
        available_videos = [v for v in state.discovered_videos if v.video_id]
        if available_videos:
            logger.info(f"Continuing with {len(available_videos)} videos despite details fetch failure")
            state.discovered_videos = available_videos
        
    elif node_name == "fetch_transcripts":
        # Recovery: Continue with videos that have transcripts, log missing ones
        videos_with_transcripts = [v for v in state.discovered_videos if v.transcript]
        videos_without_transcripts = [v for v in state.discovered_videos if not v.transcript]
        
        if videos_with_transcripts:
            logger.info(f"Continuing with {len(videos_with_transcripts)} videos with transcripts, "
                       f"{len(videos_without_transcripts)} without transcripts")
            state.add_error(f"Transcripts unavailable for {len(videos_without_transcripts)} videos", node_name)
        else:
            state.add_error("No transcripts available for any videos", node_name)
    
    elif node_name == "evaluate_quality":
        # Recovery: Assign default quality scores to videos without evaluation
        for video in state.discovered_videos:
            if video.quality_score is None:
                # Assign minimal quality score based on basic metrics
                basic_score = min(50.0, video.view_count / 1000)  # Basic scoring fallback
                video.quality_score = basic_score
                logger.debug(f"Assigned fallback quality score {basic_score} to video {video.video_id}")
        
        state.processed_videos = state.discovered_videos
        state.add_error("Quality evaluation failed, using fallback scoring", node_name)
    
    elif node_name == "rank_videos":
        # Recovery: Use simple view count ranking if quality ranking fails
        if state.processed_videos:
            sorted_videos = sorted(state.processed_videos, key=lambda v: v.view_count, reverse=True)
            state.ranked_videos = sorted_videos[:5]  # Top 5 by view count
            state.add_error("Video ranking failed, using view count fallback", node_name)
    
    elif node_name == "generate_script":
        # Recovery: Create minimal script from available video titles and descriptions
        if state.ranked_videos:
            fallback_script = _create_fallback_script(state.ranked_videos)
            state.podcast_script = fallback_script
            state.add_error("Script generation failed, created fallback script", node_name)
    
    # Add recovery metadata
    state.update_generation_metadata(**{
        f"{node_name}_recovery_attempted": True,
        f"{node_name}_recovery_time": datetime.now().isoformat()
    })
    
    return state


def _create_fallback_script(videos: List[VideoData]) -> str:
    """
    Create a basic fallback script from video metadata when full generation fails.
    
    Args:
        videos: List of videos to create script from
        
    Returns:
        Basic script text
    """
    script_parts = [
        "# AI News Weekly - Fallback Edition",
        "",
        "Welcome to this week's AI news roundup. Due to processing limitations, "
        "this is a simplified version based on the most trending videos we discovered.",
        ""
    ]
    
    for i, video in enumerate(videos[:3], 1):
        script_parts.extend([
            f"## Story {i}: {video.title}",
            f"From {video.channel}",
            f"This video has received {video.view_count:,} views and {video.like_count:,} likes.",
            ""
        ])
        
        if video.transcript and len(video.transcript) > 100:
            # Extract first few sentences as summary
            sentences = video.transcript[:500].split('.')[:3]
            summary = '. '.join(sentences) + '.'
            script_parts.extend([
                f"Key points: {summary}",
                ""
            ])
    
    script_parts.extend([
        "That's all for this week's AI news. This was a simplified edition due to processing constraints.",
        "For full analysis and insights, please check back next week."
    ])
    
    return '\n'.join(script_parts)


def handle_api_errors(
    operation: str,
    video_id: Optional[str] = None,
    retry_config: Optional[RetryConfig] = None
):
    """
    Decorator for handling API-specific errors with appropriate retry strategies.
    
    Args:
        operation: Description of the API operation
        video_id: Optional video ID for context
        retry_config: Optional retry configuration
        
    Returns:
        Decorator function
    """
    if retry_config is None:
        retry_config = RetryConfig(max_attempts=3, base_delay=1.0, max_delay=30.0)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                node_name="api_call",
                operation=operation,
                video_id=video_id
            )
            
            # Define API-specific exceptions to retry
            api_exceptions = (
                ConnectionError,
                TimeoutError,
                # Add specific API exceptions here
            )
            
            @retry_with_backoff(retry_config, api_exceptions, context)
            def api_call():
                return func(*args, **kwargs)
            
            try:
                return api_call()
            except Exception as e:
                # Log API failure with context
                error_msg = f"API operation '{operation}' failed"
                if video_id:
                    error_msg += f" for video {video_id}"
                error_msg += f": {str(e)}"
                
                logger.error(error_msg)
                
                # Return None or appropriate fallback value
                return None
                
        return wrapper
    return decorator


def handle_transcript_errors(func: Callable) -> Callable:
    """
    Specialized decorator for handling transcript fetching errors.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with transcript error handling
    """
    @wraps(func)
    def wrapper(video_id: str, *args, **kwargs):
        try:
            return func(video_id, *args, **kwargs)
        except Exception as e:
            # Log transcript error but don't fail the entire process
            logger.warning(f"Transcript fetch failed for video {video_id}: {str(e)}")
            return None
    
    return wrapper


def validate_processing_state(state: CuratorState, node_name: str) -> bool:
    """
    Validate that the state is in a valid condition for processing.
    
    Args:
        state: Current curator state
        node_name: Name of the current node
        
    Returns:
        True if state is valid for processing
    """
    validation_errors = []
    
    # Check basic state validity
    if not state.search_keywords:
        validation_errors.append("No search keywords provided")
    
    # Node-specific validations
    if node_name in ["fetch_video_details", "fetch_transcripts", "evaluate_quality"]:
        if not state.discovered_videos:
            validation_errors.append("No discovered videos available for processing")
    
    elif node_name == "rank_videos":
        if not state.processed_videos:
            validation_errors.append("No processed videos available for ranking")
    
    elif node_name == "generate_script":
        if not state.ranked_videos:
            validation_errors.append("No ranked videos available for script generation")
    
    # Log validation errors
    if validation_errors:
        for error in validation_errors:
            state.add_error(f"State validation failed: {error}", node_name)
            logger.error(f"State validation error in {node_name}: {error}")
        return False
    
    return True


def log_processing_metrics(state: CuratorState, node_name: str) -> None:
    """
    Log processing metrics for monitoring and debugging.
    
    Args:
        state: Current curator state
        node_name: Name of the current node
    """
    metrics = {
        'node': node_name,
        'timestamp': datetime.now().isoformat(),
        'discovered_videos': len(state.discovered_videos),
        'processed_videos': len(state.processed_videos),
        'ranked_videos': len(state.ranked_videos),
        'error_count': len(state.errors),
        'search_attempt': state.search_attempt,
        'has_script': state.podcast_script is not None
    }
    
    # Add node-specific metrics
    if node_name == "fetch_transcripts":
        videos_with_transcripts = len([v for v in state.discovered_videos if v.transcript])
        metrics['videos_with_transcripts'] = videos_with_transcripts
        metrics['transcript_success_rate'] = videos_with_transcripts / len(state.discovered_videos) if state.discovered_videos else 0
    
    elif node_name == "evaluate_quality":
        videos_with_scores = len([v for v in state.processed_videos if v.quality_score is not None])
        metrics['videos_with_quality_scores'] = videos_with_scores
        if videos_with_scores > 0:
            avg_score = sum(v.quality_score for v in state.processed_videos if v.quality_score) / videos_with_scores
            metrics['average_quality_score'] = avg_score
    
    logger.info(f"Processing metrics: {metrics}")