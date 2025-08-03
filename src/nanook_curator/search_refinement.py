"""
Search parameter refinement system for the nanook-curator.

This module provides iterative search refinement capabilities including:
- Progressive search strategy with multiple refinement attempts
- Keyword expansion with synonyms and related terms
- Date range expansion logic for fallback searches
- Broader search terms for final attempts
- Integration with video ranking and quality assessment
"""

import logging
import re
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .models import VideoData, CuratorState
from .config import get_config, Configuration
from .youtube_client import YouTubeClient, SearchFilters
from .video_ranking_system import VideoRankingSystem, RankingResult

# Configure logging
logger = logging.getLogger(__name__)


class RefinementStrategy(Enum):
    """Available refinement strategies for different attempt levels."""
    INITIAL = "initial"          # Original search parameters
    EXPAND_KEYWORDS = "expand_keywords"    # Add synonyms and related terms
    EXPAND_TIMEFRAME = "expand_timeframe"  # Increase date range
    BROADEN_TERMS = "broaden_terms"       # Use broader, more general terms
    FINAL_FALLBACK = "final_fallback"     # Most permissive search


@dataclass
class RefinementResult:
    """Result of a search refinement attempt."""
    videos: List[VideoData]
    strategy_used: RefinementStrategy
    search_terms: List[str]
    days_back: int
    quality_summary: Dict[str, Any]
    refinement_needed: bool
    suggestions: List[str]
    metadata: Dict[str, Any]


class SearchRefinementEngine:
    """
    Comprehensive search refinement engine with progressive strategies.
    
    Implements iterative search parameter refinement with up to 3 attempts,
    using various strategies to improve search results quality and quantity.
    """
    
    def __init__(self, config: Optional[Configuration] = None):
        """
        Initialize the search refinement engine.
        
        Args:
            config: Optional configuration instance. If not provided, uses global config.
        """
        self.config = config or get_config()
        self.youtube_client = YouTubeClient(config=self.config)
        self.ranking_system = VideoRankingSystem(config=self.config)
        
        # Refinement configuration
        self.max_attempts = 3
        self.quality_threshold = 70.0
        self.min_quality_videos = 3
        
        # Keyword expansion mappings
        self.keyword_synonyms = self._build_keyword_synonyms()
        self.domain_expansions = self._build_domain_expansions()
        self.broader_terms = self._build_broader_terms()
    
    def _build_keyword_synonyms(self) -> Dict[str, List[str]]:
        """Build keyword synonym mappings for search expansion."""
        return {
            # AI/ML Terms
            'ai': ['artificial intelligence', 'machine intelligence', 'ai technology'],
            'artificial intelligence': ['ai', 'machine intelligence', 'ai tech', 'intelligent systems'],
            'machine learning': ['ml', 'ai learning', 'automated learning', 'predictive modeling'],
            'ml': ['machine learning', 'artificial intelligence', 'ai learning'],
            'deep learning': ['neural networks', 'deep neural networks', 'dl', 'deep ai'],
            'neural networks': ['deep learning', 'neural nets', 'artificial neural networks'],
            'llm': ['large language model', 'language models', 'ai models', 'chatbot'],
            'gpt': ['generative ai', 'language model', 'ai chatbot', 'openai'],
            'chatgpt': ['gpt', 'openai', 'ai assistant', 'conversational ai'],
            'claude': ['anthropic', 'ai assistant', 'claude ai', 'conversational ai'],
            'gemini': ['google ai', 'bard', 'google gemini', 'google assistant'],
            
            # Technology Terms
            'programming': ['coding', 'software development', 'development', 'programming tutorial'],
            'coding': ['programming', 'software development', 'code tutorial', 'development'],
            'python': ['python programming', 'python tutorial', 'python coding', 'python development'],
            'javascript': ['js', 'javascript programming', 'web development', 'js tutorial'],
            'react': ['reactjs', 'react.js', 'react development', 'frontend development'],
            'nodejs': ['node.js', 'node', 'backend development', 'javascript backend'],
            
            # Business/Tech Terms
            'startup': ['startups', 'entrepreneurship', 'business', 'tech startup'],
            'tech': ['technology', 'tech news', 'tech trends', 'innovation'],
            'innovation': ['tech innovation', 'new technology', 'breakthrough', 'advancement'],
            'automation': ['automated systems', 'process automation', 'ai automation'],
            'robotics': ['robots', 'robotic systems', 'automation', 'ai robotics'],
            
            # Data/Analytics Terms
            'data': ['data science', 'big data', 'data analysis', 'analytics'],
            'analytics': ['data analytics', 'business analytics', 'data analysis'],
            'blockchain': ['crypto', 'cryptocurrency', 'distributed systems', 'web3'],
            'crypto': ['cryptocurrency', 'blockchain', 'bitcoin', 'digital currency'],
            
            # General Tech Terms
            'software': ['software development', 'applications', 'programs', 'tech'],
            'app': ['application', 'mobile app', 'software', 'program'],
            'cloud': ['cloud computing', 'cloud services', 'aws', 'azure'],
            'api': ['apis', 'web api', 'programming interface', 'integration'],
            
            # News/Update Terms
            'news': ['updates', 'latest', 'breaking', 'announcement'],
            'update': ['news', 'latest', 'new', 'announcement'],
            'review': ['analysis', 'overview', 'evaluation', 'breakdown'],
            'tutorial': ['guide', 'how-to', 'walkthrough', 'lesson'],
            'explained': ['explanation', 'breakdown', 'analysis', 'guide']
        }
    
    def _build_domain_expansions(self) -> Dict[str, List[str]]:
        """Build domain-specific expansion terms."""
        return {
            'ai_ml': [
                'artificial intelligence', 'machine learning', 'deep learning', 
                'neural networks', 'ai technology', 'ml algorithms', 'ai research',
                'computer vision', 'natural language processing', 'nlp', 'transformers'
            ],
            'programming': [
                'programming', 'coding', 'software development', 'web development',
                'mobile development', 'app development', 'code tutorial', 'programming tutorial'
            ],
            'tech_business': [
                'technology', 'tech industry', 'startups', 'innovation', 'digital transformation',
                'tech trends', 'tech news', 'business technology', 'enterprise tech'
            ],
            'data_analytics': [
                'data science', 'big data', 'analytics', 'data analysis', 'business intelligence',
                'data visualization', 'statistics', 'data mining', 'predictive analytics'
            ],
            'emerging_tech': [
                'blockchain', 'cryptocurrency', 'quantum computing', 'iot', 'edge computing',
                'augmented reality', 'virtual reality', 'ar', 'vr', 'metaverse'
            ]
        }
    
    def _build_broader_terms(self) -> List[str]:
        """Build list of broader, more general search terms for final attempts."""
        return [
            'technology', 'tech', 'innovation', 'digital', 'computer', 'software',
            'internet', 'online', 'digital transformation', 'future tech',
            'tech trends', 'emerging technology', 'breakthrough', 'advancement',
            'science', 'research', 'development', 'engineering', 'systems'
        ]
    
    def expand_keywords(self, keywords: List[str], strategy: RefinementStrategy) -> List[str]:
        """
        Expand search keywords based on refinement strategy.
        
        Args:
            keywords: Original search keywords
            strategy: Refinement strategy to apply
            
        Returns:
            Expanded list of search keywords
        """
        if strategy == RefinementStrategy.INITIAL:
            return keywords.copy()
        
        expanded_keywords = set(keywords)  # Start with original keywords
        
        if strategy == RefinementStrategy.EXPAND_KEYWORDS:
            # Add synonyms and related terms for each keyword
            for keyword in keywords:
                keyword_lower = keyword.lower().strip()
                
                # Direct synonyms
                if keyword_lower in self.keyword_synonyms:
                    expanded_keywords.update(self.keyword_synonyms[keyword_lower])
                
                # Partial matches for compound terms
                for base_term, synonyms in self.keyword_synonyms.items():
                    if base_term in keyword_lower or keyword_lower in base_term:
                        expanded_keywords.update(synonyms[:2])  # Add top 2 synonyms
            
            logger.info(f"Expanded {len(keywords)} keywords to {len(expanded_keywords)} terms")
            
        elif strategy == RefinementStrategy.EXPAND_TIMEFRAME:
            # Keep original keywords but prepare for timeframe expansion
            expanded_keywords = set(keywords)
            
        elif strategy == RefinementStrategy.BROADEN_TERMS:
            # Add domain-specific broader terms
            expanded_keywords.update(keywords)
            
            # Identify dominant domain and add related broader terms
            domains_detected = []
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if any(ai_term in keyword_lower for ai_term in ['ai', 'artificial', 'machine', 'neural', 'deep']):
                    domains_detected.append('ai_ml')
                elif any(prog_term in keyword_lower for prog_term in ['code', 'program', 'develop', 'software']):
                    domains_detected.append('programming')
                elif any(data_term in keyword_lower for data_term in ['data', 'analytic', 'big data']):
                    domains_detected.append('data_analytics')
                elif any(biz_term in keyword_lower for biz_term in ['startup', 'business', 'enterprise']):
                    domains_detected.append('tech_business')
            
            # Add domain expansions
            for domain in set(domains_detected):
                if domain in self.domain_expansions:
                    expanded_keywords.update(self.domain_expansions[domain][:3])  # Top 3 domain terms
            
            logger.info(f"Added domain expansions for: {domains_detected}")
            
        elif strategy == RefinementStrategy.FINAL_FALLBACK:
            # Use broader, more general terms
            expanded_keywords.update(keywords)
            expanded_keywords.update(self.broader_terms[:5])  # Add top 5 broader terms
            
            logger.info(f"Applied final fallback strategy with broader terms")
        
        # Convert back to list and limit total keywords to prevent API issues
        result_keywords = list(expanded_keywords)[:15]  # Limit to 15 keywords max
        
        return result_keywords
    
    def calculate_expanded_timeframe(self, original_days: int, strategy: RefinementStrategy) -> int:
        """
        Calculate expanded timeframe based on refinement strategy.
        
        Args:
            original_days: Original days back for search
            strategy: Refinement strategy to apply
            
        Returns:
            Expanded days back value
        """
        if strategy == RefinementStrategy.INITIAL:
            return original_days
        elif strategy == RefinementStrategy.EXPAND_KEYWORDS:
            return original_days  # Keep same timeframe
        elif strategy == RefinementStrategy.EXPAND_TIMEFRAME:
            return min(original_days * 2, 30)  # Double timeframe, max 30 days
        elif strategy == RefinementStrategy.BROADEN_TERMS:
            return min(original_days * 2, 30)  # Keep expanded timeframe
        elif strategy == RefinementStrategy.FINAL_FALLBACK:
            return min(original_days * 3, 60)  # Triple timeframe, max 60 days
        
        return original_days
    
    def determine_refinement_strategy(self, attempt: int, previous_results: Optional[RankingResult] = None) -> RefinementStrategy:
        """
        Determine the appropriate refinement strategy for the current attempt.
        
        Args:
            attempt: Current search attempt number (0-based)
            previous_results: Results from previous attempt for analysis
            
        Returns:
            Appropriate refinement strategy
        """
        if attempt == 0:
            return RefinementStrategy.INITIAL
        elif attempt == 1:
            return RefinementStrategy.EXPAND_KEYWORDS
        elif attempt == 2:
            # Analyze previous results to decide between timeframe expansion or broader terms
            if previous_results and len(previous_results.ranked_videos) < 5:
                return RefinementStrategy.EXPAND_TIMEFRAME
            else:
                return RefinementStrategy.BROADEN_TERMS
        else:
            return RefinementStrategy.FINAL_FALLBACK
    
    def perform_search_attempt(self, state: CuratorState, strategy: RefinementStrategy) -> RefinementResult:
        """
        Perform a single search attempt with the specified strategy.
        
        Args:
            state: Current curator state
            strategy: Refinement strategy to apply
            
        Returns:
            Results of the search attempt
        """
        logger.info(f"Performing search attempt {state.search_attempt + 1} with strategy: {strategy.value}")
        
        # Determine search parameters based on strategy
        if strategy == RefinementStrategy.INITIAL:
            search_keywords = state.search_keywords.copy()
            days_back = state.days_back
        else:
            # Use current_search_terms if available, otherwise fall back to original
            base_keywords = state.current_search_terms if state.current_search_terms else state.search_keywords
            search_keywords = self.expand_keywords(base_keywords, strategy)
            days_back = self.calculate_expanded_timeframe(state.days_back, strategy)
        
        # Update state with current search terms
        state.current_search_terms = search_keywords
        
        try:
            # Create search filters
            filters = SearchFilters(
                keywords=search_keywords,
                days_back=days_back,
                max_results=min(state.max_videos * 2, 50),  # Search for more to improve filtering
                min_views=500 if strategy in [RefinementStrategy.BROADEN_TERMS, RefinementStrategy.FINAL_FALLBACK] else 1000,
                order="relevance"
            )
            
            # Perform video discovery
            discovered_videos = self.youtube_client.discover_videos(
                keywords=search_keywords,
                max_videos=filters.max_results,
                days_back=days_back
            )
            
            logger.info(f"Strategy {strategy.value}: Found {len(discovered_videos)} videos")
            
            # Rank videos using the ranking system
            ranking_result = self.ranking_system.rank_videos(
                discovered_videos,
                target_count=state.max_videos
            )
            
            # Determine if further refinement is needed
            refinement_needed = (
                not ranking_result.threshold_analysis['meets_minimum_requirement'] and
                state.search_attempt < self.max_attempts - 1
            )
            
            # Compile refinement result
            refinement_result = RefinementResult(
                videos=ranking_result.ranked_videos,
                strategy_used=strategy,
                search_terms=search_keywords,
                days_back=days_back,
                quality_summary=ranking_result.quality_summary,
                refinement_needed=refinement_needed,
                suggestions=ranking_result.refinement_suggestions,
                metadata={
                    'total_discovered': len(discovered_videos),
                    'total_ranked': len(ranking_result.ranked_videos),
                    'threshold_analysis': ranking_result.threshold_analysis,
                    'ranking_metadata': ranking_result.ranking_metadata,
                    'search_filters': {
                        'keywords': search_keywords,
                        'days_back': days_back,
                        'max_results': filters.max_results,
                        'min_views': filters.min_views
                    }
                }
            )
            
            return refinement_result
            
        except Exception as e:
            logger.error(f"Search attempt failed with strategy {strategy.value}: {e}")
            
            # Return empty result with error information
            return RefinementResult(
                videos=[],
                strategy_used=strategy,
                search_terms=search_keywords,
                days_back=days_back,
                quality_summary={'total_videos': 0, 'error': str(e)},
                refinement_needed=True,
                suggestions=[f"Search failed: {str(e)}. Try different keywords or check API configuration."],
                metadata={
                    'error': str(e),
                    'strategy_failed': strategy.value,
                    'search_terms_attempted': search_keywords
                }
            )
    
    def refine_search_iteratively(self, state: CuratorState) -> CuratorState:
        """
        Perform iterative search refinement with progressive strategies.
        
        This is the main method that implements the complete iterative refinement process
        with up to 3 attempts using different strategies.
        
        Args:
            state: Current curator state
            
        Returns:
            Updated curator state with refined search results
        """
        logger.info(f"Starting iterative search refinement for keywords: {state.search_keywords}")
        
        best_result = None
        all_attempts = []
        
        while state.search_attempt < self.max_attempts:
            # Determine strategy for this attempt
            strategy = self.determine_refinement_strategy(
                state.search_attempt, 
                best_result.metadata.get('ranking_result') if best_result else None
            )
            
            # Perform search attempt
            attempt_result = self.perform_search_attempt(state, strategy)
            all_attempts.append(attempt_result)
            
            # Update state with attempt results
            state.search_attempt += 1
            state.discovered_videos = attempt_result.videos
            
            # Check if this is the best result so far
            if (not best_result or 
                len(attempt_result.videos) > len(best_result.videos) or
                (len(attempt_result.videos) == len(best_result.videos) and 
                 attempt_result.quality_summary.get('average_score', 0) > best_result.quality_summary.get('average_score', 0))):
                best_result = attempt_result
                logger.info(f"New best result from attempt {state.search_attempt}: "
                          f"{len(attempt_result.videos)} videos, "
                          f"avg quality: {attempt_result.quality_summary.get('average_score', 0):.1f}")
            
            # Check if refinement is still needed and we haven't exhausted attempts
            if not attempt_result.refinement_needed:
                logger.info(f"Quality requirements met after {state.search_attempt} attempts")
                break
            
            if state.search_attempt >= self.max_attempts:
                logger.info(f"Maximum refinement attempts ({self.max_attempts}) reached")
                break
            
            logger.info(f"Refinement needed. Proceeding to attempt {state.search_attempt + 1}")
        
        # Use the best result found across all attempts
        if best_result:
            state.discovered_videos = best_result.videos
            state.current_search_terms = best_result.search_terms
            
            # Update metadata with refinement summary
            state.update_generation_metadata(
                refinement_complete=True,
                total_attempts=state.search_attempt,
                best_strategy=best_result.strategy_used.value,
                final_search_terms=best_result.search_terms,
                final_days_back=best_result.days_back,
                final_video_count=len(best_result.videos),
                quality_threshold_met=not best_result.refinement_needed,
                all_attempts_summary=[
                    {
                        'attempt': i + 1,
                        'strategy': attempt.strategy_used.value,
                        'videos_found': len(attempt.videos),
                        'avg_quality': attempt.quality_summary.get('average_score', 0)
                    }
                    for i, attempt in enumerate(all_attempts)
                ]
            )
            
            logger.info(f"Iterative refinement complete: {len(best_result.videos)} videos selected "
                       f"using {best_result.strategy_used.value} strategy")
        else:
            logger.warning("No successful search results obtained from any refinement attempt")
            state.add_error("All search refinement attempts failed", "search_refinement")
        
        return state
    
    def get_refinement_summary(self, state: CuratorState) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the refinement process.
        
        Args:
            state: Curator state after refinement
            
        Returns:
            Summary of refinement results and recommendations
        """
        metadata = state.generation_metadata
        
        summary = {
            'refinement_completed': metadata.get('refinement_complete', False),
            'total_attempts': metadata.get('total_attempts', 0),
            'successful_strategy': metadata.get('best_strategy', 'none'),
            'final_video_count': len(state.discovered_videos),
            'quality_threshold_met': len([v for v in state.discovered_videos 
                                        if hasattr(v, 'quality_score') and v.quality_score and v.quality_score >= self.quality_threshold]) >= self.min_quality_videos,
            'search_evolution': {
                'original_keywords': state.search_keywords,
                'final_keywords': metadata.get('final_search_terms', state.search_keywords),
                'original_timeframe': state.days_back,
                'final_timeframe': metadata.get('final_days_back', state.days_back)
            },
            'attempts_breakdown': metadata.get('all_attempts_summary', []),
            'recommendations': self._generate_refinement_recommendations(state)
        }
        
        return summary
    
    def _generate_refinement_recommendations(self, state: CuratorState) -> List[str]:
        """Generate recommendations based on refinement results."""
        recommendations = []
        
        if len(state.discovered_videos) == 0:
            recommendations.extend([
                "No videos found across all refinement attempts. Consider:",
                "- Checking API configuration and quota limits",
                "- Using more general or popular search terms",
                "- Increasing the maximum search timeframe",
                "- Reviewing keyword spelling and relevance"
            ])
        elif len(state.discovered_videos) < state.min_quality_videos:
            recommendations.extend([
                f"Found {len(state.discovered_videos)} videos but need {state.min_quality_videos} for quality threshold.",
                "Consider:",
                "- Lowering quality threshold temporarily",
                "- Expanding to more diverse search terms",
                "- Increasing maximum video count per search"
            ])
        else:
            recommendations.append("Search refinement successful! Quality requirements met.")
            
            # Add optimization suggestions
            if state.search_attempt > 1:
                recommendations.append(f"Required {state.search_attempt} attempts to find quality content. "
                                     "Consider using the successful keywords as defaults for similar topics.")
        
        return recommendations


def refine_search_node(state: CuratorState) -> CuratorState:
    """
    LangGraph node function for iterative search refinement.
    
    This function is designed to be used as a node in the LangGraph workflow
    and performs comprehensive search refinement with progressive strategies.
    
    Args:
        state: Current curator state
        
    Returns:
        Updated state with refined search results
    """
    logger.info("Starting search refinement node")
    
    try:
        refinement_engine = SearchRefinementEngine()
        updated_state = refinement_engine.refine_search_iteratively(state)
        
        # Log refinement summary
        summary = refinement_engine.get_refinement_summary(updated_state)
        logger.info(f"Search refinement complete: {summary['total_attempts']} attempts, "
                   f"{summary['final_video_count']} videos found using '{summary['successful_strategy']}' strategy")
        
        if summary['quality_threshold_met']:
            logger.info("Quality threshold requirements met")
        else:
            logger.warning("Quality threshold requirements NOT met despite refinement")
        
        return updated_state
        
    except Exception as e:
        error_msg = f"Critical error in search refinement node: {e}"
        logger.error(error_msg)
        state.add_error(error_msg, "refine_search_node")
        return state