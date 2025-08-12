"""
Content quality scoring system for the podcast-curator.

This module provides comprehensive content quality analysis including:
- Enhanced transcript coherence evaluation with advanced algorithms
- Information density measurement with semantic analysis
- Technical accuracy indicators assessment
- Combined content and engagement scoring for final quality evaluation
"""

import logging
import re
import math
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import Counter

from .models import VideoData
from .config import get_config, Configuration
from .transcript_processor import TranscriptProcessor
from .engagement_analyzer import EngagementAnalyzer, EngagementMetrics

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ContentQualityMetrics:
    """Container for comprehensive content quality metrics."""
    coherence_score: float
    information_density: float
    technical_accuracy: float
    clarity_score: float
    depth_score: float
    structure_score: float
    overall_content_score: float
    engagement_integration_score: float
    final_quality_score: float
    detailed_metrics: Dict[str, Any]


class ContentQualityScorer:
    """
    Comprehensive content quality scoring system.
    
    Provides advanced analysis capabilities for transcript quality including
    coherence, information density, technical accuracy, and integration with
    engagement metrics for holistic quality assessment.
    """
    
    def __init__(self, config: Optional[Configuration] = None):
        """
        Initialize the content quality scorer.
        
        Args:
            config: Optional configuration instance. If not provided, uses global config.
        """
        self.config = config or get_config()
        self.transcript_processor = TranscriptProcessor(config=self.config)
        self.engagement_analyzer = EngagementAnalyzer(config=self.config)
        
        # Advanced coherence analysis parameters
        self.coherence_weights = {
            'transitions': 0.25,
            'sentence_flow': 0.20,
            'topic_consistency': 0.25,
            'structural_patterns': 0.15,
            'lexical_cohesion': 0.15
        }
        
        # Information density parameters
        self.density_weights = {
            'conceptual_diversity': 0.30,
            'technical_terminology': 0.25,
            'semantic_richness': 0.25,
            'information_novelty': 0.20
        }
        
        # Technical accuracy parameters
        self.accuracy_weights = {
            'citation_quality': 0.25,
            'source_credibility': 0.20,
            'precision_indicators': 0.20,
            'uncertainty_handling': 0.15,
            'context_provision': 0.20
        }
    
    def evaluate_transcript_coherence(self, transcript: str) -> Dict[str, float]:
        """
        Enhanced transcript coherence evaluation using advanced algorithms.
        
        Args:
            transcript: Cleaned transcript text
            
        Returns:
            Dictionary with detailed coherence metrics
        """
        if not transcript:
            return {
                'transitions_score': 0.0,
                'sentence_flow_score': 0.0,
                'topic_consistency_score': 0.0,
                'structural_patterns_score': 0.0,
                'lexical_cohesion_score': 0.0,
                'overall_coherence': 0.0
            }
        
        # Split into sentences and clean
        sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
        
        if len(sentences) < 2:
            return {
                'transitions_score': 50.0,
                'sentence_flow_score': 50.0,
                'topic_consistency_score': 50.0,
                'structural_patterns_score': 50.0,
                'lexical_cohesion_score': 50.0,
                'overall_coherence': 50.0
            }
        
        # 1. Enhanced transitions analysis
        transitions_score = self._analyze_transitions(transcript, sentences)
        
        # 2. Sentence flow analysis
        sentence_flow_score = self._analyze_sentence_flow(sentences)
        
        # 3. Topic consistency analysis
        topic_consistency_score = self._analyze_topic_consistency(sentences)
        
        # 4. Structural patterns analysis
        structural_patterns_score = self._analyze_structural_patterns(transcript, sentences)
        
        # 5. Lexical cohesion analysis
        lexical_cohesion_score = self._analyze_lexical_cohesion(sentences)
        
        # Calculate weighted overall coherence
        overall_coherence = (
            transitions_score * self.coherence_weights['transitions'] +
            sentence_flow_score * self.coherence_weights['sentence_flow'] +
            topic_consistency_score * self.coherence_weights['topic_consistency'] +
            structural_patterns_score * self.coherence_weights['structural_patterns'] +
            lexical_cohesion_score * self.coherence_weights['lexical_cohesion']
        )
        
        return {
            'transitions_score': transitions_score,
            'sentence_flow_score': sentence_flow_score,
            'topic_consistency_score': topic_consistency_score,
            'structural_patterns_score': structural_patterns_score,
            'lexical_cohesion_score': lexical_cohesion_score,
            'overall_coherence': min(100.0, max(0.0, overall_coherence))
        }
    
    def _analyze_transitions(self, transcript: str, sentences: List[str]) -> float:
        """Analyze transition quality and logical flow."""
        # Enhanced transition words categorized by function
        temporal_transitions = {
            'first', 'initially', 'then', 'next', 'after', 'before', 'during', 
            'meanwhile', 'subsequently', 'finally', 'eventually', 'previously'
        }
        
        logical_transitions = {
            'therefore', 'thus', 'consequently', 'as a result', 'hence', 
            'because', 'since', 'due to', 'owing to', 'so that'
        }
        
        contrast_transitions = {
            'however', 'nevertheless', 'nonetheless', 'on the other hand',
            'in contrast', 'conversely', 'whereas', 'while', 'although', 'despite'
        }
        
        additive_transitions = {
            'furthermore', 'moreover', 'additionally', 'besides', 'also',
            'in addition', 'what\'s more', 'similarly', 'likewise'
        }
        
        text_lower = transcript.lower()
        
        # Count different types of transitions
        temporal_count = sum(1 for word in temporal_transitions if word in text_lower)
        logical_count = sum(1 for word in logical_transitions if word in text_lower)
        contrast_count = sum(1 for word in contrast_transitions if word in text_lower)
        additive_count = sum(1 for word in additive_transitions if word in text_lower)
        
        total_transitions = temporal_count + logical_count + contrast_count + additive_count
        
        # Calculate transition diversity (using all 4 types)
        types_used = sum([
            temporal_count > 0, logical_count > 0, 
            contrast_count > 0, additive_count > 0
        ])
        
        diversity_bonus = (types_used / 4) * 20  # Up to 20 points for diversity
        
        # Base score from transition frequency
        transition_frequency = (total_transitions / len(sentences)) * 100
        frequency_score = min(transition_frequency, 60)  # Cap at 60 points
        
        return min(100.0, frequency_score + diversity_bonus)
    
    def _analyze_sentence_flow(self, sentences: List[str]) -> float:
        """Analyze sentence length variation and readability flow."""
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        
        if not sentence_lengths:
            return 0.0
        
        # Calculate length statistics
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        length_variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        length_std = math.sqrt(length_variance)
        
        # Optimal average sentence length (12-18 words)
        length_score = 100 - abs(avg_length - 15) * 3
        length_score = max(0, min(100, length_score))
        
        # Variation score (some variation is good, too much is bad)
        variation_score = 100 - min(length_std * 2, 50)  # Penalize high variation
        variation_score = max(50, variation_score)  # Minimum 50 points
        
        # Rhythm analysis (check for patterns)
        rhythm_score = self._analyze_sentence_rhythm(sentence_lengths)
        
        return (length_score * 0.4 + variation_score * 0.3 + rhythm_score * 0.3)
    
    def _analyze_sentence_rhythm(self, lengths: List[int]) -> float:
        """Analyze rhythmic patterns in sentence lengths."""
        if len(lengths) < 3:
            return 70.0
        
        # Look for alternating patterns (long-short-long, etc.)
        pattern_score = 0
        
        # Check for good variation without chaos
        consecutive_similar = 0
        for i in range(1, len(lengths)):
            diff = abs(lengths[i] - lengths[i-1])
            if diff < 3:  # Very similar lengths
                consecutive_similar += 1
            else:
                consecutive_similar = 0
            
            # Penalize too many consecutive similar lengths
            if consecutive_similar > 2:
                pattern_score -= 10
        
        # Reward some variation
        total_variation = sum(abs(lengths[i] - lengths[i-1]) for i in range(1, len(lengths)))
        avg_variation = total_variation / (len(lengths) - 1)
        
        # Optimal variation is around 4-8 words difference
        variation_score = 100 - abs(avg_variation - 6) * 5
        pattern_score += max(0, min(100, variation_score))
        
        return max(0, min(100, pattern_score))
    
    def _analyze_topic_consistency(self, sentences: List[str]) -> float:
        """Analyze topic consistency across sentences using keyword overlap."""
        if len(sentences) < 2:
            return 70.0
        
        # Extract meaningful keywords from each sentence
        sentence_keywords = []
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
            'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that', 'with',
            'have', 'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been'
        }
        
        for sentence in sentences:
            words = [w.lower() for w in re.findall(r'\b\w+\b', sentence)]
            keywords = [w for w in words if w not in stop_words and len(w) > 3]
            sentence_keywords.append(set(keywords))
        
        # Calculate topic consistency using keyword overlap
        consistency_scores = []
        
        for i in range(len(sentence_keywords) - 1):
            current_keywords = sentence_keywords[i]
            next_keywords = sentence_keywords[i + 1]
            
            if not current_keywords or not next_keywords:
                consistency_scores.append(0.5)
                continue
            
            # Calculate Jaccard similarity
            overlap = len(current_keywords.intersection(next_keywords))
            union = len(current_keywords.union(next_keywords))
            
            if union == 0:
                consistency_scores.append(0.0)
            else:
                jaccard = overlap / union
                consistency_scores.append(jaccard)
        
        # Average consistency across all adjacent pairs
        if consistency_scores:
            avg_consistency = sum(consistency_scores) / len(consistency_scores)
            return min(100.0, avg_consistency * 150)  # Scale to 0-100
        
        return 50.0
    
    def _analyze_structural_patterns(self, transcript: str, sentences: List[str]) -> float:
        """Analyze structural patterns like lists, examples, explanations."""
        text_lower = transcript.lower()
        
        # Look for structural indicators
        list_indicators = ['first', 'second', 'third', 'lastly', 'finally', 
                          'next', 'then', 'also', '1.', '2.', '3.']
        example_indicators = ['for example', 'for instance', 'such as', 'like', 
                            'including', 'namely', 'specifically']
        explanation_indicators = ['because', 'since', 'due to', 'as a result', 
                                'therefore', 'this means', 'in other words']
        
        # Count structural elements
        list_count = sum(1 for indicator in list_indicators if indicator in text_lower)
        example_count = sum(1 for indicator in example_indicators if indicator in text_lower)
        explanation_count = sum(1 for indicator in explanation_indicators if indicator in text_lower)
        
        # Calculate structural richness
        total_structures = list_count + example_count + explanation_count
        structure_density = (total_structures / len(sentences)) * 100
        
        # Look for paragraph-like structures (longer sentences followed by shorter ones)
        structure_patterns = self._detect_paragraph_patterns(sentences)
        
        return min(100.0, structure_density + structure_patterns)
    
    def _detect_paragraph_patterns(self, sentences: List[str]) -> float:
        """Detect paragraph-like structural patterns."""
        if len(sentences) < 3:
            return 30.0
        
        lengths = [len(sentence.split()) for sentence in sentences]
        pattern_score = 0
        
        # Look for introduction-body-conclusion patterns
        if len(lengths) >= 3:
            # Check if first and last sentences are shorter (intro/conclusion style)
            if lengths[0] < sum(lengths[1:-1]) / len(lengths[1:-1]):
                pattern_score += 15
            if lengths[-1] < sum(lengths[1:-1]) / len(lengths[1:-1]):
                pattern_score += 15
        
        # Check for topic sentence patterns (longer sentences followed by shorter ones)
        for i in range(len(lengths) - 2):
            if lengths[i] > lengths[i+1] and lengths[i] > lengths[i+2]:
                pattern_score += 5
        
        return min(50.0, pattern_score)
    
    def _analyze_lexical_cohesion(self, sentences: List[str]) -> float:
        """Analyze lexical cohesion through word repetition and semantic relationships."""
        all_words = []
        for sentence in sentences:
            words = [w.lower() for w in re.findall(r'\b\w+\b', sentence) if len(w) > 3]
            all_words.extend(words)
        
        if len(all_words) < 10:
            return 50.0
        
        # Calculate word frequency distribution
        word_counts = Counter(all_words)
        
        # Find repeated words (good for cohesion, but not too much repetition)
        repeated_words = {word: count for word, count in word_counts.items() if count > 1}
        
        # Calculate cohesion metrics
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        # Lexical diversity (higher is generally better for quality)
        lexical_diversity = unique_words / total_words
        
        # Repetition ratio (some repetition is good for cohesion)
        repetition_ratio = len(repeated_words) / unique_words if unique_words > 0 else 0
        
        # Optimal repetition is around 0.2-0.4 (20-40% of unique words repeated)
        repetition_score = 100 - abs(repetition_ratio - 0.3) * 200
        repetition_score = max(0, min(100, repetition_score))
        
        # Combine diversity and repetition
        cohesion_score = (lexical_diversity * 60 + repetition_score * 0.4)
        
        return min(100.0, max(0.0, cohesion_score))
    
    def measure_information_density(self, transcript: str) -> Dict[str, float]:
        """
        Enhanced information density measurement with semantic analysis.
        
        Args:
            transcript: Cleaned transcript text
            
        Returns:
            Dictionary with detailed information density metrics
        """
        if not transcript:
            return {
                'conceptual_diversity': 0.0,
                'technical_terminology': 0.0,
                'semantic_richness': 0.0,
                'information_novelty': 0.0,
                'overall_density': 0.0
            }
        
        # 1. Conceptual diversity analysis
        conceptual_diversity = self._analyze_conceptual_diversity(transcript)
        
        # 2. Technical terminology density
        technical_terminology = self._analyze_technical_terminology(transcript)
        
        # 3. Semantic richness analysis
        semantic_richness = self._analyze_semantic_richness(transcript)
        
        # 4. Information novelty assessment
        information_novelty = self._analyze_information_novelty(transcript)
        
        # Calculate weighted overall density
        overall_density = (
            conceptual_diversity * self.density_weights['conceptual_diversity'] +
            technical_terminology * self.density_weights['technical_terminology'] +
            semantic_richness * self.density_weights['semantic_richness'] +
            information_novelty * self.density_weights['information_novelty']
        )
        
        return {
            'conceptual_diversity': conceptual_diversity,
            'technical_terminology': technical_terminology,
            'semantic_richness': semantic_richness,
            'information_novelty': information_novelty,
            'overall_density': min(100.0, max(0.0, overall_density))
        }
    
    def _analyze_conceptual_diversity(self, transcript: str) -> float:
        """Analyze diversity of concepts and topics covered."""
        # Define concept categories
        concept_categories = {
            'technology': {'algorithm', 'model', 'system', 'framework', 'architecture', 
                          'implementation', 'optimization', 'performance', 'scalability'},
            'science': {'research', 'study', 'experiment', 'hypothesis', 'theory', 
                       'evidence', 'analysis', 'methodology', 'findings', 'results'},
            'mathematics': {'equation', 'function', 'variable', 'parameter', 'optimization',
                           'statistics', 'probability', 'distribution', 'calculation'},
            'business': {'strategy', 'market', 'product', 'customer', 'revenue', 
                        'investment', 'competition', 'growth', 'innovation'},
            'education': {'learning', 'teaching', 'training', 'education', 'knowledge',
                         'skill', 'understanding', 'explanation', 'tutorial'}
        }
        
        text_lower = transcript.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        # Count concepts in each category
        category_counts = {}
        for category, concept_words in concept_categories.items():
            count = len(words.intersection(concept_words))
            if count > 0:
                category_counts[category] = count
        
        # Calculate diversity metrics
        categories_covered = len(category_counts)
        total_concepts = sum(category_counts.values())
        
        # Diversity score based on number of categories and concept distribution
        category_diversity = (categories_covered / len(concept_categories)) * 60
        concept_density = min((total_concepts / len(words.union(set()))) * 1000, 40)
        
        return min(100.0, category_diversity + concept_density)
    
    def _analyze_technical_terminology(self, transcript: str) -> float:
        """Analyze density and appropriateness of technical terminology."""
        # Expanded technical terms by domain
        technical_domains = {
            'ai_ml': {
                'neural', 'network', 'deep', 'learning', 'artificial', 'intelligence',
                'machine', 'algorithm', 'model', 'training', 'inference', 'prediction',
                'classification', 'regression', 'clustering', 'supervised', 'unsupervised',
                'reinforcement', 'transformer', 'attention', 'embedding', 'tokenization'
            },
            'programming': {
                'python', 'javascript', 'api', 'framework', 'library', 'function',
                'variable', 'parameter', 'object', 'class', 'method', 'database',
                'server', 'client', 'backend', 'frontend', 'deployment', 'git'
            },
            'data_science': {
                'data', 'analysis', 'visualization', 'statistics', 'correlation',
                'regression', 'hypothesis', 'pandas', 'numpy', 'matplotlib', 'seaborn',
                'jupyter', 'notebook', 'preprocessing', 'feature', 'engineering'
            },
            'advanced_tech': {
                'blockchain', 'cryptocurrency', 'quantum', 'cloud', 'distributed',
                'microservices', 'containerization', 'kubernetes', 'docker', 'devops',
                'cybersecurity', 'encryption', 'authentication', 'authorization'
            }
        }
        
        text_lower = transcript.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Count technical terms by domain
        domain_counts = {}
        total_technical = 0
        
        for domain, terms in technical_domains.items():
            count = sum(1 for word in words if word in terms)
            if count > 0:
                domain_counts[domain] = count
                total_technical += count
        
        # Calculate technical density
        technical_density = (total_technical / total_words) * 100
        
        # Domain diversity bonus
        domains_used = len(domain_counts)
        diversity_bonus = (domains_used / len(technical_domains)) * 20
        
        # Balance check - too many technical terms can hurt readability
        if technical_density > 15:  # More than 15% technical terms
            density_score = 100 - (technical_density - 15) * 2
        else:
            density_score = technical_density * 5  # Up to 75 points for optimal density
        
        return min(100.0, max(0.0, density_score + diversity_bonus))
    
    def _analyze_semantic_richness(self, transcript: str) -> float:
        """Analyze semantic richness through vocabulary sophistication."""
        words = re.findall(r'\b\w+\b', transcript.lower())
        
        if len(words) < 10:
            return 30.0
        
        # Analyze word length distribution (longer words often more sophisticated)
        word_lengths = [len(word) for word in words if len(word) > 2]
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        
        # Sophisticated vocabulary indicators
        sophisticated_words = {
            'sophisticated', 'comprehensive', 'fundamental', 'substantial', 'significant',
            'contemporary', 'methodology', 'implementation', 'optimization', 'systematic',
            'theoretical', 'practical', 'analytical', 'innovative', 'revolutionary',
            'paradigm', 'framework', 'architecture', 'infrastructure', 'scalability',
            'efficiency', 'effectiveness', 'reliability', 'sustainability', 'versatility'
        }
        
        # Count sophisticated words
        sophisticated_count = sum(1 for word in words if word in sophisticated_words)
        sophisticated_density = (sophisticated_count / len(words)) * 100
        
        # Vocabulary diversity (type-token ratio)
        unique_words = len(set(words))
        vocabulary_diversity = (unique_words / len(words)) * 100
        
        # Word length score (optimal around 5-6 characters)
        length_score = 100 - abs(avg_word_length - 5.5) * 10
        length_score = max(0, min(100, length_score))
        
        # Combine metrics
        richness_score = (
            sophisticated_density * 0.4 +
            vocabulary_diversity * 0.4 +
            length_score * 0.2
        )
        
        return min(100.0, max(0.0, richness_score))
    
    def _analyze_information_novelty(self, transcript: str) -> float:
        """Analyze information novelty and uniqueness."""
        text_lower = transcript.lower()
        
        # Look for novelty indicators
        novelty_indicators = {
            'new', 'novel', 'innovative', 'breakthrough', 'cutting-edge', 'state-of-the-art',
            'recent', 'latest', 'emerging', 'pioneering', 'revolutionary', 'groundbreaking',
            'unprecedented', 'first-time', 'never-before', 'newly', 'recently', 'just'
        }
        
        research_indicators = {
            'research', 'study', 'experiment', 'finding', 'discovery', 'investigation',
            'analysis', 'survey', 'paper', 'publication', 'journal', 'conference'
        }
        
        temporal_indicators = {
            '2024', '2023', 'this year', 'last month', 'recently', 'lately',
            'current', 'now', 'today', 'modern', 'contemporary'
        }
        
        # Count indicators
        novelty_count = sum(1 for word in novelty_indicators if word in text_lower)
        research_count = sum(1 for word in research_indicators if word in text_lower)
        temporal_count = sum(1 for word in temporal_indicators if word in text_lower)
        
        # Calculate novelty score
        words = text_lower.split()
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        novelty_density = (novelty_count / total_words) * 1000
        research_density = (research_count / total_words) * 500
        temporal_density = (temporal_count / total_words) * 300
        
        novelty_score = min(100.0, novelty_density + research_density + temporal_density)
        
        return novelty_score
    
    def assess_technical_accuracy_indicators(self, transcript: str) -> Dict[str, Any]:
        """
        Enhanced technical accuracy indicators assessment.
        
        Args:
            transcript: Cleaned transcript text
            
        Returns:
            Dictionary with detailed technical accuracy metrics
        """
        # Use existing method as base and enhance it
        base_accuracy = self.transcript_processor.detect_technical_accuracy_indicators(transcript)
        
        if not transcript:
            return base_accuracy
        
        # Enhanced accuracy analysis
        enhanced_metrics = self._analyze_enhanced_accuracy(transcript)
        
        # Combine base and enhanced metrics
        citation_quality = self._assess_citation_quality(transcript)
        source_credibility = self._assess_source_credibility(transcript)
        precision_indicators = self._assess_precision_indicators(transcript)
        uncertainty_handling = self._assess_uncertainty_handling(transcript)
        context_provision = self._assess_context_provision(transcript)
        
        # Calculate weighted accuracy score
        enhanced_accuracy_score = (
            citation_quality * self.accuracy_weights['citation_quality'] +
            source_credibility * self.accuracy_weights['source_credibility'] +
            precision_indicators * self.accuracy_weights['precision_indicators'] +
            uncertainty_handling * self.accuracy_weights['uncertainty_handling'] +
            context_provision * self.accuracy_weights['context_provision']
        )
        
        # Merge with base accuracy metrics
        result = base_accuracy.copy()
        result.update({
            'citation_quality': citation_quality,
            'source_credibility': source_credibility,
            'precision_indicators': precision_indicators,
            'uncertainty_handling': uncertainty_handling,
            'context_provision': context_provision,
            'enhanced_accuracy_score': min(100.0, max(0.0, enhanced_accuracy_score)),
            'detailed_metrics': enhanced_metrics
        })
        
        return result
    
    def _analyze_enhanced_accuracy(self, transcript: str) -> Dict[str, Any]:
        """Perform enhanced accuracy analysis."""
        text_lower = transcript.lower()
        
        # Quantitative evidence indicators
        quantitative_patterns = [
            r'\b\d+\s*%',  # Percentages
            r'\b\d+\s*(?:million|billion|thousand)',  # Large numbers
            r'\b\d+(?:\.\d+)?\s*(?:times|fold)',  # Multipliers
            r'\bstudies?\s+(?:show|found|indicate)',  # Study references
            r'\bresearch\s+(?:shows|indicates|suggests)'  # Research references
        ]
        
        quantitative_count = sum(len(re.findall(pattern, text_lower)) for pattern in quantitative_patterns)
        
        # Methodological indicators
        methodology_terms = {
            'methodology', 'approach', 'technique', 'procedure', 'protocol',
            'systematic', 'controlled', 'randomized', 'peer-reviewed', 'validated'
        }
        
        methodology_count = sum(1 for term in methodology_terms if term in text_lower)
        
        return {
            'quantitative_evidence_count': quantitative_count,
            'methodology_indicators_count': methodology_count,
            'has_numerical_support': quantitative_count > 0,
            'uses_scientific_methodology': methodology_count > 0
        }
    
    def _assess_citation_quality(self, transcript: str) -> float:
        """Assess quality of citations and references."""
        text_lower = transcript.lower()
        
        # High-quality citation patterns
        high_quality_patterns = [
            r'according to (?:a )?(?:study|research|paper) (?:by|from|published)',
            r'(?:research|study) published in',
            r'peer[- ]reviewed (?:study|research|paper)',
            r'meta[- ]analysis (?:shows|found|indicates)',
            r'systematic review (?:shows|found|indicates)'
        ]
        
        # Medium-quality citation patterns
        medium_quality_patterns = [
            r'according to (?:researchers|scientists|experts)',
            r'(?:study|research) (?:shows|found|indicates)',
            r'published research',
            r'scientific (?:study|research|evidence)'
        ]
        
        # Low-quality citation patterns (vague references)
        low_quality_patterns = [
            r'(?:they|people|experts) say',
            r'it is (?:said|believed|thought)',
            r'(?:some|many) (?:believe|think|say)',
            r'according to (?:some|sources)'
        ]
        
        high_quality_count = sum(len(re.findall(pattern, text_lower)) for pattern in high_quality_patterns)
        medium_quality_count = sum(len(re.findall(pattern, text_lower)) for pattern in medium_quality_patterns)
        low_quality_count = sum(len(re.findall(pattern, text_lower)) for pattern in low_quality_patterns)
        
        # Calculate citation quality score
        citation_score = (high_quality_count * 30 + medium_quality_count * 15 - low_quality_count * 10)
        
        # Normalize to 0-100 scale
        words = len(text_lower.split())
        if words > 0:
            citation_score = min(100.0, max(0.0, (citation_score / words) * 1000))
        else:
            citation_score = 0.0
        
        return citation_score
    
    def _assess_source_credibility(self, transcript: str) -> float:
        """Assess credibility of mentioned sources."""
        text_lower = transcript.lower()
        
        # Highly credible sources
        highly_credible = {
            'mit', 'stanford', 'harvard', 'berkeley', 'carnegie mellon', 'caltech',
            'oxford', 'cambridge', 'nature', 'science', 'cell', 'lancet',
            'openai', 'deepmind', 'google research', 'microsoft research',
            'arxiv', 'ieee', 'acm', 'nips', 'icml', 'iclr'
        }
        
        # Moderately credible sources
        moderately_credible = {
            'university', 'research institute', 'laboratory', 'tech company',
            'government agency', 'academic', 'professor', 'researcher'
        }
        
        # Count credible source mentions
        highly_credible_count = sum(1 for source in highly_credible if source in text_lower)
        moderately_credible_count = sum(1 for source in moderately_credible if source in text_lower)
        
        # Calculate credibility score
        credibility_score = (highly_credible_count * 25 + moderately_credible_count * 15)
        
        # Bonus for multiple different sources
        total_sources = highly_credible_count + moderately_credible_count
        if total_sources > 1:
            credibility_score += min(total_sources * 5, 20)  # Up to 20 bonus points
        
        return min(100.0, credibility_score)
    
    def _assess_precision_indicators(self, transcript: str) -> float:
        """Assess use of precise language and quantification."""
        text_lower = transcript.lower()
        
        # Precision indicators
        precise_terms = {
            'exactly', 'precisely', 'specifically', 'approximately', 'roughly',
            'about', 'around', 'nearly', 'close to', 'in the range of',
            'between', 'from', 'to', 'up to', 'as much as', 'at least',
            'on average', 'typically', 'generally', 'usually', 'often'
        }
        
        # Quantitative precision patterns
        quantitative_patterns = [
            r'\b\d+(?:\.\d+)?\s*%',  # Exact percentages
            r'\b\d+(?:\.\d+)?\s*(?:seconds|minutes|hours|days|months|years)',  # Time measurements
            r'\b\d+(?:\.\d+)?\s*(?:pixels|bytes|mb|gb|tb)',  # Technical measurements
            r'\b\d+(?:\.\d+)?\s*(?:million|billion|thousand|hundred)',  # Large numbers
        ]
        
        # Count precision indicators
        precise_count = sum(1 for term in precise_terms if term in text_lower)
        quantitative_count = sum(len(re.findall(pattern, text_lower)) for pattern in quantitative_patterns)
        
        # Calculate precision score
        words = len(text_lower.split())
        if words > 0:
            precision_density = ((precise_count + quantitative_count) / words) * 100
            precision_score = min(100.0, precision_density * 10)
        else:
            precision_score = 0.0
        
        return precision_score
    
    def _assess_uncertainty_handling(self, transcript: str) -> float:
        """Assess appropriate handling of uncertainty."""
        text_lower = transcript.lower()
        
        # Appropriate uncertainty expressions
        good_uncertainty = {
            'likely', 'probably', 'possibly', 'potentially', 'appears to',
            'seems to', 'suggests that', 'indicates that', 'preliminary',
            'early results', 'initial findings', 'more research needed',
            'further investigation', 'remains to be seen', 'unclear',
            'uncertain', 'may', 'might', 'could', 'would'
        }
        
        # Inappropriate certainty (red flags)
        overconfident_terms = {
            'definitely', 'absolutely', 'certainly', 'without doubt',
            'guaranteed', 'always', 'never', 'impossible', 'definitely true',
            'proven fact', 'undeniable', 'beyond question'
        }
        
        # Count uncertainty expressions
        good_uncertainty_count = sum(1 for term in good_uncertainty if term in text_lower)
        overconfident_count = sum(1 for term in overconfident_terms if term in text_lower)
        
        # Calculate uncertainty handling score
        words = len(text_lower.split())
        if words > 0:
            uncertainty_density = (good_uncertainty_count / words) * 100
            overconfidence_penalty = (overconfident_count / words) * 200
            
            uncertainty_score = min(100.0, uncertainty_density * 10 - overconfidence_penalty)
        else:
            uncertainty_score = 0.0
        
        return max(0.0, uncertainty_score)
    
    def _assess_context_provision(self, transcript: str) -> float:
        """Assess provision of appropriate context."""
        text_lower = transcript.lower()
        
        # Context provision indicators
        context_terms = {
            'background', 'context', 'historically', 'previously', 'in the past',
            'traditionally', 'compared to', 'in contrast', 'unlike', 'similar to',
            'building on', 'based on', 'following', 'as a result of',
            'motivated by', 'inspired by', 'in response to', 'addressing'
        }
        
        # Temporal context
        temporal_context = {
            'recently', 'lately', 'currently', 'now', 'today', 'this year',
            'last year', 'in 2023', 'in 2024', 'over the past', 'since'
        }
        
        # Comparative context
        comparative_context = {
            'compared to', 'versus', 'rather than', 'instead of', 'unlike',
            'similar to', 'different from', 'better than', 'worse than'
        }
        
        # Count context indicators
        context_count = sum(1 for term in context_terms if term in text_lower)
        temporal_count = sum(1 for term in temporal_context if term in text_lower)
        comparative_count = sum(1 for term in comparative_context if term in text_lower)
        
        total_context = context_count + temporal_count + comparative_count
        
        # Calculate context provision score
        words = len(text_lower.split())
        if words > 0:
            context_density = (total_context / words) * 100
            context_score = min(100.0, context_density * 15)
        else:
            context_score = 0.0
        
        # Bonus for multiple types of context
        context_types = sum([context_count > 0, temporal_count > 0, comparative_count > 0])
        type_bonus = context_types * 10  # Up to 30 points for all three types
        
        return min(100.0, context_score + type_bonus)
    
    def calculate_combined_quality_score(self, video: VideoData, 
                                       engagement_metrics: Optional[EngagementMetrics] = None,
                                       comments: Optional[List[str]] = None) -> ContentQualityMetrics:
        """
        Calculate combined content and engagement quality score.
        
        Args:
            video: VideoData instance with transcript
            engagement_metrics: Optional pre-calculated engagement metrics
            comments: Optional list of comments for engagement analysis
            
        Returns:
            ContentQualityMetrics with comprehensive quality assessment
        """
        try:
            if not video.transcript:
                # Return minimal metrics for videos without transcripts
                return ContentQualityMetrics(
                    coherence_score=0.0,
                    information_density=0.0,
                    technical_accuracy=0.0,
                    clarity_score=0.0,
                    depth_score=0.0,
                    structure_score=0.0,
                    overall_content_score=0.0,
                    engagement_integration_score=0.0,
                    final_quality_score=0.0,
                    detailed_metrics={'error': 'No transcript available'}
                )
            
            # Calculate content quality metrics
            coherence_metrics = self.evaluate_transcript_coherence(video.transcript)
            density_metrics = self.measure_information_density(video.transcript)
            accuracy_metrics = self.assess_technical_accuracy_indicators(video.transcript)
            
            # Calculate engagement metrics if not provided
            if engagement_metrics is None:
                engagement_metrics = self.engagement_analyzer.analyze_video_engagement(video, comments)
            
            # Extract individual scores
            coherence_score = coherence_metrics['overall_coherence']
            information_density = density_metrics['overall_density']
            technical_accuracy = accuracy_metrics.get('enhanced_accuracy_score', 
                                                    accuracy_metrics.get('accuracy_score', 0.0))
            
            # Calculate additional quality dimensions
            clarity_score = self._calculate_clarity_score(video.transcript, coherence_metrics)
            depth_score = self._calculate_depth_score(video.transcript, density_metrics, accuracy_metrics)
            structure_score = self._calculate_structure_score(video.transcript, coherence_metrics)
            
            # Calculate overall content score (weighted combination)
            content_weights = {
                'coherence': 0.25,
                'density': 0.20,
                'accuracy': 0.20,
                'clarity': 0.15,
                'depth': 0.15,
                'structure': 0.05
            }
            
            overall_content_score = (
                coherence_score * content_weights['coherence'] +
                information_density * content_weights['density'] +
                technical_accuracy * content_weights['accuracy'] +
                clarity_score * content_weights['clarity'] +
                depth_score * content_weights['depth'] +
                structure_score * content_weights['structure']
            )
            
            # Calculate engagement integration score
            engagement_integration_score = self._calculate_engagement_integration(
                engagement_metrics, overall_content_score
            )
            
            # Calculate final quality score (content + engagement integration)
            final_quality_score = (
                overall_content_score * 0.7 +  # 70% content quality
                engagement_integration_score * 0.3  # 30% engagement quality
            )
            
            # Compile detailed metrics
            detailed_metrics = {
                'coherence_details': coherence_metrics,
                'density_details': density_metrics,
                'accuracy_details': accuracy_metrics,
                'engagement_details': engagement_metrics.__dict__ if engagement_metrics else {},
                'content_weights': content_weights,
                'video_id': video.video_id,
                'transcript_length': len(video.transcript.split()) if video.transcript else 0
            }
            
            return ContentQualityMetrics(
                coherence_score=coherence_score,
                information_density=information_density,
                technical_accuracy=technical_accuracy,
                clarity_score=clarity_score,
                depth_score=depth_score,
                structure_score=structure_score,
                overall_content_score=overall_content_score,
                engagement_integration_score=engagement_integration_score,
                final_quality_score=min(100.0, max(0.0, final_quality_score)),
                detailed_metrics=detailed_metrics
            )
            
        except Exception as e:
            logger.error(f"Error calculating quality score for video {video.video_id}: {e}")
            
            # Return error metrics
            return ContentQualityMetrics(
                coherence_score=0.0,
                information_density=0.0,
                technical_accuracy=0.0,
                clarity_score=0.0,
                depth_score=0.0,
                structure_score=0.0,
                overall_content_score=0.0,
                engagement_integration_score=0.0,
                final_quality_score=0.0,
                detailed_metrics={'error': str(e)}
            )
    
    def _calculate_clarity_score(self, transcript: str, coherence_metrics: Dict[str, float]) -> float:
        """Calculate clarity score based on readability and structure."""
        if not transcript:
            return 0.0
        
        # Use coherence sentence flow as base
        base_clarity = coherence_metrics.get('sentence_flow_score', 50.0)
        
        # Add readability factors
        sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
        
        if not sentences:
            return base_clarity
        
        # Average sentence length (clarity factor)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        length_clarity = 100 - abs(avg_sentence_length - 15) * 2  # Optimal around 15 words
        length_clarity = max(0, min(100, length_clarity))
        
        # Jargon density (too much jargon reduces clarity)
        words = transcript.lower().split()
        jargon_terms = {
            'algorithm', 'paradigm', 'methodology', 'infrastructure', 'architecture',
            'implementation', 'optimization', 'scalability', 'heterogeneous', 'synchronous'
        }
        
        jargon_count = sum(1 for word in words if word in jargon_terms)
        jargon_density = (jargon_count / len(words)) * 100 if words else 0
        
        # Moderate jargon is good, too much hurts clarity
        if jargon_density > 5:  # More than 5% jargon
            jargon_score = 100 - (jargon_density - 5) * 5
        else:
            jargon_score = 80 + jargon_density * 4  # Up to 100 for optimal jargon
        
        jargon_score = max(0, min(100, jargon_score))
        
        # Combine clarity factors
        clarity_score = (base_clarity * 0.5 + length_clarity * 0.3 + jargon_score * 0.2)
        
        return min(100.0, max(0.0, clarity_score))
    
    def _calculate_depth_score(self, transcript: str, density_metrics: Dict[str, float], 
                             accuracy_metrics: Dict[str, Any]) -> float:
        """Calculate depth score based on technical sophistication and detail."""
        if not transcript:
            return 0.0
        
        # Base depth from technical terminology
        base_depth = density_metrics.get('technical_terminology', 50.0)
        
        # Add technical accuracy component
        accuracy_component = accuracy_metrics.get('enhanced_accuracy_score', 
                                                accuracy_metrics.get('accuracy_score', 0.0))
        
        # Add conceptual depth indicators
        conceptual_depth = density_metrics.get('conceptual_diversity', 50.0)
        
        # Look for advanced concepts
        text_lower = transcript.lower()
        advanced_concepts = {
            'theoretical', 'empirical', 'systematic', 'comprehensive', 'sophisticated',
            'fundamental', 'underlying', 'principles', 'mechanisms', 'implications',
            'ramifications', 'consequences', 'interdisciplinary', 'multifaceted'
        }
        
        advanced_count = sum(1 for concept in advanced_concepts if concept in text_lower)
        words = len(text_lower.split())
        
        if words > 0:
            advanced_density = (advanced_count / words) * 100
            advanced_score = min(advanced_density * 10, 30)  # Up to 30 points
        else:
            advanced_score = 0
        
        # Combine depth factors
        depth_score = (
            base_depth * 0.4 +
            accuracy_component * 0.3 +
            conceptual_depth * 0.2 +
            advanced_score * 0.1
        )
        
        return min(100.0, max(0.0, depth_score))
    
    def _calculate_structure_score(self, transcript: str, coherence_metrics: Dict[str, float]) -> float:
        """Calculate structure score based on organization and flow."""
        if not transcript:
            return 0.0
        
        # Use structural patterns from coherence analysis
        structure_base = coherence_metrics.get('structural_patterns_score', 50.0)
        
        # Add logical flow assessment
        flow_score = coherence_metrics.get('transitions_score', 50.0)
        
        # Combine structure factors
        structure_score = (structure_base * 0.6 + flow_score * 0.4)
        
        return min(100.0, max(0.0, structure_score))
    
    def _calculate_engagement_integration(self, engagement_metrics: EngagementMetrics, 
                                        content_score: float) -> float:
        """Calculate how well engagement metrics integrate with content quality."""
        if not engagement_metrics:
            return 50.0  # Neutral score if no engagement data
        
        # Base engagement score
        base_engagement = engagement_metrics.overall_engagement_score
        
        # Sentiment integration (positive sentiment supports quality content)
        sentiment_score = engagement_metrics.comment_sentiment_score
        sentiment_bonus = (sentiment_score + 1) * 25  # Convert -1 to 1 scale to 0-50
        
        # Like ratio integration
        like_ratio_score = engagement_metrics.like_ratio * 100
        
        # Threshold bonus (meeting 80% threshold is important)
        threshold_bonus = 20 if engagement_metrics.meets_threshold else 0
        
        # Content-engagement synergy bonus
        # Good content with good engagement gets extra points
        if content_score > 70 and base_engagement > 70:
            synergy_bonus = min((content_score + base_engagement - 140) * 0.2, 10)
        else:
            synergy_bonus = 0
        
        # Combine engagement integration factors
        integration_score = (
            base_engagement * 0.4 +
            sentiment_bonus * 0.2 +
            like_ratio_score * 0.2 +
            threshold_bonus * 0.1 +
            synergy_bonus * 0.1
        )
        
        return min(100.0, max(0.0, integration_score))
    
    def batch_analyze_content_quality(self, videos: List[VideoData],
                                     video_comments: Optional[Dict[str, List[str]]] = None) -> Dict[str, ContentQualityMetrics]:
        """
        Analyze content quality for multiple videos.
        
        Args:
            videos: List of VideoData instances to analyze
            video_comments: Optional dictionary mapping video_id to comments
            
        Returns:
            Dictionary mapping video_id to ContentQualityMetrics
        """
        results = {}
        
        for video in videos:
            comments = video_comments.get(video.video_id, []) if video_comments else []
            results[video.video_id] = self.calculate_combined_quality_score(video, comments=comments)
        
        return results
    
    def filter_videos_by_quality(self, videos: List[VideoData],
                                min_content_score: float = 70.0,
                                min_final_score: float = 75.0,
                                require_transcript: bool = True) -> List[VideoData]:
        """
        Filter videos based on content quality criteria.
        
        Args:
            videos: List of videos to filter
            min_content_score: Minimum content quality score required
            min_final_score: Minimum final quality score required
            require_transcript: Whether to require transcript availability
            
        Returns:
            List of videos meeting quality criteria
        """
        filtered_videos = []
        
        for video in videos:
            # Skip videos without transcript if required
            if require_transcript and not video.transcript:
                continue
            
            # Calculate quality metrics
            quality_metrics = self.calculate_combined_quality_score(video)
            
            # Check criteria
            meets_content_score = quality_metrics.overall_content_score >= min_content_score
            meets_final_score = quality_metrics.final_quality_score >= min_final_score
            
            if meets_content_score and meets_final_score:
                # Attach quality metrics to video for later use
                if not hasattr(video, 'content_quality_analysis'):
                    video.content_quality_analysis = quality_metrics
                filtered_videos.append(video)
        
        return filtered_videos
    
    def get_quality_summary(self, videos: List[VideoData]) -> Dict[str, Any]:
        """
        Get summary statistics for content quality across multiple videos.
        
        Args:
            videos: List of videos to summarize
            
        Returns:
            Dictionary with quality summary statistics
        """
        if not videos:
            return {
                'total_videos': 0,
                'avg_content_score': 0.0,
                'avg_final_score': 0.0,
                'high_quality_count': 0,
                'high_quality_rate': 0.0
            }
        
        total_content_score = 0.0
        total_final_score = 0.0
        high_quality_count = 0
        
        for video in videos:
            quality_metrics = self.calculate_combined_quality_score(video)
            total_content_score += quality_metrics.overall_content_score
            total_final_score += quality_metrics.final_quality_score
            
            if quality_metrics.final_quality_score >= 75.0:
                high_quality_count += 1
        
        return {
            'total_videos': len(videos),
            'avg_content_score': total_content_score / len(videos),
            'avg_final_score': total_final_score / len(videos),
            'high_quality_count': high_quality_count,
            'high_quality_rate': high_quality_count / len(videos)
        }