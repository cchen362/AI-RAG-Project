"""
Evaluation Metrics - Performance Comparison for Agentic RAG

This module provides comprehensive evaluation metrics to compare the performance
of agentic multi-turn reasoning against baseline single-turn approaches.
"""

import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json
import math

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    overall_score: float
    relevance_score: float
    completeness_score: float
    accuracy_score: float
    coherence_score: float
    efficiency_score: float
    detailed_scores: Dict[str, float]
    explanation: str

class EvaluationMetrics:
    """
    Comprehensive evaluation system for comparing agentic and baseline RAG approaches.
    Uses multiple metrics to assess quality, relevance, efficiency, and user experience.
    """
    
    def __init__(self):
        """Initialize evaluation metrics system."""
        self.evaluation_history = []
        self.baseline_stats = defaultdict(list)
        self.agentic_stats = defaultdict(list)
    
    def evaluate_response(self, response: str, expected_answer: str, 
                         query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a response against expected answer.
        
        Args:
            response: Generated response to evaluate
            expected_answer: Expected/reference answer
            query: Original query
            context: Optional context information (execution time, sources, etc.)
            
        Returns:
            Evaluation results dictionary
        """
        if not response or not expected_answer:
            return self._create_empty_evaluation()
        
        # Calculate individual metrics
        relevance = self._calculate_relevance(response, expected_answer, query)
        completeness = self._calculate_completeness(response, expected_answer)
        accuracy = self._calculate_accuracy(response, expected_answer)
        coherence = self._calculate_coherence(response)
        
        # Calculate efficiency if context provided
        efficiency = 1.0  # Default perfect efficiency
        if context and "execution_time" in context:
            efficiency = self._calculate_efficiency(context["execution_time"], len(response))
        
        # Calculate overall score (weighted average)
        weights = {
            "relevance": 0.25,
            "completeness": 0.25,
            "accuracy": 0.25,
            "coherence": 0.15,
            "efficiency": 0.10
        }
        
        overall_score = (
            relevance * weights["relevance"] +
            completeness * weights["completeness"] +
            accuracy * weights["accuracy"] +
            coherence * weights["coherence"] +
            efficiency * weights["efficiency"]
        )
        
        # Detailed scores for analysis
        detailed_scores = {
            "keyword_overlap": self._calculate_keyword_overlap(response, expected_answer),
            "semantic_similarity": self._estimate_semantic_similarity(response, expected_answer),
            "response_length_ratio": len(response) / max(len(expected_answer), 1),
            "technical_terms_coverage": self._calculate_technical_coverage(response, expected_answer),
            "structure_quality": self._assess_structure_quality(response)
        }
        
        # Generate explanation
        explanation = self._generate_evaluation_explanation(
            relevance, completeness, accuracy, coherence, efficiency, detailed_scores
        )
        
        result = {
            "overall_score": round(overall_score, 3),
            "relevance_score": round(relevance, 3),
            "completeness_score": round(completeness, 3),
            "accuracy_score": round(accuracy, 3),
            "coherence_score": round(coherence, 3),
            "efficiency_score": round(efficiency, 3),
            "detailed_scores": {k: round(v, 3) for k, v in detailed_scores.items()},
            "explanation": explanation,
            "timestamp": time.time()
        }
        
        # Store for historical analysis
        self.evaluation_history.append({
            "query": query,
            "response": response,
            "expected": expected_answer,
            "results": result,
            "context": context
        })
        
        return result
    
    def compare_approaches(self, agentic_result: Dict, baseline_result: Dict,
                          query: str) -> Dict[str, Any]:
        """
        Compare agentic and baseline approaches side by side.
        
        Args:
            agentic_result: Results from agentic approach
            baseline_result: Results from baseline approach
            query: Original query
            
        Returns:
            Comparative analysis
        """
        comparison = {
            "query": query,
            "winner": "tie",
            "score_difference": 0.0,
            "advantages": {
                "agentic": [],
                "baseline": []
            },
            "metrics_comparison": {},
            "recommendation": ""
        }
        
        if "evaluation" not in agentic_result or "evaluation" not in baseline_result:
            comparison["error"] = "Missing evaluation data"
            return comparison
        
        agentic_eval = agentic_result["evaluation"]
        baseline_eval = baseline_result["evaluation"]
        
        # Compare overall scores
        agentic_score = agentic_eval["overall_score"]
        baseline_score = baseline_eval["overall_score"]
        score_diff = agentic_score - baseline_score
        
        comparison["score_difference"] = round(score_diff, 3)
        
        if score_diff > 0.05:  # Significant improvement threshold
            comparison["winner"] = "agentic"
        elif score_diff < -0.05:
            comparison["winner"] = "baseline"
        else:
            comparison["winner"] = "tie"
        
        # Detailed metric comparisons
        metrics = ["relevance_score", "completeness_score", "accuracy_score", "coherence_score", "efficiency_score"]
        for metric in metrics:
            comparison["metrics_comparison"][metric] = {
                "agentic": agentic_eval.get(metric, 0),
                "baseline": baseline_eval.get(metric, 0),
                "difference": agentic_eval.get(metric, 0) - baseline_eval.get(metric, 0)
            }
        
        # Identify advantages
        if agentic_result.get("steps", 0) > 1:
            comparison["advantages"]["agentic"].append("Multi-step reasoning")
        
        if len(agentic_result.get("sources_used", [])) > len(baseline_result.get("sources_used", [])):
            comparison["advantages"]["agentic"].append("Multi-source intelligence")
        
        if baseline_result.get("execution_time", float('inf')) < agentic_result.get("execution_time", 0):
            comparison["advantages"]["baseline"].append("Faster execution")
        
        if baseline_eval.get("efficiency_score", 0) > agentic_eval.get("efficiency_score", 0):
            comparison["advantages"]["baseline"].append("Better efficiency")
        
        # Generate recommendation
        comparison["recommendation"] = self._generate_recommendation(comparison, query)
        
        return comparison
    
    def calculate_improvement_metrics(self, test_results: List[Dict]) -> Dict[str, Any]:
        """
        Calculate aggregate improvement metrics across multiple tests.
        
        Args:
            test_results: List of test results with both agentic and baseline data
            
        Returns:
            Aggregate improvement analysis
        """
        agentic_scores = []
        baseline_scores = []
        time_comparisons = []
        complexity_analysis = defaultdict(list)
        
        for result in test_results:
            if "agentic" in result and "baseline" in result:
                # Score comparisons
                if "evaluation" in result["agentic"]:
                    agentic_scores.append(result["agentic"]["evaluation"]["overall_score"])
                if "evaluation" in result["baseline"]:
                    baseline_scores.append(result["baseline"]["evaluation"]["overall_score"])
                
                # Time comparisons
                agentic_time = result["agentic"].get("execution_time", 0)
                baseline_time = result["baseline"].get("execution_time", 0)
                if agentic_time > 0 and baseline_time > 0:
                    time_comparisons.append({
                        "agentic": agentic_time,
                        "baseline": baseline_time,
                        "ratio": agentic_time / baseline_time
                    })
                
                # Complexity analysis
                complexity = result.get("test_category", "general")
                if "evaluation" in result["agentic"] and "evaluation" in result["baseline"]:
                    improvement = (result["agentic"]["evaluation"]["overall_score"] - 
                                 result["baseline"]["evaluation"]["overall_score"])
                    complexity_analysis[complexity].append(improvement)
        
        # Calculate aggregate metrics
        metrics = {}
        
        if agentic_scores and baseline_scores:
            metrics["average_agentic_score"] = sum(agentic_scores) / len(agentic_scores)
            metrics["average_baseline_score"] = sum(baseline_scores) / len(baseline_scores)
            metrics["average_improvement"] = metrics["average_agentic_score"] - metrics["average_baseline_score"]
            metrics["improvement_percentage"] = (metrics["average_improvement"] / 
                                               max(metrics["average_baseline_score"], 0.01)) * 100
            
            # Count wins
            wins = {"agentic": 0, "baseline": 0, "tie": 0}
            for i in range(min(len(agentic_scores), len(baseline_scores))):
                diff = agentic_scores[i] - baseline_scores[i]
                if diff > 0.05:
                    wins["agentic"] += 1
                elif diff < -0.05:
                    wins["baseline"] += 1
                else:
                    wins["tie"] += 1
            
            metrics["win_statistics"] = wins
            metrics["win_rate"] = wins["agentic"] / sum(wins.values()) if sum(wins.values()) > 0 else 0
        
        # Time analysis
        if time_comparisons:
            avg_time_ratio = sum(t["ratio"] for t in time_comparisons) / len(time_comparisons)
            metrics["average_time_ratio"] = avg_time_ratio
            metrics["agentic_slower_ratio"] = avg_time_ratio
            
            faster_count = sum(1 for t in time_comparisons if t["ratio"] < 1.0)
            metrics["agentic_faster_percentage"] = faster_count / len(time_comparisons) * 100
        
        # Complexity-based analysis
        metrics["complexity_analysis"] = {}
        for complexity, improvements in complexity_analysis.items():
            if improvements:
                metrics["complexity_analysis"][complexity] = {
                    "average_improvement": sum(improvements) / len(improvements),
                    "positive_improvements": sum(1 for i in improvements if i > 0),
                    "total_tests": len(improvements)
                }
        
        return metrics
    
    def _calculate_relevance(self, response: str, expected: str, query: str) -> float:
        """Calculate relevance score based on query-response alignment."""
        # Extract key terms from query
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        response_terms = set(re.findall(r'\b\w+\b', response.lower()))
        expected_terms = set(re.findall(r'\b\w+\b', expected.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        query_terms -= stop_words
        response_terms -= stop_words
        expected_terms -= stop_words
        
        if not query_terms:
            return 0.5  # Neutral score if no meaningful query terms
        
        # Calculate term overlap
        query_response_overlap = len(query_terms.intersection(response_terms))
        query_expected_overlap = len(query_terms.intersection(expected_terms))
        
        # Relevance based on how well response addresses query terms
        relevance = query_response_overlap / len(query_terms) if query_terms else 0
        
        return min(1.0, relevance)
    
    def _calculate_completeness(self, response: str, expected: str) -> float:
        """Calculate completeness score based on coverage of expected content."""
        expected_terms = set(re.findall(r'\b\w+\b', expected.lower()))
        response_terms = set(re.findall(r'\b\w+\b', response.lower()))
        
        if not expected_terms:
            return 1.0  # Perfect score if no expected content
        
        # Calculate coverage
        coverage = len(expected_terms.intersection(response_terms)) / len(expected_terms)
        
        # Adjust for response length (penalize too short responses)
        length_ratio = len(response) / max(len(expected), 1)
        length_factor = min(1.0, length_ratio * 0.5 + 0.5)  # Ranges from 0.5 to 1.0
        
        return min(1.0, coverage * length_factor)
    
    def _calculate_accuracy(self, response: str, expected: str) -> float:
        """Calculate accuracy score based on factual correctness."""
        # Simple keyword-based accuracy (can be enhanced with LLM evaluation)
        response_lower = response.lower()
        expected_lower = expected.lower()
        
        # Look for key technical terms and concepts
        key_terms = re.findall(r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)*\b', expected)  # CamelCase terms
        key_terms.extend(re.findall(r'\b\d+(?:\.\d+)?\b', expected))  # Numbers
        
        if not key_terms:
            # Fallback to semantic similarity estimation
            return self._estimate_semantic_similarity(response, expected)
        
        # Check presence of key terms
        present_terms = sum(1 for term in key_terms if term.lower() in response_lower)
        accuracy = present_terms / len(key_terms) if key_terms else 0.5
        
        return min(1.0, accuracy)
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate coherence score based on response structure and flow."""
        if not response:
            return 0.0
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.8  # Short responses are typically coherent
        
        # Check for structure indicators
        structure_indicators = 0
        
        # Look for logical connectors
        connectors = ['however', 'therefore', 'furthermore', 'additionally', 'consequently', 'moreover', 'thus', 'hence']
        for connector in connectors:
            if connector in response.lower():
                structure_indicators += 1
        
        # Look for formatting (bullets, numbers, etc.)
        if re.search(r'^\s*[-*•]\s', response, re.MULTILINE) or re.search(r'^\s*\d+\.', response, re.MULTILINE):
            structure_indicators += 2
        
        # Look for clear sections
        if '**' in response or re.search(r'^[A-Z][^a-z]*:', response, re.MULTILINE):
            structure_indicators += 1
        
        # Calculate coherence based on indicators and length
        base_coherence = 0.7  # Base score
        structure_bonus = min(0.3, structure_indicators * 0.1)
        
        return min(1.0, base_coherence + structure_bonus)
    
    def _calculate_efficiency(self, execution_time: float, response_length: int) -> float:
        """Calculate efficiency score based on time and output quality."""
        if execution_time <= 0:
            return 1.0
        
        # Optimal time ranges (can be adjusted based on system performance)
        optimal_time = 2.0  # 2 seconds optimal
        acceptable_time = 10.0  # 10 seconds acceptable
        
        # Calculate time efficiency
        if execution_time <= optimal_time:
            time_score = 1.0
        elif execution_time <= acceptable_time:
            time_score = 1.0 - (execution_time - optimal_time) / (acceptable_time - optimal_time) * 0.3
        else:
            time_score = 0.7 * (acceptable_time / execution_time)  # Diminishing returns
        
        # Adjust for response quality (longer responses get some time bonus)
        quality_factor = min(1.2, (response_length / 500) + 0.8)  # Assume 500 chars is good length
        
        return min(1.0, time_score * quality_factor)
    
    def _calculate_keyword_overlap(self, response: str, expected: str) -> float:
        """Calculate simple keyword overlap score."""
        response_words = set(re.findall(r'\b\w+\b', response.lower()))
        expected_words = set(re.findall(r'\b\w+\b', expected.lower()))
        
        if not expected_words:
            return 0.0
        
        overlap = len(response_words.intersection(expected_words))
        return overlap / len(expected_words)
    
    def _estimate_semantic_similarity(self, response: str, expected: str) -> float:
        """Estimate semantic similarity (simple implementation)."""
        # This is a simplified version - in production, you'd use embeddings
        response_bigrams = set(self._get_bigrams(response.lower()))
        expected_bigrams = set(self._get_bigrams(expected.lower()))
        
        if not expected_bigrams:
            return 0.5
        
        overlap = len(response_bigrams.intersection(expected_bigrams))
        return min(1.0, overlap / len(expected_bigrams) * 2)  # Boost bigram matches
    
    def _get_bigrams(self, text: str) -> List[str]:
        """Extract bigrams from text."""
        words = re.findall(r'\b\w+\b', text)
        return [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    
    def _calculate_technical_coverage(self, response: str, expected: str) -> float:
        """Calculate coverage of technical terms."""
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+(?:_\w+)+\b',  # Snake_case
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
            r'\b\d+(?:\.\d+)?%?\b',  # Numbers/percentages
        ]
        
        expected_technical = set()
        response_technical = set()
        
        for pattern in technical_patterns:
            expected_technical.update(re.findall(pattern, expected))
            response_technical.update(re.findall(pattern, response))
        
        if not expected_technical:
            return 1.0
        
        coverage = len(expected_technical.intersection(response_technical)) / len(expected_technical)
        return min(1.0, coverage)
    
    def _assess_structure_quality(self, response: str) -> float:
        """Assess the structural quality of the response."""
        score = 0.5  # Base score
        
        # Check for clear formatting
        if '**' in response or '__' in response:
            score += 0.1  # Bold formatting
        
        if re.search(r'^\s*[-*•]\s', response, re.MULTILINE):
            score += 0.1  # Bullet points
        
        if re.search(r'^\s*\d+\.', response, re.MULTILINE):
            score += 0.1  # Numbered lists
        
        # Check for logical flow
        paragraphs = response.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.1  # Multiple paragraphs
        
        # Check for conclusion/summary
        if any(word in response.lower() for word in ['conclusion', 'summary', 'in summary', 'overall']):
            score += 0.1
        
        return min(1.0, score)
    
    def _generate_evaluation_explanation(self, relevance: float, completeness: float,
                                       accuracy: float, coherence: float, efficiency: float,
                                       detailed: Dict[str, float]) -> str:
        """Generate human-readable explanation of evaluation."""
        explanations = []
        
        # Relevance
        if relevance > 0.8:
            explanations.append("Highly relevant to the query")
        elif relevance > 0.6:
            explanations.append("Reasonably relevant to the query")
        else:
            explanations.append("Limited relevance to the query")
        
        # Completeness
        if completeness > 0.8:
            explanations.append("comprehensive coverage")
        elif completeness > 0.6:
            explanations.append("adequate coverage")
        else:
            explanations.append("incomplete coverage")
        
        # Accuracy
        if accuracy > 0.8:
            explanations.append("high accuracy")
        elif accuracy > 0.6:
            explanations.append("reasonable accuracy")
        else:
            explanations.append("questionable accuracy")
        
        # Efficiency
        if efficiency > 0.8:
            explanations.append("efficient delivery")
        elif efficiency > 0.6:
            explanations.append("acceptable performance")
        else:
            explanations.append("slow performance")
        
        return f"Response shows {', '.join(explanations)}."
    
    def _generate_recommendation(self, comparison: Dict, query: str) -> str:
        """Generate recommendation based on comparison results."""
        winner = comparison["winner"]
        score_diff = abs(comparison["score_difference"])
        
        if winner == "agentic" and score_diff > 0.1:
            return "Strong recommendation for agentic approach - significant quality improvement"
        elif winner == "agentic" and score_diff > 0.05:
            return "Moderate recommendation for agentic approach - noticeable improvement"
        elif winner == "baseline" and score_diff > 0.1:
            return "Baseline approach preferred - simpler and more efficient for this query type"
        elif winner == "baseline" and score_diff > 0.05:
            return "Baseline approach slightly better - consider complexity vs. benefit"
        else:
            return "Both approaches perform similarly - choice depends on specific requirements"
    
    def _create_empty_evaluation(self) -> Dict[str, Any]:
        """Create empty evaluation result for error cases."""
        return {
            "overall_score": 0.0,
            "relevance_score": 0.0,
            "completeness_score": 0.0,
            "accuracy_score": 0.0,
            "coherence_score": 0.0,
            "efficiency_score": 0.0,
            "detailed_scores": {},
            "explanation": "Evaluation failed - missing response or expected answer",
            "timestamp": time.time()
        }