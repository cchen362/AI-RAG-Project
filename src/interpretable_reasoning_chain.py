"""
Interpretable Reasoning Chain for Graph-R1 Agentic RAG System

Provides comprehensive audit trails and reasoning transparency with:
1. Step-by-step decision logging with rationale
2. Interactive graph traversal visualization  
3. Complete document path tracking
4. Performance metrics and cost analysis
5. Confidence evolution tracking
6. Export capabilities for audit requirements

Key Innovation: Makes the "black box" of RAG completely transparent
with detailed explanations of why each decision was made.
"""

import os
import sys
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict

# Visualization imports (optional)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import related components
try:
    from src.graph_traversal_engine import TraversalPath, PathNode, TraversalBudget
    from src.hypergraph_constructor import HypergraphNode, HypergraphEdge
except ImportError:
    try:
        from graph_traversal_engine import TraversalPath, PathNode, TraversalBudget
        from hypergraph_constructor import HypergraphNode, HypergraphEdge
    except ImportError:
        # Define minimal classes for standalone operation
        pass

logger = logging.getLogger(__name__)

@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning chain."""
    step_id: str
    timestamp: float
    step_type: str  # 'query_analysis', 'entry_selection', 'node_visit', 'path_decision', 'confidence_check', 'stopping'
    description: str
    details: Dict[str, Any]
    confidence_before: float
    confidence_after: float
    tokens_used: int
    cost_estimate: float
    decision_rationale: str
    alternatives_considered: List[str]
    node_id: Optional[str] = None
    source_type: Optional[str] = None

@dataclass
class DocumentPath:
    """Tracks which documents were accessed and why."""
    document_id: str
    document_name: str
    source_type: str
    access_timestamp: float
    access_reason: str
    confidence_score: float
    content_snippet: str
    hierarchical_level: int
    parent_document: Optional[str] = None
    cost_incurred: float = 0.0

@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking."""
    total_time: float
    query_analysis_time: float
    graph_construction_time: float
    traversal_time: float
    synthesis_time: float
    total_tokens_used: int
    tokens_by_operation: Dict[str, int]
    total_cost: float
    cost_by_operation: Dict[str, float]
    nodes_visited: int
    edges_traversed: int
    sources_accessed: Dict[str, int]
    confidence_evolution: List[Tuple[float, float]]  # (timestamp, confidence)
    memory_usage_mb: Optional[float] = None

class ReasoningChain:
    """Main reasoning chain manager with complete audit capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Chain state
        self.chain_id = f"reasoning_chain_{int(time.time() * 1000)}"
        self.start_time = time.time()
        self.query = ""
        self.final_answer = ""
        
        # Reasoning components
        self.reasoning_steps: List[ReasoningStep] = []
        self.document_paths: List[DocumentPath] = []
        self.performance_metrics = PerformanceMetrics(
            total_time=0.0,
            query_analysis_time=0.0,
            graph_construction_time=0.0,
            traversal_time=0.0,
            synthesis_time=0.0,
            total_tokens_used=0,
            tokens_by_operation={},
            total_cost=0.0,
            cost_by_operation={},
            nodes_visited=0,
            edges_traversed=0,
            sources_accessed={},
            confidence_evolution=[]
        )
        
        # State tracking
        self.current_confidence = 0.0
        self.step_counter = 0
        self.decision_points: List[Dict[str, Any]] = []
        
        # Export configuration
        self.export_formats = ['json', 'html', 'markdown', 'csv']
        self.include_visualizations = PLOTLY_AVAILABLE
        
        logger.info(f"✅ ReasoningChain initialized: {self.chain_id}")
    
    def start_reasoning(self, query: str):
        """Initialize reasoning chain for a new query."""
        self.query = query
        self.start_time = time.time()
        
        # Log initial step
        self.add_reasoning_step(
            step_type='query_analysis',
            description=f"Starting agentic analysis for query: '{query[:50]}...'",
            details={
                'query': query,
                'query_length': len(query),
                'query_complexity': self._assess_query_complexity(query)
            },
            confidence_before=0.0,
            confidence_after=0.0,
            decision_rationale="Beginning comprehensive multi-source graph traversal",
            alternatives_considered=["Simple similarity search", "Single-source lookup"]
        )
        
        logger.info(f"🚀 Started reasoning chain for: '{query[:30]}...'")
    
    def add_reasoning_step(self, step_type: str, description: str, details: Dict[str, Any],
                          confidence_before: float, confidence_after: float,
                          decision_rationale: str, alternatives_considered: List[str],
                          tokens_used: int = 0, cost_estimate: float = 0.0,
                          node_id: Optional[str] = None, source_type: Optional[str] = None):
        """Add a new reasoning step with complete context."""
        
        self.step_counter += 1
        step_id = f"step_{self.step_counter:03d}_{step_type}"
        timestamp = time.time() - self.start_time
        
        step = ReasoningStep(
            step_id=step_id,
            timestamp=timestamp,
            step_type=step_type,
            description=description,
            details=details,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            tokens_used=tokens_used,
            cost_estimate=cost_estimate,
            decision_rationale=decision_rationale,
            alternatives_considered=alternatives_considered,
            node_id=node_id,
            source_type=source_type
        )
        
        self.reasoning_steps.append(step)
        
        # Update performance metrics
        self.performance_metrics.total_tokens_used += tokens_used
        self.performance_metrics.total_cost += cost_estimate
        self.performance_metrics.confidence_evolution.append((timestamp, confidence_after))
        
        if step_type not in self.performance_metrics.tokens_by_operation:
            self.performance_metrics.tokens_by_operation[step_type] = 0
            self.performance_metrics.cost_by_operation[step_type] = 0.0
        
        self.performance_metrics.tokens_by_operation[step_type] += tokens_used
        self.performance_metrics.cost_by_operation[step_type] += cost_estimate
        
        # Track source access
        if source_type:
            if source_type not in self.performance_metrics.sources_accessed:
                self.performance_metrics.sources_accessed[source_type] = 0
            self.performance_metrics.sources_accessed[source_type] += 1
        
        self.current_confidence = confidence_after
        
        logger.debug(f"📝 Added reasoning step: {step_id} - {description[:50]}...")
    
    def log_document_access(self, node: HypergraphNode, access_reason: str, 
                           confidence_score: float, cost_incurred: float = 0.0):
        """Log access to a specific document with context."""
        
        document_path = DocumentPath(
            document_id=node.node_id,
            document_name=node.source_metadata.get('filename', 'Unknown'),
            source_type=node.source_type,
            access_timestamp=time.time() - self.start_time,
            access_reason=access_reason,
            confidence_score=confidence_score,
            content_snippet=node.content[:200] + "..." if len(node.content) > 200 else node.content,
            hierarchical_level=node.hierarchical_level,
            parent_document=node.parent_node_id,
            cost_incurred=cost_incurred
        )
        
        self.document_paths.append(document_path)
        self.performance_metrics.nodes_visited += 1
        
        logger.debug(f"📄 Logged document access: {document_path.document_name}")
    
    def log_decision_point(self, decision_type: str, options: List[str], 
                          chosen_option: str, reasoning: str, confidence_impact: float):
        """Log a critical decision point with alternatives."""
        
        decision_point = {
            'timestamp': time.time() - self.start_time,
            'decision_type': decision_type,
            'options_considered': options,
            'chosen_option': chosen_option,
            'reasoning': reasoning,
            'confidence_impact': confidence_impact,
            'step_number': self.step_counter
        }
        
        self.decision_points.append(decision_point)
        
        logger.debug(f"🔀 Logged decision point: {decision_type} -> {chosen_option}")
    
    def finalize_reasoning(self, final_answer: str, stopping_reason: str):
        """Finalize the reasoning chain with results."""
        
        self.final_answer = final_answer
        total_time = time.time() - self.start_time
        
        # Add final step
        self.add_reasoning_step(
            step_type='completion',
            description=f"Analysis complete: {stopping_reason}",
            details={
                'final_answer_length': len(final_answer),
                'stopping_reason': stopping_reason,
                'total_steps': len(self.reasoning_steps),
                'documents_accessed': len(self.document_paths),
                'decision_points': len(self.decision_points)
            },
            confidence_before=self.current_confidence,
            confidence_after=self.current_confidence,
            decision_rationale=f"Completed analysis with sufficient confidence: {self.current_confidence:.3f}",
            alternatives_considered=["Continue exploration", "Request additional sources"]
        )
        
        # Update final performance metrics
        self.performance_metrics.total_time = total_time
        
        logger.info(f"🏁 Finalized reasoning chain: {len(self.reasoning_steps)} steps, {total_time:.2f}s")
    
    def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get comprehensive reasoning summary."""
        
        # Calculate key metrics
        confidence_gain = (self.current_confidence - 
                          (self.reasoning_steps[0].confidence_after if self.reasoning_steps else 0))
        
        source_diversity = len(self.performance_metrics.sources_accessed)
        
        # Most confident steps
        confident_steps = sorted(
            [step for step in self.reasoning_steps if step.confidence_after > 0.7],
            key=lambda x: x.confidence_after,
            reverse=True
        )[:3]
        
        # Cost efficiency
        cost_per_confidence = (self.performance_metrics.total_cost / 
                              max(self.current_confidence, 0.1))
        
        summary = {
            'chain_id': self.chain_id,
            'query': self.query,
            'final_answer': self.final_answer,
            'total_steps': len(self.reasoning_steps),
            'total_time': self.performance_metrics.total_time,
            'final_confidence': self.current_confidence,
            'confidence_gain': confidence_gain,
            'documents_accessed': len(self.document_paths),
            'sources_used': list(self.performance_metrics.sources_accessed.keys()),
            'source_diversity': source_diversity,
            'total_cost': self.performance_metrics.total_cost,
            'cost_efficiency': cost_per_confidence,
            'decision_points': len(self.decision_points),
            'most_confident_steps': [
                {
                    'description': step.description,
                    'confidence': step.confidence_after,
                    'source': step.source_type
                }
                for step in confident_steps
            ],
            'performance_metrics': asdict(self.performance_metrics)
        }
        
        return summary
    
    def generate_narrative_explanation(self) -> str:
        """Generate human-readable narrative of the reasoning process."""
        
        if not self.reasoning_steps:
            return "No reasoning steps recorded."
        
        narrative_parts = []
        
        # Introduction
        narrative_parts.append(f"**Agentic Analysis of Query: '{self.query}'**\n")
        
        # Process overview
        narrative_parts.append(
            f"I conducted a comprehensive analysis involving {len(self.reasoning_steps)} reasoning steps "
            f"across {len(self.performance_metrics.sources_accessed)} different source types, "
            f"visiting {self.performance_metrics.nodes_visited} nodes in the knowledge graph.\n"
        )
        
        # Key phases
        phases = self._group_steps_by_phase()
        
        for phase_name, steps in phases.items():
            if not steps:
                continue
                
            narrative_parts.append(f"**{phase_name.title()} Phase:**")
            
            if phase_name == 'query_analysis':
                narrative_parts.append(
                    f"I began by analyzing your query to determine the optimal search strategy. "
                    f"Based on the query patterns, I classified this as requiring "
                    f"{steps[0].details.get('query_complexity', 'standard')} analysis."
                )
            
            elif phase_name == 'entry_selection':
                entry_steps = [s for s in steps if 'entry' in s.description.lower()]
                if entry_steps:
                    narrative_parts.append(
                        f"I identified {len(entry_steps)} optimal entry points into the knowledge graph "
                        f"based on semantic similarity to your query."
                    )
            
            elif phase_name == 'exploration':
                confidence_progression = [s.confidence_after for s in steps if s.confidence_after > 0]
                if confidence_progression:
                    avg_confidence = sum(confidence_progression) / len(confidence_progression)
                    narrative_parts.append(
                        f"During exploration, I traversed {len(steps)} nodes with an average "
                        f"confidence of {avg_confidence:.3f}. The most valuable discoveries were "
                        f"from {self._get_most_valuable_sources(steps)}."
                    )
            
            elif phase_name == 'synthesis':
                synthesis_steps = [s for s in steps if 'synthesis' in s.step_type.lower()]
                if synthesis_steps:
                    narrative_parts.append(
                        f"I synthesized findings from multiple sources to provide a comprehensive answer."
                    )
            
            narrative_parts.append("")  # Add spacing
        
        # Key insights
        narrative_parts.append("**Key Insights from Analysis:**")
        
        # Document path insights
        if self.document_paths:
            source_counts = {}
            for doc_path in self.document_paths:
                source_counts[doc_path.source_type] = source_counts.get(doc_path.source_type, 0) + 1
            
            source_descriptions = []
            for source, count in source_counts.items():
                source_descriptions.append(f"{count} {source} documents")
            
            narrative_parts.append(f"- Analyzed content from {', '.join(source_descriptions)}")
        
        # Decision points
        if self.decision_points:
            key_decisions = [dp for dp in self.decision_points if dp['confidence_impact'] > 0.1]
            if key_decisions:
                narrative_parts.append(
                    f"- Made {len(key_decisions)} critical decisions that significantly "
                    f"improved understanding"
                )
        
        # Confidence evolution
        if len(self.performance_metrics.confidence_evolution) > 1:
            initial_conf = self.performance_metrics.confidence_evolution[0][1]
            final_conf = self.performance_metrics.confidence_evolution[-1][1]
            confidence_gain = final_conf - initial_conf
            
            narrative_parts.append(
                f"- Confidence increased from {initial_conf:.3f} to {final_conf:.3f} "
                f"(+{confidence_gain:.3f}) through systematic exploration"
            )
        
        # Efficiency metrics
        narrative_parts.append(
            f"- Completed analysis in {self.performance_metrics.total_time:.2f} seconds "
            f"using {self.performance_metrics.total_tokens_used} tokens"
        )
        
        # Conclusion
        narrative_parts.append(f"\n**Final Answer:**\n{self.final_answer}")
        
        return "\n".join(narrative_parts)
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess the complexity of the query."""
        query_lower = query.lower()
        
        # Count complexity indicators
        complex_indicators = ['compare', 'analyze', 'explain how', 'why', 'relationship', 'impact']
        simple_indicators = ['what is', 'who is', 'when', 'where']
        
        complex_count = sum(1 for indicator in complex_indicators if indicator in query_lower)
        simple_count = sum(1 for indicator in simple_indicators if indicator in query_lower)
        
        if complex_count > simple_count:
            return 'complex'
        elif simple_count > 0:
            return 'simple'
        else:
            return 'moderate'
    
    def _group_steps_by_phase(self) -> Dict[str, List[ReasoningStep]]:
        """Group reasoning steps into logical phases."""
        phases = {
            'query_analysis': [],
            'entry_selection': [],
            'exploration': [],
            'synthesis': [],
            'completion': []
        }
        
        for step in self.reasoning_steps:
            if step.step_type in ['query_analysis']:
                phases['query_analysis'].append(step)
            elif 'entry' in step.step_type or 'entry' in step.description.lower():
                phases['entry_selection'].append(step)
            elif step.step_type in ['node_visit', 'path_decision', 'confidence_check']:
                phases['exploration'].append(step)
            elif 'synthesis' in step.step_type.lower():
                phases['synthesis'].append(step)
            elif step.step_type in ['completion', 'stopping']:
                phases['completion'].append(step)
            else:
                phases['exploration'].append(step)  # Default to exploration
        
        return phases
    
    def _get_most_valuable_sources(self, steps: List[ReasoningStep]) -> str:
        """Identify the most valuable sources from exploration steps."""
        source_values = {}
        
        for step in steps:
            if step.source_type and step.confidence_after > 0.6:
                if step.source_type not in source_values:
                    source_values[step.source_type] = []
                source_values[step.source_type].append(step.confidence_after)
        
        # Calculate average confidence by source
        source_averages = {}
        for source, confidences in source_values.items():
            source_averages[source] = sum(confidences) / len(confidences)
        
        if not source_averages:
            return "various sources"
        
        # Get top sources
        top_sources = sorted(source_averages.items(), key=lambda x: x[1], reverse=True)
        top_source_names = [source for source, _ in top_sources[:2]]
        
        return " and ".join(top_source_names)
    
    def export_audit_trail(self, format_type: str = 'json', 
                          filepath: Optional[str] = None) -> str:
        """Export complete audit trail in specified format."""
        
        if format_type not in self.export_formats:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Generate filename if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"reasoning_chain_{timestamp}.{format_type}"
        
        try:
            if format_type == 'json':
                return self._export_json(filepath)
            elif format_type == 'html':
                return self._export_html(filepath)
            elif format_type == 'markdown':
                return self._export_markdown(filepath)
            elif format_type == 'csv':
                return self._export_csv(filepath)
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
    
    def _export_json(self, filepath: str) -> str:
        """Export audit trail as JSON."""
        export_data = {
            'reasoning_chain': {
                'chain_id': self.chain_id,
                'query': self.query,
                'final_answer': self.final_answer,
                'start_time': self.start_time,
                'total_time': self.performance_metrics.total_time
            },
            'reasoning_steps': [asdict(step) for step in self.reasoning_steps],
            'document_paths': [asdict(path) for path in self.document_paths],
            'decision_points': self.decision_points,
            'performance_metrics': asdict(self.performance_metrics),
            'summary': self.get_reasoning_summary(),
            'export_metadata': {
                'export_time': datetime.now().isoformat(),
                'format': 'json',
                'version': '1.0'
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"✅ Exported audit trail to JSON: {filepath}")
        return filepath
    
    def _export_markdown(self, filepath: str) -> str:
        """Export audit trail as Markdown."""
        
        markdown_content = []
        
        # Header
        markdown_content.append(f"# Reasoning Chain Audit Trail")
        markdown_content.append(f"**Chain ID:** {self.chain_id}")
        markdown_content.append(f"**Query:** {self.query}")
        markdown_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_content.append("")
        
        # Summary
        summary = self.get_reasoning_summary()
        markdown_content.append("## Summary")
        markdown_content.append(f"- **Total Steps:** {summary['total_steps']}")
        markdown_content.append(f"- **Total Time:** {summary['total_time']:.2f} seconds")
        markdown_content.append(f"- **Final Confidence:** {summary['final_confidence']:.3f}")
        markdown_content.append(f"- **Documents Accessed:** {summary['documents_accessed']}")
        markdown_content.append(f"- **Sources Used:** {', '.join(summary['sources_used'])}")
        markdown_content.append(f"- **Total Cost:** ${summary['total_cost']:.4f}")
        markdown_content.append("")
        
        # Narrative explanation
        markdown_content.append("## Reasoning Process")
        markdown_content.append(self.generate_narrative_explanation())
        markdown_content.append("")
        
        # Detailed steps
        markdown_content.append("## Detailed Steps")
        for i, step in enumerate(self.reasoning_steps, 1):
            markdown_content.append(f"### Step {i}: {step.description}")
            markdown_content.append(f"- **Type:** {step.step_type}")
            markdown_content.append(f"- **Timestamp:** {step.timestamp:.2f}s")
            markdown_content.append(f"- **Confidence:** {step.confidence_before:.3f} → {step.confidence_after:.3f}")
            markdown_content.append(f"- **Rationale:** {step.decision_rationale}")
            if step.alternatives_considered:
                markdown_content.append(f"- **Alternatives:** {', '.join(step.alternatives_considered)}")
            if step.tokens_used > 0:
                markdown_content.append(f"- **Tokens Used:** {step.tokens_used}")
            markdown_content.append("")
        
        # Document paths
        if self.document_paths:
            markdown_content.append("## Documents Accessed")
            for doc_path in self.document_paths:
                markdown_content.append(f"### {doc_path.document_name}")
                markdown_content.append(f"- **Source:** {doc_path.source_type}")
                markdown_content.append(f"- **Reason:** {doc_path.access_reason}")
                markdown_content.append(f"- **Confidence:** {doc_path.confidence_score:.3f}")
                markdown_content.append(f"- **Content:** {doc_path.content_snippet}")
                markdown_content.append("")
        
        # Final answer
        markdown_content.append("## Final Answer")
        markdown_content.append(self.final_answer)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(markdown_content))
        
        logger.info(f"✅ Exported audit trail to Markdown: {filepath}")
        return filepath
    
    def _export_csv(self, filepath: str) -> str:
        """Export reasoning steps as CSV."""
        
        steps_data = []
        for step in self.reasoning_steps:
            steps_data.append({
                'step_id': step.step_id,
                'timestamp': step.timestamp,
                'step_type': step.step_type,
                'description': step.description,
                'confidence_before': step.confidence_before,
                'confidence_after': step.confidence_after,
                'tokens_used': step.tokens_used,
                'cost_estimate': step.cost_estimate,
                'decision_rationale': step.decision_rationale,
                'source_type': step.source_type or '',
                'node_id': step.node_id or ''
            })
        
        df = pd.DataFrame(steps_data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"✅ Exported reasoning steps to CSV: {filepath}")
        return filepath
    
    def _export_html(self, filepath: str) -> str:
        """Export audit trail as interactive HTML."""
        
        # Basic HTML template with embedded CSS and JavaScript
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reasoning Chain Audit Trail</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
                .header { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .step { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; background: #fafafa; }
                .step.high-confidence { border-left: 5px solid #4caf50; }
                .step.medium-confidence { border-left: 5px solid #ff9800; }
                .step.low-confidence { border-left: 5px solid #f44336; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #e3f2fd; border-radius: 5px; }
                .document { border: 1px solid #ccc; margin: 5px 0; padding: 10px; border-radius: 3px; }
                .source-text { background: #e3f2fd; }
                .source-visual { background: #e8f5e8; }
                .source-salesforce { background: #fff3e0; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧠 Reasoning Chain Audit Trail</h1>
                    <p><strong>Query:</strong> {query}</p>
                    <p><strong>Chain ID:</strong> {chain_id}</p>
                    <p><strong>Generated:</strong> {timestamp}</p>
                </div>
                
                <div class="summary">
                    <h2>📊 Summary</h2>
                    {summary_metrics}
                </div>
                
                <div class="reasoning-steps">
                    <h2>🔗 Reasoning Steps</h2>
                    {reasoning_steps}
                </div>
                
                <div class="documents">
                    <h2>📄 Documents Accessed</h2>
                    {document_paths}
                </div>
                
                <div class="final-answer">
                    <h2>💡 Final Answer</h2>
                    <div style="background: #f0f8ff; padding: 15px; border-radius: 5px;">
                        {final_answer}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Generate summary metrics HTML
        summary = self.get_reasoning_summary()
        summary_html = ""
        for key, value in summary.items():
            if key not in ['chain_id', 'query', 'final_answer', 'performance_metrics', 'most_confident_steps']:
                summary_html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
        
        # Generate reasoning steps HTML
        steps_html = ""
        for step in self.reasoning_steps:
            confidence_class = "high-confidence" if step.confidence_after > 0.7 else "medium-confidence" if step.confidence_after > 0.4 else "low-confidence"
            
            steps_html += f"""
            <div class="step {confidence_class}">
                <h4>{step.description}</h4>
                <p><strong>Type:</strong> {step.step_type} | <strong>Time:</strong> {step.timestamp:.2f}s | 
                   <strong>Confidence:</strong> {step.confidence_before:.3f} → {step.confidence_after:.3f}</p>
                <p><strong>Rationale:</strong> {step.decision_rationale}</p>
                {f'<p><strong>Alternatives:</strong> {", ".join(step.alternatives_considered)}</p>' if step.alternatives_considered else ''}
            </div>
            """
        
        # Generate document paths HTML
        docs_html = ""
        for doc_path in self.document_paths:
            source_class = f"source-{doc_path.source_type}"
            docs_html += f"""
            <div class="document {source_class}">
                <h4>{doc_path.document_name}</h4>
                <p><strong>Source:</strong> {doc_path.source_type} | <strong>Confidence:</strong> {doc_path.confidence_score:.3f}</p>
                <p><strong>Reason:</strong> {doc_path.access_reason}</p>
                <p><strong>Content:</strong> {doc_path.content_snippet}</p>
            </div>
            """
        
        # Fill template
        html_content = html_template.format(
            query=self.query,
            chain_id=self.chain_id,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            summary_metrics=summary_html,
            reasoning_steps=steps_html,
            document_paths=docs_html,
            final_answer=self.final_answer
        )
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Exported audit trail to HTML: {filepath}")
        return filepath

class PathVisualizer:
    """Creates interactive visualizations of graph traversal paths."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.visualizations_enabled = PLOTLY_AVAILABLE and NETWORKX_AVAILABLE
        
        if not self.visualizations_enabled:
            logger.warning("⚠️ Visualization libraries not available. Install plotly and networkx for full functionality.")
    
    def create_path_visualization(self, paths: List[TraversalPath], 
                                 reasoning_chain: ReasoningChain) -> Optional[go.Figure]:
        """Create interactive visualization of graph traversal paths."""
        
        if not self.visualizations_enabled:
            logger.warning("Visualization not available - missing dependencies")
            return None
        
        try:
            # Create subplot layout
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Graph Traversal Paths', 'Confidence Evolution', 
                               'Source Distribution', 'Performance Metrics'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "pie"}, {"type": "bar"}]]
            )
            
            # 1. Graph traversal paths
            self._add_path_graph(fig, paths, row=1, col=1)
            
            # 2. Confidence evolution
            self._add_confidence_evolution(fig, reasoning_chain, row=1, col=2)
            
            # 3. Source distribution
            self._add_source_distribution(fig, reasoning_chain, row=2, col=1)
            
            # 4. Performance metrics
            self._add_performance_metrics(fig, reasoning_chain, row=2, col=2)
            
            # Update layout
            fig.update_layout(
                title="Graph-R1 Agentic Reasoning Visualization",
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return None
    
    def _add_path_graph(self, fig: go.Figure, paths: List[TraversalPath], row: int, col: int):
        """Add graph traversal path visualization."""
        
        if not paths:
            return
        
        # Use the best path for visualization
        best_path = max(paths, key=lambda p: p.total_confidence)
        
        # Create node positions
        x_coords = list(range(len(best_path.nodes)))
        y_coords = [node.confidence for node in best_path.nodes]
        
        # Color mapping for sources
        color_map = {'text': '#1976d2', 'visual': '#388e3c', 'salesforce': '#f57c00'}
        colors = [color_map.get(node.node.source_type, '#666666') for node in best_path.nodes]
        
        # Add path line
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines+markers',
                line=dict(color='rgba(0,0,0,0.3)', width=2),
                marker=dict(size=12, color=colors, line=dict(width=2, color='white')),
                text=[f"Step {i+1}: {node.node.source_type}<br>Confidence: {node.confidence:.3f}" 
                      for i, node in enumerate(best_path.nodes)],
                hovertemplate='%{text}<extra></extra>',
                name='Reasoning Path'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Reasoning Step", row=row, col=col)
        fig.update_yaxes(title_text="Confidence", range=[0, 1], row=row, col=col)
    
    def _add_confidence_evolution(self, fig: go.Figure, reasoning_chain: ReasoningChain, row: int, col: int):
        """Add confidence evolution over time."""
        
        evolution = reasoning_chain.performance_metrics.confidence_evolution
        if not evolution:
            return
        
        timestamps = [point[0] for point in evolution]
        confidences = [point[1] for point in evolution]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=confidences,
                mode='lines+markers',
                line=dict(color='#2196f3', width=3),
                marker=dict(size=8),
                name='Confidence Evolution'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Time (seconds)", row=row, col=col)
        fig.update_yaxes(title_text="Confidence", range=[0, 1], row=row, col=col)
    
    def _add_source_distribution(self, fig: go.Figure, reasoning_chain: ReasoningChain, row: int, col: int):
        """Add source type distribution pie chart."""
        
        sources = reasoning_chain.performance_metrics.sources_accessed
        if not sources:
            return
        
        fig.add_trace(
            go.Pie(
                labels=list(sources.keys()),
                values=list(sources.values()),
                marker=dict(colors=['#1976d2', '#388e3c', '#f57c00'][:len(sources)]),
                name='Source Distribution'
            ),
            row=row, col=col
        )
    
    def _add_performance_metrics(self, fig: go.Figure, reasoning_chain: ReasoningChain, row: int, col: int):
        """Add performance metrics bar chart."""
        
        metrics = reasoning_chain.performance_metrics
        
        metric_names = ['Query Analysis', 'Graph Construction', 'Traversal', 'Synthesis']
        metric_values = [
            metrics.query_analysis_time,
            metrics.graph_construction_time,
            metrics.traversal_time,
            metrics.synthesis_time
        ]
        
        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker=dict(color=['#2196f3', '#4caf50', '#ff9800', '#9c27b0']),
                name='Time by Phase'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Process Phase", row=row, col=col)
        fig.update_yaxes(title_text="Time (seconds)", row=row, col=col)
    
    def create_confidence_heatmap(self, reasoning_chain: ReasoningChain) -> Optional[go.Figure]:
        """Create heatmap showing confidence across different sources and steps."""
        
        if not self.visualizations_enabled:
            return None
        
        try:
            # Organize steps by source type
            source_steps = {}
            for step in reasoning_chain.reasoning_steps:
                if step.source_type:
                    if step.source_type not in source_steps:
                        source_steps[step.source_type] = []
                    source_steps[step.source_type].append(step.confidence_after)
            
            if not source_steps:
                return None
            
            # Create heatmap data
            sources = list(source_steps.keys())
            max_steps = max(len(steps) for steps in source_steps.values())
            
            # Pad with zeros for equal length
            heatmap_data = []
            for source in sources:
                steps = source_steps[source]
                padded_steps = steps + [0] * (max_steps - len(steps))
                heatmap_data.append(padded_steps)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                y=sources,
                x=[f"Step {i+1}" for i in range(max_steps)],
                colorscale='RdYlBu_r',
                zmin=0,
                zmax=1,
                colorbar=dict(title="Confidence Score")
            ))
            
            fig.update_layout(
                title="Confidence Heatmap by Source Type",
                xaxis_title="Reasoning Steps",
                yaxis_title="Source Types",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Heatmap creation failed: {e}")
            return None

# Factory functions for easy initialization
def create_reasoning_chain(config: Dict[str, Any]) -> ReasoningChain:
    """Factory function to create reasoning chain."""
    logger.info("🔗 Creating interpretable reasoning chain...")
    
    reasoning_chain = ReasoningChain(config)
    logger.info("✅ Reasoning chain created successfully!")
    
    return reasoning_chain

def create_path_visualizer(config: Dict[str, Any]) -> PathVisualizer:
    """Factory function to create path visualizer."""
    logger.info("📊 Creating path visualizer...")
    
    visualizer = PathVisualizer(config)
    logger.info("✅ Path visualizer created successfully!")
    
    return visualizer

# Example usage and testing
if __name__ == "__main__":
    """Test interpretable reasoning chain components."""
    print("🧪 Testing Interpretable Reasoning Chain...")
    print("="*50)
    
    # Test configuration
    test_config = {
        'export_formats': ['json', 'markdown', 'html'],
        'include_visualizations': True
    }
    
    try:
        # Create reasoning chain
        reasoning_chain = create_reasoning_chain(test_config)
        
        # Test reasoning process
        test_query = "How does transformer attention mechanism work?"
        reasoning_chain.start_reasoning(test_query)
        
        # Add some test steps
        reasoning_chain.add_reasoning_step(
            step_type='entry_selection',
            description='Selected optimal entry points into knowledge graph',
            details={'entry_points': 3, 'similarity_threshold': 0.7},
            confidence_before=0.0,
            confidence_after=0.6,
            decision_rationale='High semantic similarity to query',
            alternatives_considered=['Random selection', 'Popularity-based'],
            tokens_used=50,
            cost_estimate=0.001
        )
        
        reasoning_chain.add_reasoning_step(
            step_type='node_visit',
            description='Explored transformer architecture document',
            details={'document': 'transformer_paper.pdf', 'section': 'attention'},
            confidence_before=0.6,
            confidence_after=0.8,
            decision_rationale='Found highly relevant technical details',
            alternatives_considered=['Skip to summary', 'Search other sources'],
            tokens_used=100,
            cost_estimate=0.002,
            source_type='text'
        )
        
        # Finalize
        reasoning_chain.finalize_reasoning(
            "Transformer attention mechanism works by computing attention scores...",
            "High confidence achieved"
        )
        
        # Test export
        json_path = reasoning_chain.export_audit_trail('json')
        markdown_path = reasoning_chain.export_audit_trail('markdown')
        
        # Test visualization
        visualizer = create_path_visualizer(test_config)
        
        print("✅ Reasoning chain test completed successfully!")
        print(f"📄 Exported to: {json_path}")
        print(f"📝 Exported to: {markdown_path}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Note: Full testing requires actual graph traversal data")