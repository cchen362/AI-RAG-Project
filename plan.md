# Graph-R1 Agentic RAG Implementation Plan

## Problem Analysis
Based on user feedback, the current implementation has critical limitations:

1. **Insufficient agentic demonstration**: Test app only uses text_rag source, hiding multi-source intelligence
2. **Double synthesis issue**: "From text_rag" outputs indicate unprofessional double processing  
3. **Missing interpretable reasoning**: No clear audit trail of document paths and decisions
4. **Missing Graph-R1 architecture**: No hypergraph representation or graph traversal

## User Requirements
> "We are looking for interpretable, auditable retrieval. A true agentic system logs:
> - The path it took,
> - Which documents it touched and why,
> - How it decided to stop searching."

> "Agentic Graph Construction + Planner + Retriever Setup from Graph‑R1:
> - Build a multi-source hypergraph: Nodes (embeddings of KB chunks, PDF sections, image captions, titles), Edge types (semantic similarity, source-type link, hierarchy)
> - Use a trained planner or lightweight LLM agent to: Decide the next hop (KB or PDF?), Prune irrelevant node paths early
> - Do graph traversal with budgeted retrieval: Limit hops, source types, or chunk sizes dynamically based on query type or confidence
> - Then only pass top N candidates to BGE for final reranking"

## Solution: True Graph-R1 Implementation

### Phase 1: Unified Hypergraph Architecture (Week 1-2)

**1.1 Cross-Modal Embedding Unification**
- Implement learnable linear projection (128D→512D) for ColPali embeddings
- Create unified vector database combining text, visual, and Salesforce content
- Preserve source provenance while enabling cross-modal similarity search

**1.2 Hypergraph Construction**
- **Nodes**: Document chunks (512D), PDF sections (128D→512D), image captions, Salesforce articles
- **Edges**: Semantic similarity, hierarchical relationships (article→section→paragraph), source-type links
- **Metadata**: Source attribution, confidence scores, retrieval costs, document paths

**1.3 Graph-Aware Vector Database**
- Replace separate retrieval systems with unified graph-based search
- Implement MaxSim scoring for visual content, cosine similarity for text
- Add graph traversal capabilities with budgeted retrieval

### Phase 2: Graph Traversal Retrieval Engine (Week 2-3)

**2.1 LLM-Driven Path Planning**
- Replace rule-based source selection with dynamic LLM planning
- Implement query analysis for optimal graph entry points
- Add confidence-based path pruning and early stopping

**2.2 Interpretable Reasoning Chain**
- Log exact document nodes visited and selection rationale
- Track graph traversal paths with visual representation
- Record stopping decisions with confidence thresholds
- Audit trail: path taken, documents touched, stopping logic

**2.3 Budgeted Retrieval System**
- Dynamic hop limits based on query complexity
- Source-type balancing (visual vs text vs business)
- Cost-aware traversal with token budget management
- Prune irrelevant node paths early to optimize performance

### Phase 3: Enhanced Test Interface (Week 3-4)

**3.1 Multi-Source Demonstration App**
- Build comprehensive test interface showcasing all three sources (Text + ColPali + Salesforce)
- Add graph visualization of traversal paths
- Implement side-by-side baseline vs Graph-R1 comparison
- Show clear agentic advantages with multi-source intelligence

**3.2 Interpretable Audit Trail**
- Real-time reasoning chain display
- Document path visualization with decision points
- Performance metrics and cost tracking
- Professional output without "From text_rag" annotations

### Phase 4: Production Integration (Week 4-5)

**4.1 Streamlit App Enhancement**
- Add Graph-R1 agentic mode toggle
- Integrate graph visualization components
- Maintain existing glassmorphism UI design

**4.2 Performance Monitoring**
- A/B testing framework for baseline vs agentic
- Cost analysis and efficiency metrics
- User preference tracking

## Technical Architecture

### Core Components

1. **`src/hypergraph_constructor.py`**
   - CrossModalProjector: Learnable 128D→512D transformation for ColPali
   - UnifiedEmbeddingSpace: Single vector database for all modalities
   - HypergraphBuilder: Node and edge construction with metadata
   - ValidationSuite: Embedding quality and similarity preservation tests

2. **`src/graph_traversal_engine.py`**
   - LLMPathPlanner: Dynamic query analysis and traversal planning
   - GraphTraverser: Budgeted retrieval with hop limits and pruning
   - ConfidenceManager: Dynamic stopping criteria and path evaluation
   - ReasoningLogger: Complete audit trail of decisions and paths

3. **`src/interpretable_reasoning_chain.py`**
   - ReasoningChain: Step-by-step decision logging
   - PathVisualizer: Graph traversal visualization
   - AuditTrail: Complete document path and selection rationale
   - PerformanceMetrics: Cost, timing, and efficiency tracking

4. **`test_graph_r1_demo.py`**
   - Multi-source demonstration app
   - Side-by-side baseline vs Graph-R1 comparison
   - Interactive reasoning chain visualization
   - Professional output formatting

### Implementation Tasks

#### Immediate Tasks (Current Session)
- [x] Create `plan.md` - Document complete implementation plan  
- [ ] Build `src/hypergraph_constructor.py` - Unified embedding space
- [ ] Create `src/graph_traversal_engine.py` - LLM-driven graph search
- [ ] Build `test_graph_r1_demo.py` - Multi-source demonstration
- [ ] Update development journal with approved plan

#### Next Session Tasks
- [ ] Create `src/interpretable_reasoning_chain.py`
- [ ] Integrate Graph-R1 mode into main Streamlit app
- [ ] Add graph visualization components
- [ ] Implement A/B testing framework
- [ ] Performance optimization and monitoring

## Expected Outcomes

### Immediate Benefits
- **True agentic behavior**: Dynamic graph traversal vs fixed source selection
- **Interpretable reasoning**: Clear audit trail of paths and decisions  
- **Multi-source intelligence**: Demonstrable advantages across text, visual, and business content
- **Professional output**: Clean synthesis without source annotations

### Long-term Impact
- **Production readiness**: Seamless integration with existing architecture
- **Scalability**: Graph-based architecture supports hundreds of documents + 800+ SF articles
- **Auditability**: Complete transparency for enterprise deployment
- **Cost efficiency**: Budgeted retrieval optimizes token usage

## Success Metrics

1. **Agentic Demonstration**: Clear showing of multi-source reasoning advantages
2. **Reasoning Transparency**: Complete audit trail of document paths and decisions
3. **Performance**: Faster retrieval with higher confidence scores
4. **User Experience**: Professional output with interpretable reasoning chains
5. **Cost Efficiency**: Optimized token usage through budgeted graph traversal

---

**Status**: ✅ Plan Approved - Implementation Started
**Next Phase**: Unified Hypergraph Architecture Development
**Target**: True Graph-R1 agentic reasoning with interpretable audit trails