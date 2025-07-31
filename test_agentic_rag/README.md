# Agentic RAG Testing Framework

A lightweight testing environment for Graph-R1 inspired multi-turn reasoning enhancements to the AI-RAG-Project.

## Overview

This framework implements a **test-first approach** to validate agentic RAG concepts before integration into the main production system. It provides:

- 🤖 **Agentic Orchestrator**: Graph-R1 inspired "think-retrieve-rethink-generate" reasoning
- 🧠 **Agent Memory**: Conversation state and knowledge persistence  
- 🧪 **Test Harness**: Simple interface for comparing approaches
- 📊 **Evaluation Metrics**: Comprehensive performance comparison
- 🎯 **Test Scenarios**: 15 complex test cases designed to showcase agentic benefits

## Quick Start

### 1. Interactive Testing Mode
```bash
cd test_agentic_rag
python test_harness.py --mode interactive
```

### 2. Run Test Suite
```bash
python test_harness.py --mode suite --scenarios test_scenarios.json
```

### 3. Single Query Test
```bash
python test_harness.py --mode single --query "Your test query here"
```

## Architecture

### Core Components

- **`agentic_orchestrator.py`**: Main agentic reasoning engine
  - Multi-turn "think-retrieve-rethink-generate" loops
  - Intelligent source selection and orchestration  
  - Confidence-based stopping criteria
  - Reuses existing production components (RAGSystem, ColPali, Salesforce)

- **`agent_memory.py`**: Conversation state management
  - Cross-turn context preservation
  - Knowledge fragment accumulation
  - Query pattern learning
  - Persistent memory across sessions

- **`test_harness.py`**: Testing and comparison framework
  - Side-by-side agentic vs baseline comparison
  - Interactive testing mode
  - Automated test suite execution
  - Results persistence and analysis

- **`evaluation_metrics.py`**: Performance evaluation system
  - Multi-dimensional scoring (relevance, completeness, accuracy, coherence, efficiency)
  - Comparative analysis and recommendations
  - Aggregate improvement metrics

### Test Scenarios

The `test_scenarios.json` file contains 15 carefully designed test cases:

1. **Simple Technical Query** - Baseline comparison
2. **Visual Content Query** - ColPali visual analysis
3. **Business Context Query** - Salesforce integration
4. **Complex Multi-hop Query** - Multi-source reasoning
5. **Progressive Query Refinement** - Memory and context usage
6. **Cross-Source Intelligence** - Intelligent orchestration
7. **Visual + Technical Integration** - Multi-modal analysis
8. **Ambiguous Query Resolution** - Context understanding
9. **Resource Optimization Query** - Business + technical synthesis
10. **Comparative Analysis** - Complex analytical reasoning
11. **Error Handling Test** - Edge case handling
12. **Context-Dependent Follow-up** - Conversation flow
13. **Detailed Technical Deep-dive** - Knowledge depth
14. **Real-world Application** - Practical implementation guidance
15. **Time-Sensitive Query** - Current information handling

## Expected Benefits

Based on the Graph-R1 analysis, this framework should demonstrate:

- **200-400% improvement** on complex multi-hop queries
- **New capabilities**: Cross-source intelligence, query decomposition, context preservation
- **30-50% improvement** in source selection accuracy
- **20-50% improvement** in resource efficiency through adaptive optimization

## Usage Examples

### Interactive Mode Commands
- `memory` - Show conversation summary
- `clear` - Clear agent memory
- `quit` - Exit interactive mode

### Configuration
The system automatically detects available components:
- ✅ Text RAG (usually available)
- ⚠️ ColPali Visual (requires GPU/CPU setup)
- ⚠️ Salesforce (requires credentials)

### Sample Output
```
🔍 Testing Query: Based on performance charts, analyze business requirements and recommend optimal configuration

🤖 AGENTIC APPROACH:
✅ Answer: Based on my multi-source analysis:
   **Visual Analysis**: Found 3 relevant performance charts showing model comparisons
   **Technical Information**: Detailed analysis of transformer architectures and performance metrics
   **Business Context**: Current deployment constraints and resource requirements
⏱️ Time: 4.2s
🔗 Steps: 4
📊 Confidence: 0.89
🔍 Sources: colpali_visual, text_rag, salesforce

📝 BASELINE APPROACH (Single-turn):
✅ Answer: Transformer architectures are neural networks that use attention mechanisms...
⏱️ Time: 1.8s
📊 Confidence: 0.65
📄 Chunks: 5

⚖️ COMPARISON:
🏆 Agentic provides comprehensive multi-source analysis vs single-source baseline
```

## Integration Strategy

This framework is designed for **risk-free validation**:

1. **Phase 1** (Current): Validate core concepts in isolated environment
2. **Phase 2**: Optimize performance and add advanced features  
3. **Phase 3**: Integrate proven capabilities into main Streamlit app

The framework reuses all existing production components, ensuring that validated enhancements can be seamlessly integrated without disrupting the current system.

## Next Steps

1. Run interactive tests to validate basic functionality
2. Execute test suite to measure performance improvements
3. Analyze results to identify most beneficial use cases
4. Optimize based on findings
5. Plan integration into main application

This test-first approach ensures that only proven, beneficial enhancements make it into the production system while maintaining the excellent stability and performance of the current AI-RAG-Project.