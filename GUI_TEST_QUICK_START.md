# GUI Testing Interface - Quick Start Guide

## 🚀 Launch the Interface

### Option 1: Using the Launcher (Recommended)
```bash
cd C:\Users\cchen362\OneDrive\Desktop\AI-RAG-Project
python launch_gui_test.py
```

### Option 2: Direct Streamlit Command
```bash
cd C:\Users\cchen362\OneDrive\Desktop\AI-RAG-Project
streamlit run gui_test_interface.py
```

## 🖥️ Interface Overview

The GUI will open in your browser at `http://localhost:8501` with these features:

### **Test Controls (Sidebar)**
- **Test Mode Selection**: Choose from 4 testing approaches
- **System Status**: Real-time component availability 
- **Configuration**: Adjust reasoning parameters

### **Main Interface Modes**

#### 1. **Interactive Testing** 
- Enter custom queries in the text area
- Click "Run Agentic Test" to see real-time reasoning
- Watch step-by-step reasoning chain unfold
- Compare agentic vs baseline responses

#### 2. **Predefined Scenarios**
- Choose from 5 pre-built test scenarios:
  - Simple Technical Query
  - Attention Mechanism Deep Dive  
  - Multi-hop Complex Query
  - Visual Content Query
  - Business Context Query

#### 3. **Performance Analysis**
- View charts comparing reasoning approaches
- Analyze execution times and step counts
- Monitor confidence scores and source usage

#### 4. **Memory Inspection**
- Examine agent conversation history
- View knowledge fragments stored in memory
- Track learning patterns over time

## 🧠 Real-Time Reasoning Visualization

When you run a test, you'll see:

### **Reasoning Chain Display**
- **THINK** (Green): Initial query analysis and planning
- **RETRIEVE** (Blue): Source querying and information gathering  
- **RETHINK** (Orange): Result analysis and decision making
- **GENERATE** (Purple): Final answer synthesis

### **Performance Metrics**
- **Execution Time**: Total processing duration
- **Reasoning Steps**: Number of agent reasoning cycles
- **Final Confidence**: Agent's confidence in the answer
- **Sources Used**: Number of different sources queried

### **Side-by-Side Comparison** (if enabled)
- **Agentic Response**: Multi-turn reasoning result
- **Baseline Response**: Single-turn traditional RAG result
- **Metrics Comparison**: Performance differences highlighted

## 🛠️ Troubleshooting

### **If the interface fails to initialize:**
1. Check that you're in the project root directory
2. Ensure all dependencies are installed:
   ```bash
   pip install streamlit plotly pandas
   ```
3. Verify test components exist:
   ```bash
   ls test_agentic_rag/
   ```

### **If agentic tests fail:**
- The system will show detailed error information
- Check the error details in the interface
- Most issues are related to missing API keys or component initialization

### **If no sources are available:**
- Text RAG requires documents in `data/documents/` folder
- Salesforce requires proper credentials in `.env` file
- ColPali is disabled by default for faster testing

## 💡 Testing Tips

### **Best Queries to Try:**
1. **Technical**: "What is attention mechanism in transformers?"
2. **Complex**: "How do transformers compare to RNNs for language modeling?"
3. **Business**: "What are the latest AI trends for business applications?"

### **What to Watch For:**
- **Step Count**: Agentic should use 6-10 steps vs baseline's 1 step
- **Confidence Changes**: Watch how confidence evolves through reasoning
- **Source Selection**: See which sources the agent chooses and why
- **Response Quality**: Compare depth and comprehensiveness

### **Configuration Tweaks:**
- **Max Steps**: Increase for more thorough reasoning (10-15)
- **Confidence Threshold**: Lower for more exploration (0.5-0.6)
- **Enable Comparison**: Always keep ON to see the difference

## 🎯 Key Features Demonstrated

### **Agentic Reasoning Transparency**
- See exactly how the AI thinks step-by-step
- Understand source selection decisions
- Track confidence evolution through reasoning

### **Performance Analysis**
- Compare multi-turn vs single-turn approaches
- Quantify reasoning overhead vs quality gains
- Identify optimal configurations for different query types

### **System Debugging**
- Real-time error reporting and diagnosis
- Component status monitoring
- Memory inspection and conversation tracking

---

## 🚀 Ready to Test!

Launch the interface and start with the **"Attention Mechanism Deep Dive"** predefined scenario to see the full power of agentic reasoning visualization!

The interface showcases the breakthrough capabilities you've achieved with the Graph-R1 inspired multi-turn reasoning system.