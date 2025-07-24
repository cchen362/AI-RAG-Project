# AI-RAG-Project: Multi-Source RAG with Visual Understanding

A sophisticated **Retrieval-Augmented Generation (RAG)** system that combines **Text RAG**, **ColPali visual document understanding**, and **Salesforce integration** with intelligent source selection for enterprise-grade document processing.

## ✨ Key Features

- **🔍 Multi-Source Intelligence**: Text embeddings + ColPali visual understanding + Salesforce knowledge base
- **🖼️ Visual Document Processing**: ColPali processes PDFs directly as images (no OCR required)
- **🧠 BGE Re-ranker**: Intelligent semantic source selection using BAAI/bge-reranker-base
- **⚡ Instant Docker Startup**: Pre-loaded models enable 0-5 second container startup (vs 2-3 minutes)
- **🔥 GPU & CPU Support**: Hardware-optimized deployments with automatic detection
- **🚀 Auto-initialization**: Seamless user experience with loading indicators
- **💰 Token Transparency**: 6-column token breakdown for cost monitoring
- **📊 Production Ready**: Comprehensive health monitoring and container orchestration

## 🚀 Quick Start

### **Option 1: Docker Deployment (Recommended)**
```bash
# CPU-optimized instant startup
docker-compose up ai-rag-app

# GPU-accelerated deployment (requires nvidia-docker)
docker-compose --profile gpu up ai-rag-app-gpu

# Production deployment (detached)
docker-compose up -d ai-rag-app
```
Access at: **http://localhost:8501**

### **Option 2: Local Development**
```bash
# Clone and setup
git clone https://github.com/cchen362/AI-RAG-Project
cd AI-RAG-Project
pip install -r requirements.txt

# Launch application
streamlit run streamlit_rag_app.py
```

### **Option 3: Quick Launcher**
```bash
python launch_app.py
```

## 🏗️ Architecture Overview

### **Multi-Source Query Processing**
```
User Query → SimpleRAGOrchestrator → [Text RAG | ColPali Visual | Salesforce] 
                    ↓
           BGE Cross-Encoder Re-ranker → Single Best Response
```

### **Core Components**
- **SimpleRAGOrchestrator**: Coordinates multi-source search and response generation
- **BGE Cross-Encoder**: Semantic source selection (replaces rule-based intent logic)
- **ColPali Integration**: Visual document understanding using vidore/colqwen2-v1.0
- **Token Counter**: Comprehensive usage tracking for cost transparency

### **Model Specifications**
- **Text Embeddings**: all-MiniLM-L6-v2 (local) or OpenAI text-embedding-ada-002
- **Visual Understanding**: ColQwen2-v1.0 (ColPali engine)
- **Re-ranking**: BAAI/bge-reranker-base
- **Vector Database**: FAISS with metadata support

## 📁 Project Structure

```
AI-RAG-Project/
├── streamlit_rag_app.py          # Main production application
├── launch_app.py                 # Development launcher utility
├── requirements.txt              # Python dependencies  
├── Dockerfile                    # Multi-stage container build
├── docker-compose.yml            # GPU support & orchestration
├── .dockerignore                 # Optimized build context
├── scripts/
│   ├── warm_up_models.py         # Model pre-loading for Docker
│   └── health_check.py           # Container health monitoring
├── src/                          # Core components (7 files)
│   ├── rag_system.py            # Main RAG orchestrator
│   ├── colpali_retriever.py     # Visual document understanding  
│   ├── salesforce_connector.py  # External data integration
│   ├── cross_encoder_reranker.py # Intelligent source selection
│   ├── document_processor.py    # Document processing pipeline
│   ├── text_chunker.py          # Text chunking utilities
│   └── embedding_manager.py     # Embedding management
├── data/documents/              # Document storage
├── cache/embeddings/            # Runtime embedding cache
└── CLAUDE.md                    # Development documentation
```

## 🐳 Docker Optimization

### **Instant Startup Performance**
| Deployment Mode | Startup Time | Model Loading | Memory Usage |
|-----------------|-------------|---------------|--------------|
| Local Development | 2-3 minutes | On-demand | Variable |
| Docker CPU | **0-5 seconds** | Pre-loaded | Optimized |
| Docker GPU | **0-5 seconds** | Pre-loaded | GPU-accelerated |

### **Multi-Stage Architecture**
```dockerfile
Stage 1 (model-builder): Download and cache all AI models
Stage 2 (production): Copy pre-loaded models + application code
Result: Instant startup with full functionality
```

### **Container Variants**
- **ai-rag-app** (Port 8501): CPU-optimized for standard deployments
- **ai-rag-app-gpu** (Port 8502): GPU-accelerated for high-performance processing
- **model-builder**: Development container for model management

### **Health Monitoring**
- Automated health checks every 30 seconds
- Model verification via manifest validation
- Streamlit health endpoint monitoring
- Container restart policies for reliability

## 💡 Usage Examples

### **Multi-Source Queries**
The system automatically queries all sources and selects the best response:

```
Query: "What is the cancellation policy?"

Processing:
✓ Text RAG: Searches uploaded documents
✓ ColPali: Analyzes PDF images for visual information  
✓ Salesforce: Searches knowledge base articles

Result: BGE re-ranker selects best source (e.g., "SALESFORCE" score: 0.847)
Reasoning: "Salesforce KB article provides most comprehensive cancellation policy"
```

### **Visual Document Understanding**
ColPali processes PDFs as images, understanding layout and visual elements:
- Tables, charts, and diagrams
- Multi-column layouts
- Visual hierarchies and formatting
- No OCR pipeline required

### **Token Transparency**
6-column breakdown for cost monitoring:
| Query | VLM | Salesforce | Re-rank | Response | Total |
|-------|-----|------------|---------|----------|-------|
| 12 | 245 | 156 | 10 | 168 | 591 |

## ⚙️ Configuration

### **Environment Variables** (Optional)
```bash
# API Keys (local embeddings work without OpenAI)
OPENAI_API_KEY=your_openai_key

# Salesforce Integration
SALESFORCE_USERNAME=your_username@company.com
SALESFORCE_PASSWORD=your_password
SALESFORCE_SECURITY_TOKEN=your_security_token

# Docker Optimization
DOCKER_PRELOADED_MODELS=true
TRANSFORMERS_CACHE=/app/models/transformers
HF_HOME=/app/models/huggingface
```

### **Hardware Requirements**
- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores, GPU (optional)
- **Docker**: 4GB available to Docker Desktop
- **GPU**: CUDA-compatible for ColPali acceleration

## 🛠️ Development

### **Local Testing**
```bash
# Test individual components
python scripts/health_check.py

# Test model loading
python scripts/warm_up_models.py

# Launch with debugging
streamlit run streamlit_rag_app.py --logger.level=debug
```

### **Docker Development**
```bash
# Build with model pre-loading
docker build -t ai-rag-app .

# Test specific profiles
docker-compose --profile build-models up model-builder
docker-compose --profile gpu up ai-rag-app-gpu

# Monitor health
docker exec ai-rag-cpu python scripts/health_check.py
```

## 📊 Technical Specifications

### **Performance Metrics**
- **Query Processing**: ~2-5 seconds (includes all sources + re-ranking)
- **Document Upload**: ~30-60 seconds per PDF (ColPali processing)
- **Model Loading**: Instant (Docker) vs 2-3 minutes (local)
- **Memory Usage**: ~2-4GB (CPU) / ~6-8GB (GPU)

### **Scalability Features**
- Stateless application design
- Horizontal scaling via container orchestration
- Persistent volume support for data and models
- Health-check based auto-recovery

## ✅ Current Status

**🎯 Production Ready (All Phases Complete)**
- ✅ **Phase 1**: Research & ColPali integration
- ✅ **Phase 2**: Production app with multi-source architecture  
- ✅ **Phase 3**: Docker optimization with instant startup
- ✅ **Cleanup**: Codebase optimization (56+ obsolete files removed)

**🚀 Ready for:**
- Enterprise deployments with instant startup
- Kubernetes orchestration and scaling
- GPU-accelerated cloud environments
- Multi-tenant configurations

## 🤝 Contributing

### **Development Workflow**
```bash
git checkout -b feature/your-feature-name
# Make changes
python scripts/health_check.py  # Verify functionality
git add .
git commit -m "feat: your feature description"
git push -u origin feature/your-feature-name
```

### **Testing Guidelines**
- Test both CPU and GPU configurations
- Verify Docker builds complete successfully
- Ensure health checks pass
- Validate multi-source query processing

## 📄 License

MIT License - See LICENSE file for details

---

**Built with**: Custom Multi-Source RAG Pipeline, ColPali Visual Understanding, BGE Cross-Encoder, OpenAI/Sentence-Transformers, Salesforce API, Streamlit, FAISS, Docker

**Architecture**: SimpleRAGOrchestrator + BGE Re-ranker + ColPali + Instant Docker Startup