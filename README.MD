# AI-RAG-Project: Multi-Source RAG with Visual Understanding

A sophisticated **Retrieval-Augmented Generation (RAG)** system that combines **Text RAG**, **ColPali visual document understanding**, and **Salesforce integration** with intelligent source selection for enterprise-grade document processing.

## ✨ Key Features

- **🔍 Multi-Source Intelligence**: Text embeddings + ColPali visual understanding + Salesforce knowledge base
- **🖼️ Visual Document Processing**: ColPali processes PDFs directly as images (no OCR required)
- **🧠 BGE Re-ranker**: Intelligent semantic source selection using BAAI/bge-reranker-base
- **⚡ Instant Docker Startup**: Pre-loaded models enable 0-5 second container startup (vs 2-3 minutes)
- **🔥 GPU & CPU Support**: Hardware-optimized deployments with automatic detection
- **🚀 Auto-initialization**: Seamless user experience with loading indicators
- **🎨 Modern Glassmorphism UI**: Vibrant glassmorphic design with dynamic backgrounds and interactive elements
- **🌍 Cross-Platform**: Windows, Linux, macOS support with intelligent poppler detection
- **🛡️ Graceful Degradation**: Text RAG continues working when visual processing unavailable
- **📋 System Status**: Real-time feature availability indicators in UI
- **💰 Token Transparency**: 7-column token breakdown for cost monitoring
- **📊 Production Ready**: Comprehensive health monitoring and container orchestration

## 🚀 Quick Start

### **Option 1: Quick Launcher (Recommended for Development)**
```bash
# Clone and setup
git clone https://github.com/cchen362/AI-RAG-Project
cd AI-RAG-Project
pip install -r requirements.txt

# One-command launch with automatic setup
python launch_app.py
```

### **Option 2: Docker Deployment (Recommended for Production)**
```bash
# CPU-optimized instant startup
docker-compose up ai-rag-app

# GPU-accelerated deployment (requires nvidia-docker)
docker-compose --profile gpu up ai-rag-app-gpu

# Production deployment (detached)
docker-compose up -d ai-rag-app
```
Access at: **http://localhost:8501**

### **Option 3: Manual Local Development**
```bash
# Traditional streamlit launch
streamlit run streamlit_rag_app.py
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

## 🎨 Modern UI/UX Features (July 2025)

### **Glassmorphism Design System**
- **Dynamic Backgrounds**: Animated gradient backgrounds with glassmorphic overlays
- **Interactive Elements**: Hover effects, color transitions, and visual feedback
- **Glass Containers**: Backdrop blur effects with transparency and vibrant borders
- **Responsive Layout**: Mobile-friendly design with adaptive sidebar navigation
- **Enhanced Typography**: Glassmorphic text effects with improved visual hierarchy

### **User Experience Enhancements**
- **Auto-clearing Forms**: Query fields clear automatically after submission
- **Real-time Status**: Live system capability indicators (Text ✅, Visual ✅/⚠️, Salesforce ✅/❌)
- **Enhanced Formatting**: Structured responses with bullet points and proper line breaks
- **Visual Feedback**: Color-coded confidence indicators and loading animations
- **Development Launcher**: `launch_app.py` utility for streamlined development workflow

## 📁 Project Structure

```
AI-RAG-Project/
├── streamlit_rag_app.py          # Main production application
├── launch_app.py                 # Development launcher utility
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Multi-stage container build
├── docker-compose.yml            # GPU support & orchestration
├── deploy-gpu-server.sh          # GPU server deployment script
├── .dockerignore                 # Optimized build context
├── scripts/
│   ├── warm_up_models.py         # Model pre-loading for Docker
│   └── health_check.py           # Container health monitoring
├── src/                          # Core components (8 files)
│   ├── rag_system.py            # Main RAG orchestrator
│   ├── colpali_retriever.py     # Visual document understanding  
│   ├── salesforce_connector.py  # External data integration
│   ├── cross_encoder_reranker.py # Intelligent source selection
│   ├── document_processor.py    # Document processing pipeline
│   ├── visual_document_processor.py # Visual document processing
│   ├── text_chunker.py          # Text chunking utilities
│   └── embedding_manager.py     # Embedding management
├── tests/                       # Test suite
│   ├── test_comprehensive.py    # Comprehensive system tests
│   └── test_salesforce.py       # Salesforce integration tests
├── data/documents/              # Document storage
├── cache/embeddings/            # Runtime embedding cache
├── evaluation_results/          # Performance metrics and analysis
└── CLAUDE.md                    # Development documentation
```

## 🌍 Cross-Platform Support

### **Platform Compatibility**
The system automatically detects and adapts to different operating systems:

| Platform | Poppler Detection | Visual Processing | Fallback Behavior |
|----------|------------------|-------------------|-------------------|
| **Linux/Docker** | `/usr/bin`, `/usr/local/bin` | ✅ Full Support | Text RAG continues |
| **Windows** | Program Files, Conda paths | ✅ Full Support | Text RAG continues |
| **macOS** | Homebrew, MacPorts | ✅ Full Support | Text RAG continues |

### **Graceful Degradation**
When poppler is unavailable, the system provides clear feedback and continues operating:

```
⚠️ Visual Processing Status:
✅ Text RAG: Available 
⚠️ Visual RAG (ColPali): Limited (Poppler unavailable)
✅ Salesforce: Connected
✅ Re-ranker: Active

💡 Install poppler-utils for visual document processing
```

### **System Requirements**
- **Required**: Python 3.9+, basic dependencies
- **Optional**: Poppler-utils (for visual processing)
- **Recommended**: GPU support for optimal performance

### **Installation Commands**
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# Windows (Conda)
conda install -c conda-forge poppler

# macOS (Homebrew)  
brew install poppler

# Docker (automatically included)
# No additional setup required
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

## 🖥️ Linux Server Deployment

### **Production Server Setup**
```bash
# 1. Deploy container with proper port mapping
docker run -d \
  --name ai-rag-production \
  --gpus all \
  -p 3015:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/cache:/app/cache \
  --env-file .env \
  ai-rag-app:gpu

# 2. Configure nginx reverse proxy (example)
server {
    listen 80;
    server_name ragtest.zyroi.com;
    
    location / {
        proxy_pass http://172.19.0.1:3015;  # Docker gateway IP
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### **Environment Variable Override Strategy**
**Critical for production deployments** - Use override pattern to ensure environment variables take precedence over baked-in container values:

```python
# In src/salesforce_connector.py and similar files
from dotenv import load_dotenv
load_dotenv("/app/.env", override=True)  # override=True is crucial
```

### **Container Replacement vs Modification**
**Best Practice**: Create new containers rather than modifying existing ones:
```bash
# ✅ Recommended: Create new container
docker run -d --name ai-rag-new-version ...

# ❌ Avoid: Modifying running containers
# (leads to inconsistent state and hard-to-debug issues)
```

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
7-column breakdown for cost monitoring:
| Query | Text RAG | VLM | Salesforce | Re-rank | Response | Total |
|-------|----------|-----|------------|---------|----------|-------|
| 12 | 45 | 245 | 156 | 10 | 168 | 636 |

## ⚙️ Configuration

### **Environment Variables** (Optional)
```bash
# API Keys (local embeddings work without OpenAI)
OPENAI_API_KEY=your_openai_key

# Salesforce Integration
SALESFORCE_USERNAME=your_username@company.com
SALESFORCE_PASSWORD=your_password
SALESFORCE_SECURITY_TOKEN=your_security_token

# Cross-Platform Poppler Support (optional)
POPPLER_PATH=/custom/poppler/bin

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
- ✅ **Phase 4**: Modern UI/UX with glassmorphism design system
- ✅ **Cleanup**: Codebase optimization (45+ obsolete files removed)

**🚀 Ready for:**
- Enterprise deployments with instant startup and modern UI
- Kubernetes orchestration and scaling
- GPU-accelerated cloud environments with glassmorphic interface
- Multi-tenant configurations with professional user experience

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

## 🔧 Troubleshooting

### **Common Issues**

#### **Visual Processing Unavailable**
```
⚠️ Visual RAG (ColPali): Limited (Poppler unavailable)
```
**Solution**: Install poppler-utils for your platform:
- Linux: `sudo apt-get install poppler-utils`  
- Windows: `conda install -c conda-forge poppler`
- macOS: `brew install poppler`
- Docker: Already included

#### **OpenAI API Key Warning** 
```
⚠️ OpenAI API key not found - visual analysis will be limited
```
**Solution**: Create `.env` file with:
```bash
OPENAI_API_KEY=your_key_here
```

#### **Slow Performance on CPU**
**Symptoms**: ColPali processing taking 30+ seconds per page
**Solution**: 
- Use GPU deployment: `docker-compose --profile gpu up ai-rag-app-gpu`
- Or accept CPU limitations with reduced page counts

#### **Docker Build Fails**
**Common causes**:
- Insufficient disk space (requires ~10GB)
- Network issues during model download
- CUDA version mismatch (GPU builds)

**Solution**:
```bash
# Clean Docker cache
docker system prune -a

# Rebuild with verbose output  
docker-compose build --no-cache --progress=plain
```

#### **Health Check Failures**
**Check status**:
```bash
# View container logs
docker logs ai-rag-cpu

# Manual health check
docker exec ai-rag-cpu python scripts/health_check.py
```

#### **System Initialization Failed (Production)**
**Symptoms**: "System initialization failed" error when processing documents
**Root Cause**: Invalid/truncated OpenAI API key baked into container layers
**Solution**:
```bash
# 1. Create new container with fresh environment
docker run -d \
  --name ai-rag-clean-new \
  --gpus all \
  -p 3016:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/cache:/app/cache \
  --env-file .env \
  -e OPENAI_API_KEY=your_full_valid_key_here \
  ai-rag-app:gpu

# 2. Update nginx proxy to point to new container
# proxy_pass http://172.19.0.1:3016;
```

#### **Nginx 502/504 Gateway Errors**
**Symptoms**: Bad Gateway errors when accessing via domain
**Root Cause**: Nginx cannot reach Docker container due to networking issues
**Solutions**:
```bash
# Option 1: Use Docker gateway IP (most reliable)
proxy_pass http://172.19.0.1:3015;

# Option 2: Use container IP (if gateway fails)
# Find container IP: docker inspect <container_name> | grep IPAddress
proxy_pass http://172.19.0.2:8501;

# Option 3: Use host network (development only)
docker run --network host ...
```

#### **Salesforce Authentication Still Shows Placeholder**
**Symptoms**: Shows "your_salesforce_username" despite setting environment variables
**Root Cause**: Baked-in environment variables not being overridden by .env file
**Solution**: Modify connector to force override:
```python
# In src/salesforce_connector.py
load_dotenv("/app/.env", override=True)  # Add override=True
```

#### **Docker Container Network Discovery**
**Find Docker gateway IP**:
```bash
# Method 1: Gateway IP
docker network inspect bridge | grep Gateway

# Method 2: Container inspection
docker inspect <container> | grep Gateway

# Method 3: From inside container
ip route | grep default
```

### **Environment Variable Reference**
| Variable | Purpose | Required |
|----------|---------|----------|
| `OPENAI_API_KEY` | Enhanced visual analysis | Optional |
| `POPPLER_PATH` | Custom poppler location | Optional |
| `SALESFORCE_*` | Knowledge base integration | Optional |
| `DOCKER_PRELOADED_MODELS` | Enable instant startup | Auto-set |

## 📄 License

MIT License - See LICENSE file for details

---

**Built with**: Custom Multi-Source RAG Pipeline, ColPali Visual Understanding, BGE Cross-Encoder, OpenAI/Sentence-Transformers, Salesforce API, Streamlit, FAISS, Docker, Modern Glassmorphism UI

**Architecture**: SimpleRAGOrchestrator + BGE Re-ranker + ColPali + Instant Docker Startup + Modern UI/UX