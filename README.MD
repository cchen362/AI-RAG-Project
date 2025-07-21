# AI RAG Project - AI Knowledge Assignment

A smart document assistant that combines **Retrieval-Augmented Generation (RAG)** with Salesforce integration to provide intelligent document search and question-answering capabilities.

## ✨ Features

- **📄 Multi-format Document Support**: PDF, DOCX, TXT, CSV, XLSX
- **🏢 Salesforce Integration**: Real-time search of Knowledge Articles
- **🧠 Intent-Driven Search**: Automatically detects travel-related queries (hotel, air, car operations)
- **🚀 Transformative Semantic Search**: Advanced AI-powered search with fallback mechanisms
- **💬 Two Web Interfaces**: Streamlit (full-featured) and Gradio (simplified)
- **🎯 Smart Source Selection**: Automatically chooses between local documents and Salesforce based on query intent

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (optional, for enhanced search)
- Salesforce credentials (optional, for knowledge article search)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd AI-RAG-Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment** (optional)
   Create a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   SALESFORCE_USERNAME=your_username@company.com
   SALESFORCE_PASSWORD=your_password
   SALESFORCE_SECURITY_TOKEN=your_security_token
   ```

4. **Launch the application**
   
   **Option A: Direct Python (Development)**
   ```bash
   python launch_app.py
   ```
   
   **Option B: Docker (Recommended for Production)**
   ```bash
   docker-compose up -d
   ```
   
   Access at: http://localhost:8501

## 📁 Project Structure

```
AI-RAG-Project/
├── src/                     # Core RAG system components
│   ├── rag_system.py        # Main RAG orchestrator
│   ├── salesforce_connector.py  # Salesforce integration
│   ├── semantic_enhancer.py     # Advanced search capabilities
│   └── document_processor.py    # Document handling
├── streamlit_rag_app.py     # Full-featured web interface
├── gradio_interface.py      # Alternative simple interface
├── launch_app.py            # Application launcher
├── requirements.txt         # Python dependencies
└── docs/                    # Documentation and guides
```

## 🎯 How It Works

1. **Intent Detection**: Analyzes your question to understand what you're asking about
2. **Smart Routing**: 
   - Travel queries (cancel hotel, modify flight) → Salesforce search
   - General document questions → Local document search
3. **Enhanced Search**: Uses transformative semantic search for deeper understanding
4. **Quality Results**: Only returns results above 70% relevance threshold

## 🛠️ Usage Examples

### Document Sources
1. **Sample Documents**: Use the provided documentation in the `docs/` folder
2. **Upload Your Own**: Support for PDF, DOCX, TXT, CSV, XLSX files via the web interface. **Remember to select "Process Documents"!**
3. **Salesforce Integration**: Real-time search of Knowledge Articles (if configured)

### Example Questions
**For uploaded documents:**
- "Who is in charge of recruitment?"
- "What are the project requirements?"
- "Summarize the key points from the documentation"

**For Salesforce integration (travel/service scenarios):**
- "How do I cancel a hotel booking for a late arrival?"
- "What to do when a flight passenger no-shows?"
- "Modify air reservation procedures"
- "Handle car rental customer complaints"

## 🔧 Configuration

The system automatically detects query intent and selects the appropriate search method. You can also force specific search strategies in the advanced options.

## 🐳 Docker Deployment

### Quick Start with Docker
```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or build manually
docker build -t rag-app .
docker run -p 8501:8501 rag-app
```

### Environment Setup
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` with your API keys (optional - app works with local embeddings)

### Docker Features
- ✅ **Fast startup** - Optimized imports for quick loading
- ✅ **No API keys required** - Works with local embeddings by default
- ✅ **Data persistence** - Volumes for cache and documents
- ✅ **Production ready** - Includes restart policies

For detailed Docker instructions, see [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)

## 📊 Current Status

- ✅ Basic RAG system implemented
- ✅ Streamlit and Gradio interfaces
- ✅ Salesforce integration with intent recognition
- ✅ Transformative semantic search
- ✅ Docker containerization completed
- ✅ Production deployment ready

## 🤝 Contributing

Please create a feature branch for any new features or improvements:
```bash
git checkout -b feature-name
# Make changes
git add .
git commit -m "Add new feature"
git push -u origin feature-name
```

## 📄 License

MIT License - See LICENSE file for details

---

**Built with**: Custom RAG Pipeline, OpenAI/Sentence-Transformers, Salesforce API, Streamlit, FAISS, Docker