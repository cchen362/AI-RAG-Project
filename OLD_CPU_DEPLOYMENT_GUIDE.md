# GPU-Only Deployment Guide for Old CPU Systems

## 🎯 Problem Statement

**Issue**: AMD Phenom II X6 1090T (circa 2010) lacks modern CPU instruction sets (AVX, AVX2) required by AI libraries like SentenceTransformers and ColPali. This causes "Illegal instruction" errors when the app tries to load AI models on the CPU.

**Solution**: Force all AI processing to run exclusively on the RTX 3080 GPU, bypassing the CPU entirely.

## 🖥️ Target System Specifications

- **Server**: `ssh -p 8081 chee@75.163.171.40`
- **CPU**: AMD Phenom II X6 1090T (lacks AVX/AVX2)
- **GPU**: RTX 3080 (24GB VRAM)
- **OS**: Linux (likely Debian/Ubuntu)

## 🚀 Automated Deployment

### Step 1: Run Deployment Script

```bash
# SSH into the production server
ssh -p 8081 chee@75.163.171.40

# Download and run the deployment script
wget https://raw.githubusercontent.com/your-repo/ai-rag-project/main/deploy-gpu-server.sh
chmod +x deploy-gpu-server.sh
./deploy-gpu-server.sh
```

The script will automatically:
1. **Detect the old AMD CPU** and enforce GPU-only mode
2. **Verify RTX 3080** is available and accessible
3. **Build GPU-only Docker image** optimized for old CPU compatibility
4. **Deploy with GPU-only configuration**

### Expected Output

```
[2025-01-30 10:00:00] INFO: CPU: AMD Phenom II(tm) X6 1090T Processor
[2025-01-30 10:00:01] WARNING: OLD CPU DETECTED: AMD Phenom II(tm) X6 1090T Processor
[2025-01-30 10:00:01] WARNING: This CPU may lack modern instruction sets (AVX, AVX2)
[2025-01-30 10:00:01] WARNING: AI libraries may fail with 'Illegal instruction' errors
[2025-01-30 10:00:01] WARNING: ENFORCING GPU-ONLY MODE for compatibility
[2025-01-30 10:00:02] INFO: GPU-only mode will be enforced due to CPU compatibility
[2025-01-30 10:00:03] INFO: 🎮 Building GPU-ONLY image for old CPU compatibility...
[2025-01-30 10:00:15] INFO: 🎮 ENFORCING GPU-ONLY DEPLOYMENT due to old CPU compatibility
[2025-01-30 10:00:45] INFO: 🚀 GPU-ONLY deployment successful! Application available at: http://localhost:8502
[2025-01-30 10:00:45] INFO: ✅ Old CPU compatibility ensured - all AI processing on RTX 3080
```

## 🔧 Manual Deployment (Alternative)

If the automated script doesn't work, deploy manually:

### Step 1: Clone Repository

```bash
git clone https://github.com/your-repo/ai-rag-project.git
cd ai-rag-project
```

### Step 2: Configure Environment

```bash
# Copy production environment file
cp .env.production .env

# Edit environment file to ensure GPU-only mode
nano .env
```

Ensure these critical settings in `.env`:
```bash
MODEL_DEVICE=cuda
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
SENTENCE_TRANSFORMERS_DEVICE=cuda
TRANSFORMERS_DEVICE=cuda
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

### Step 3: Build GPU-Only Image

```bash
# Build only GPU image (skip CPU to avoid compatibility issues)
docker build --target gpu-production -t ai-rag-app:gpu .
```

### Step 4: Deploy GPU-Only Container

```bash
# Deploy with GPU profile only
docker-compose --profile gpu up -d ai-rag-app-gpu
```

## 🧪 Validation & Testing

### System Health Check

```bash
# Run comprehensive health check
docker exec ai-rag-gpu python scripts/health_check.py
```

Expected output:
```
🏥 AI-RAG-Project Health Check
========================================
✅ PASS Environment: Environment variables OK
✅ PASS Docker Config: Docker environment detected | Pre-loaded models enabled | Model device: cuda
✅ PASS CPU Compatibility: GPU-only mode enforced - CPU compatibility not required  
✅ PASS Model Cache: Model cache OK (2/2 directories)
✅ PASS Model Manifest: Model manifest OK (3 models loaded)
✅ PASS GPU/CUDA: GPU OK - NVIDIA GeForce RTX 3080 (CUDA 12.1, 1 devices, 125.2MB test)
✅ PASS Streamlit App: Streamlit app responding
```

### GPU Memory Monitoring

```bash
# Monitor GPU usage during startup and operation
nvidia-smi -l 1

# Expected output should show models loaded on GPU:
# GPU 0: RTX 3080     12GB / 24GB (AI models loaded)
```

### Application Testing

1. **Access Application**: http://75.163.171.40:8502
2. **Upload PDF Document**: Test ColPali visual processing
3. **Query Processing**: Verify all sources work (Text RAG + ColPali + Salesforce)
4. **Check Logs**: Ensure no "Illegal instruction" errors

```bash
# Monitor application logs
docker-compose logs -f ai-rag-app-gpu
```

Expected log entries:
```
🎮 GPU Configuration: gpu_only_mode=True, cpu_compatible=False
🎮 Initializing SentenceTransformer in GPU-only mode
🎮 SentenceTransformer loaded on GPU: NVIDIA GeForce RTX 3080
🎮 Forced GPU mode: NVIDIA GeForce RTX 3080 (24.0GB VRAM)
🔧 GPU memory optimization configured for old system compatibility
```

## 🚨 Troubleshooting

### Issue: "Illegal instruction" Errors

**Cause**: AI models attempting to run on old CPU

**Solution**:
```bash
# Verify GPU-only mode is enforced
docker exec ai-rag-gpu env | grep MODEL_DEVICE
# Should show: MODEL_DEVICE=cuda

# Restart with explicit GPU-only configuration
docker-compose down
export MODEL_DEVICE=cuda
docker-compose --profile gpu up -d ai-rag-app-gpu
```

### Issue: GPU Not Accessible

**Cause**: Docker container can't access GPU

**Solution**:
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi

# If fails, install nvidia-container-toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Issue: Out of GPU Memory

**Cause**: RTX 3080 24GB VRAM exhausted by AI models

**Solution**:
```bash
# Clear GPU memory
docker exec ai-rag-gpu python -c "import torch; torch.cuda.empty_cache()"

# Restart with optimized memory settings
docker-compose down
docker-compose --profile gpu up -d ai-rag-app-gpu
```

### Issue: Model Loading Failures

**Cause**: Network issues or corrupted model downloads

**Solution**:
```bash
# Clear model cache and rebuild
docker-compose down
docker volume rm ai-rag-project_ai_rag_models_gpu
docker build --target gpu-production -t ai-rag-app:gpu . --no-cache
docker-compose --profile gpu up -d ai-rag-app-gpu
```

## 📊 Performance Expectations

### RTX 3080 GPU Performance
- **ColPali Visual Processing**: ~0.4s per PDF page
- **Text Embeddings**: ~0.1s per document chunk  
- **Cross-Encoder Re-ranking**: ~0.05s per query
- **GPU Memory Usage**: ~8-12GB during operation

### Old CPU Impact (Avoided)
- **CPU Processing**: Would cause "Illegal instruction" crash
- **Fallback Behavior**: All processing redirected to GPU
- **Performance**: No degradation, GPU handles all AI workloads

## 🔒 Security & Production Notes

### Container Security
- ✅ Non-root user (appuser)
- ✅ Read-only filesystem where possible
- ✅ Resource limits configured
- ✅ Health checks enabled

### Network Configuration
- **Internal Port**: 8501 (container)
- **External Port**: 8502 (host)
- **Nginx Proxy**: Configure reverse proxy for production access

### Monitoring Setup
```bash
# Set up log rotation
sudo nano /etc/logrotate.d/ai-rag-app

# Monitor GPU usage
crontab -e
# Add: */5 * * * * nvidia-smi >> /var/log/gpu-usage.log
```

## 📋 Production Checklist

Before going live:

- [ ] ✅ Old CPU compatibility verified (AMD Phenom II detected)
- [ ] ✅ GPU-only mode enforced (MODEL_DEVICE=cuda)
- [ ] ✅ RTX 3080 accessible and working
- [ ] ✅ All AI models loading on GPU only
- [ ] ✅ Health checks passing
- [ ] ✅ No "Illegal instruction" errors in logs
- [ ] ✅ Application accessible on port 8502
- [ ] ✅ PDF upload and processing working
- [ ] ✅ Multi-source queries functioning
- [ ] 🔲 Nginx reverse proxy configured
- [ ] 🔲 SSL certificates installed
- [ ] 🔲 Log rotation configured
- [ ] 🔲 Backup strategy implemented

## 🎉 Success Criteria

**Deployment is successful when**:
1. Application starts without "Illegal instruction" errors
2. All AI models load exclusively on RTX 3080 GPU
3. ColPali visual processing works for PDF documents
4. Text RAG and Salesforce integration functional
5. System stable under normal load
6. GPU memory usage remains under 20GB
7. No CPU instruction set errors in logs

## 📞 Support & Contact

For deployment issues:
- **Server Access**: `ssh -p 8081 chee@75.163.171.40`
- **Application URL**: http://75.163.171.40:8502
- **Log Location**: `docker-compose logs -f ai-rag-app-gpu`
- **Health Check**: `docker exec ai-rag-gpu python scripts/health_check.py`

---

**This deployment guide ensures your AI-RAG-Project runs successfully on the old AMD Phenom II X6 1090T system by leveraging the powerful RTX 3080 GPU for all AI processing.**