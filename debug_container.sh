#!/bin/bash
# Container Diagnosis Script for AI-RAG System Initialization Error
# Run this script on your Debian server as: bash debug_container.sh

echo "🔍 AI-RAG Container Diagnosis Script"
echo "===================================="
echo

# Check system info
echo "📋 System Information:"
uname -a
echo

# Check Docker service status
echo "🐳 Docker Service Status:"
sudo systemctl status docker --no-pager -l
echo

# List all containers
echo "📦 Docker Containers:"
sudo docker ps -a
echo

# Find the AI-RAG container
AI_RAG_CONTAINER=$(sudo docker ps -a --format "table {{.Names}}\t{{.Status}}" | grep -i "ai-rag\|rag" | head -1 | awk '{print $1}')

if [ -z "$AI_RAG_CONTAINER" ]; then
    echo "❌ No AI-RAG container found. Looking for any running containers..."
    AI_RAG_CONTAINER=$(sudo docker ps --format "{{.Names}}" | head -1)
fi

if [ ! -z "$AI_RAG_CONTAINER" ]; then
    echo "🎯 Found container: $AI_RAG_CONTAINER"
    echo
    
    # Check container logs
    echo "📄 Container Logs (last 100 lines):"
    echo "=================================="
    sudo docker logs $AI_RAG_CONTAINER --tail 100
    echo
    
    # Check container environment
    echo "🌍 Container Environment Variables:"
    echo "================================="
    sudo docker exec $AI_RAG_CONTAINER env | grep -E "(MODEL|CUDA|GPU|TRANSFORM|HF_|TORCH|OPENAI)" | sort
    echo
    
    # Check GPU availability in container
    echo "🚀 GPU Status in Container:"
    echo "========================="
    sudo docker exec $AI_RAG_CONTAINER nvidia-smi 2>/dev/null || echo "❌ No GPU access in container"
    echo
    
    # Check Python packages
    echo "📚 Python Package Status:"
    echo "========================"
    sudo docker exec $AI_RAG_CONTAINER pip list | grep -E "(torch|transformers|sentence|colpali|streamlit)" | sort
    echo
    
    # Check model cache directories
    echo "💾 Model Cache Status:"
    echo "===================="
    sudo docker exec $AI_RAG_CONTAINER ls -la /app/models/ 2>/dev/null || echo "❌ No /app/models directory"
    sudo docker exec $AI_RAG_CONTAINER du -sh /app/models/* 2>/dev/null || echo "❌ Model cache empty or inaccessible"
    echo
    
    # Check memory usage
    echo "💡 Container Resource Usage:"
    echo "==========================="
    sudo docker stats $AI_RAG_CONTAINER --no-stream
    echo

else
    echo "❌ No containers found!"
fi

echo "🔧 Next Steps:"
echo "============="
echo "1. Check the logs above for specific error messages"
echo "2. Verify model cache directories have proper permissions"
echo "3. Ensure GPU is accessible if using GPU mode"
echo "4. Check if models failed to download due to network issues"
echo