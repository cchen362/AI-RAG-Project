# Docker Deployment Guide

## Prerequisites
- Docker and Docker Compose installed
- Clone this repository

## Quick Start

### Option 1: Using Docker Compose (Recommended)
```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### Option 2: Using Docker directly
```bash
# Build the image
docker build -t rag-app .

# Run the container
docker run -d -p 8501:8501 --name rag-app rag-app

# Stop and remove
docker stop rag-app && docker rm rag-app
```

## Environment Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your API keys (optional):
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   SALESFORCE_USERNAME=your_salesforce_username
   SALESFORCE_PASSWORD=your_salesforce_password
   SALESFORCE_SECURITY_TOKEN=your_salesforce_security_token
   ```

**Note**: The application works with local embeddings by default - no API keys required!

## Accessing the Application

- Open your browser and go to: http://localhost:8501
- The application should load quickly with the optimized imports

## Features Available in Docker

✅ **Document Upload & Processing**
- Upload PDF, TXT, DOCX files
- Automatic text chunking and embedding

✅ **Smart Search**
- Semantic search using local embeddings
- No API key required for basic functionality

✅ **Salesforce Integration** (optional)
- Requires Salesforce credentials in environment

✅ **Data Persistence**
- Document cache stored in `./cache` volume
- Documents stored in `./data` volume

## Troubleshooting

### Container won't start
```bash
docker-compose logs ai-rag-app
```

### Port already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8502:8501"  # Use different external port
```

### Clear cache
```bash
# Remove cache volume
docker-compose down
sudo rm -rf ./cache/*
docker-compose up -d
```

## Production Deployment

For production, consider:
1. Using environment-specific docker-compose files
2. Setting up reverse proxy (nginx)
3. Enabling HTTPS
4. Monitoring and logging
5. Backup strategies for data volumes

## Performance Notes

- **Fast Startup**: Heavy ML imports moved to lazy loading
- **Memory Efficient**: Uses lightweight sentence transformers by default
- **Scalable**: Can switch to OpenAI embeddings for production scale