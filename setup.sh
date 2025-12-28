#!/bin/bash

# Setup script for Medical GraphRAG System

echo "🚀 Setting up Medical GraphRAG System..."

# Create directories
echo "📁 Creating directories..."
mkdir -p data cache logs

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create .env file if not exists
if [ ! -f .env ]; then
    echo "⚙️ Creating .env file..."
    cat > .env << EOF
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:7b

# Weaviate Configuration (if using)
WEAVIATE_URL=http://localhost:8080

# Logging
LOG_LEVEL=INFO
EOF
    echo "✅ Created .env file"
fi

# Download embedding model
echo "🧠 Downloading Vietnamese embedding model (this may take a while)..."
python3 << 'PYTHON_EOF'
from sentence_transformers import SentenceTransformer

try:
    model = SentenceTransformer('AITeamVN/Vietnamese_Embedding')
    print("✅ Embedding model downloaded successfully")
except Exception as e:
    print(f"⚠️ Error downloading embedding model: {e}")
    print("You can download it manually later or it will be downloaded on first use")
PYTHON_EOF

echo ""
echo "✅ Setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Start Neo4j database:"
echo "   docker run -d --name neo4j -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password123 neo4j:latest"
echo ""
echo "2. Start Ollama server:"
echo "   ollama serve"
echo ""
echo "3. Pull Qwen model (in another terminal):"
echo "   ollama pull qwen2.5:7b"
echo ""
echo "4. Process data:"
echo "   python data_ingestion.py"
echo ""
echo "5. Build knowledge graph:"
echo "   python graph_builder.py"
echo ""
echo "6. Start chatbot:"
echo "   chainlit run chainlit_app.py"
echo ""
