#!/usr/bin/env bash
# Quick start script for RAG Chatbot
set -e

echo ""
echo "🤖 RAG Chatbot — Quick Start"
echo "=============================="

# Check Python
python3 --version || { echo "❌ Python 3 not found"; exit 1; }

# Create virtualenv
if [ ! -d "venv" ]; then
  echo "→ Creating virtual environment..."
  python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install deps
echo "→ Installing dependencies..."
pip install -q -r requirements.txt

# Check .env
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ""
  echo "⚠️  Created .env from template."
  echo "    Edit .env and add your OPENAI_API_KEY, then run this script again."
  exit 0
fi

# Start server
echo ""
echo "→ Starting RAG Chatbot API on http://localhost:8000"
echo "→ Open frontend/index.html in your browser"
echo ""
echo "Press Ctrl+C to stop."
echo ""
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
