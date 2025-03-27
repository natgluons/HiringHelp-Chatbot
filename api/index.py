from flask import Flask, request, jsonify, send_from_directory, send_file
from logging.config import dictConfig
import os
import pandas as pd
from dotenv import load_dotenv
import time
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from collections import deque
from threading import Lock
import requests
import json
from typing import List
from werkzeug.middleware.proxy_fix import ProxyFix
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from lib.blob_storage import BlobStorage

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.embeddings.base import Embeddings
from PyPDF2 import PdfReader

# Initialize Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["10 per minute", "100 per day"]
)

# Load environment variables from .env file
load_dotenv()

# OpenRouter API configuration
API_URL = "https://openrouter.ai/api/v1/chat/completions"
EMBEDDING_URL = "https://api.openai.com/v1/embeddings"  # OpenAI compatible endpoint
HEADERS = {
    "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
    "HTTP-Referer": "https://github.com/natgluons/RAG-Chatbot",
    "X-Title": "HiringHelp Chatbot",
    "Content-Type": "application/json"
}
MODEL = "qwen/qwen-2-7b-instruct:free"
EMBEDDING_MODEL = "openai/text-embedding-ada-002"  # OpenAI compatible model

class OpenRouterEmbeddings(Embeddings):
    def __init__(self, headers):
        self.headers = headers

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using OpenRouter API"""
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    EMBEDDING_URL,
                    headers=self.headers,
                    json={
                        "model": EMBEDDING_MODEL,
                        "input": text
                    }
                )
                response.raise_for_status()
                embedding = response.json()["data"][0]["embedding"]
                embeddings.append(embedding)
            except Exception as e:
                app.logger.error(f"Error getting embedding: {e}")
                # Return a zero vector as fallback
                embeddings.append([0.0] * 1536)  # OpenAI embeddings are 1536-dimensional
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a single text using OpenRouter API"""
        try:
            response = requests.post(
                EMBEDDING_URL,
                headers=self.headers,
                json={
                    "model": EMBEDDING_MODEL,
                    "input": text
                }
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            app.logger.error(f"Error getting embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1536

# Configure folder paths
KNOWLEDGE_SOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'knowledge_sources')

@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    print("\n=== Chat Request Started ===")
    print("Checking OpenRouter API configuration...")
    print(f"OpenRouter API Key present: {bool(os.environ.get('OPENROUTER_API_KEY'))}")
    print(f"Using model: {MODEL}")
    
    try:
        data = request.json
        prompt = data['prompt']
        print(f"\nReceived prompt: {prompt}")
        
        # For listing candidates, extract names from PDF filenames
        if "list" in prompt.lower() and "candidate" in prompt.lower():
            # Get all names directly from the knowledge_sources directory
            names = set()
            for filename in os.listdir(KNOWLEDGE_SOURCES_DIR):
                if filename.startswith("CV_") and filename.endswith(".pdf"):
                    # Keep the name exactly as it appears in the filename
                    name = filename[3:-4]  # Remove "CV_" prefix and ".pdf" extension
                    names.add(name)
            # Format the response with a specific template
            result = f"Here is a list of all the available candidates: ({', '.join(sorted(names))})"
            
            # For listing candidates, we don't need sources
            return jsonify({"response": result})
        
        # ... rest of the function code ...
        
    except Exception as e:
        print("\n=== Error Occurred ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/cv/<filename>')
def serve_cv(filename):
    """Serve CV files from the knowledge_sources directory"""
    try:
        # Security check: ensure filename only contains safe characters
        if '..' in filename or filename.startswith('/'):
            return jsonify({"error": "Invalid filename"}), 400
            
        return send_from_directory(KNOWLEDGE_SOURCES_DIR, filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/cvs')
def list_cvs():
    """List all available CV files"""
    try:
        cvs = []
        for filename in os.listdir(KNOWLEDGE_SOURCES_DIR):
            if filename.startswith('CV_') and filename.endswith('.pdf'):
                name = filename[3:-4]  # Remove "CV_" prefix and ".pdf" extension
                cvs.append({
                    'name': name,
                    'filename': filename,
                    'url': f'/cv/{filename}'
                })
        return jsonify({
            'cvs': sorted(cvs, key=lambda x: x['name']),
            'message': 'These are sample CVs for demonstration purposes only.'
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('../public', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 