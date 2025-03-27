from flask import Flask, request, jsonify, send_from_directory
from logging.config import dictConfig
from models import get_db_connection, init_db
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

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.embeddings.base import Embeddings
from PyPDF2 import PdfReader

# Load environment variables from .env file
load_dotenv()

# OpenRouter API configuration
API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
    "HTTP-Referer": "https://github.com/natgluons/RAG-Chatbot",
    "X-Title": "HiringHelp Chatbot",
    "Content-Type": "application/json"
}
MODEL = "qwen/qwen-2-7b-instruct:free"

class OpenRouterEmbeddings(Embeddings):
    def __init__(self, headers):
        self.headers = headers

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using OpenRouter API"""
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    headers=self.headers,
                    json={
                        "model": "text-embedding-ada-002",
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
                "https://openrouter.ai/api/v1/embeddings",
                headers=self.headers,
                json={
                    "model": "text-embedding-ada-002",
                    "input": text
                }
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            app.logger.error(f"Error getting embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1536

# Rate limiter settings
MAX_REQUESTS_PER_MINUTE = 10
MAX_REQUESTS_PER_DAY = 100
MINUTE_WINDOW = 60  # seconds
DAY_WINDOW = 24 * 60 * 60  # seconds (24 hours)
minute_timestamps = deque()
day_timestamps = deque()
rate_limit_lock = Lock()

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

app = Flask(__name__, static_folder='..', static_url_path='')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DOCS_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'knowledge_sources')
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'csv'}

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["10 per minute", "100 per day"]
)

def get_chat_completion(messages):
    """Get chat completion using OpenRouter API"""
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": 100,
                "temperature": 0.7
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        app.logger.error(f"Error getting chat completion: {e}")
        return None

init_db()

def chunk_text(text, max_length=50):
    """Split text into chunks at sentence boundaries, ensuring no sentence is cut off and each chunk ends with a period"""
    # Split text into sentences (handle common sentence endings)
    sentences = []
    current_sentence = []
    
    for word in text.split():
        current_sentence.append(word)
        
        # Check if this word ends with a sentence-ending punctuation
        if word.endswith(('.', '!', '?', '...')):
            # Join the current sentence and add it to sentences
            sentences.append(' '.join(current_sentence))
            current_sentence = []
    
    # Add any remaining words as a sentence
    if current_sentence:
        sentences.append(' '.join(current_sentence))
    
    # Group sentences into chunks
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        # If adding this sentence would exceed max_length, start a new chunk
        if current_length + sentence_length > max_length and current_chunk:
            # Ensure the chunk ends with a period
            chunk_text = ' '.join(current_chunk)
            if not chunk_text.strip().endswith('.'):
                chunk_text = chunk_text.rstrip('.!?') + '.'
            chunks.append(chunk_text)
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the last chunk if it exists
    if current_chunk:
        # Ensure the last chunk ends with a period
        chunk_text = ' '.join(current_chunk)
        if not chunk_text.strip().endswith('.'):
            chunk_text = chunk_text.rstrip('.!?') + '.'
        chunks.append(chunk_text)
    
    return chunks

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                text_chunks = chunk_text(file.read())
                for chunk in text_chunks:
                    documents.append({
                        "content": chunk,
                        "metadata": {
                            "source": filename,
                            "page": 1
                        }
                    })
        elif filename.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_chunks = chunk_text(text)
                    for chunk in text_chunks:
                        documents.append({
                            "content": chunk,
                            "metadata": {
                                "source": filename,
                                "page": page_num + 1
                            }
                        })
        elif filename.endswith(".csv"):
            df = pd.read_csv(file_path)
            text_chunks = chunk_text(df.to_string())
            for chunk in text_chunks:
                documents.append({
                    "content": chunk,
                    "metadata": {
                        "source": filename,
                        "page": 1
                    }
                })
    return documents

folder_path = 'knowledge_sources'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    app.logger.warning(f"Created {folder_path} directory as it did not exist")

documents = load_documents(folder_path)
if not documents:
    app.logger.warning("No documents found in knowledge_sources directory")
    documents = [{"content": "No documents available.", "metadata": {"source": "empty", "page": 1}}]

app.logger.info(f"{len(documents)} documents loaded")
app.logger.info("FAISS indexing...")
start_time = time.time()

# Create FAISS index using the custom embedding class
embeddings = OpenRouterEmbeddings(HEADERS)
faiss_index = FAISS.from_texts(
    [doc['content'] for doc in documents], 
    embedding=embeddings,
    metadatas=[doc['metadata'] for doc in documents]
)
app.logger.info(f"FAISS indexing done in {time.time() - start_time} seconds")

retriever = faiss_index.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.7
    }
)

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="result"
)

@app.route('/chat', methods=['POST'])
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
            for filename in os.listdir(folder_path):
                if filename.startswith("CV_") and filename.endswith(".pdf"):
                    # Keep the name exactly as it appears in the filename
                    name = filename[3:-4]  # Remove "CV_" prefix and ".pdf" extension
                    names.add(name)
            # Format the response with a specific template
            result = f"Here is a list of all the available candidates: ({', '.join(sorted(names))})"
            
            # For listing candidates, we don't need sources
            return jsonify({"response": result})
        
        # For specific candidate queries, load their PDF directly
        filtered_docs = []
        target_filename = None
        
        # Check if the prompt contains a specific candidate name
        for filename in os.listdir(folder_path):
            if filename.startswith("CV_") and filename.endswith(".pdf"):
                candidate_name = filename[3:-4]  # Remove "CV_" prefix and ".pdf" extension
                if candidate_name.lower() in prompt.lower():
                    target_filename = filename
                    # Load the specific candidate's PDF
                    file_path = os.path.join(folder_path, filename)
                    reader = PdfReader(file_path)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            text_chunks = chunk_text(text)
                            for chunk in text_chunks:
                                filtered_docs.append({
                                    "content": chunk,
                                    "metadata": {
                                        "source": filename,
                                        "page": page_num + 1
                                    }
                                })
                    break
        
        # If no specific candidate was found, use all relevant docs
        if not filtered_docs:
            # For queries about best candidates, use all documents
            if "best" in prompt.lower() and "candidate" in prompt.lower():
                for filename in os.listdir(folder_path):
                    if filename.startswith("CV_") and filename.endswith(".pdf"):
                        file_path = os.path.join(folder_path, filename)
                        reader = PdfReader(file_path)
                        for page_num, page in enumerate(reader.pages):
                            text = page.extract_text()
                            if text:
                                text_chunks = chunk_text(text)
                                for chunk in text_chunks:
                                    filtered_docs.append({
                                        "content": chunk,
                                        "metadata": {
                                            "source": filename,
                                            "page": page_num + 1
                                        }
                                    })
            else:
                # For other queries, use the retriever
                relevant_docs = retriever.get_relevant_documents(prompt)
                filtered_docs = relevant_docs
        
        if not filtered_docs:
            return jsonify({"error": "No relevant information found"}), 404
        
        context = "\n".join([doc['content'] for doc in filtered_docs])
        # Prepare messages for chat completion with strict instructions
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides concise answers in plain text without any formatting symbols. Keep responses to maximum 2 sentences and always end with a period."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
        ]
        
        # Get chat completion with reduced max_tokens
        response = get_chat_completion(messages)
        
        if response and "choices" in response:
            result = response["choices"][0]["message"]["content"]
            # Remove any remaining star symbols
            result = result.replace('*', '')
            # Ensure the response ends with a period and doesn't end with prepositions/conjunctions
            result = result.strip()
            
            # List of prepositions and conjunctions to check
            prepositions = {'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'around', 'as', 'at', 
                          'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'by', 'concerning', 
                          'considering', 'despite', 'down', 'during', 'except', 'excluding', 'for', 'from', 'in', 'inside', 
                          'into', 'like', 'minus', 'near', 'of', 'off', 'on', 'onto', 'opposite', 'outside', 'over', 'past', 
                          'per', 'plus', 'regarding', 'round', 'save', 'since', 'than', 'through', 'to', 'toward', 'towards', 
                          'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'versus', 'via', 'with', 'within', 'without'}
            
            conjunctions = {'for', 'and', 'nor', 'but', 'or', 'yet', 'so', 'after', 'although', 'as', 'as if', 'as long as', 
                          'as soon as', 'because', 'before', 'even if', 'even though', 'if', 'in order that', 'once', 'since', 
                          'so that', 'than', 'that', 'though', 'unless', 'until', 'when', 'whenever', 'where', 'whereas', 
                          'whether', 'while'}
            
            # Remove any sentence-ending punctuation
            result = result.rstrip('.!?')
            
            # Split into words and check the last word
            words = result.split()
            if words:
                last_word = words[-1].lower()
                # If the last word is a preposition or conjunction, add a complete thought
                if last_word in prepositions or last_word in conjunctions:
                    result = result + " is the case."
            
            # Add period if not present
            if not result.endswith('.'):
                result = result + '.'
        else:
            raise Exception("Failed to get response from OpenRouter API")
        
        # Deduplicate sources by document title
        unique_sources = {}
        for doc in filtered_docs:
            source_title = doc['metadata']["source"]
            if source_title not in unique_sources:
                unique_sources[source_title] = {
                    "title": source_title,
                    "page_number": doc['metadata']["page"]
                }

        return jsonify({
            "response": result,
            "sources": list(unique_sources.values())
        })
        
    except Exception as e:
        print("\n=== Error Occurred ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('..', 'index.html')

@app.route('/app.js')
def app_js():
    return send_from_directory('..', 'app.js')

@app.route('/docs')
def serve_docs():
    docs_path = app.config['DOCS_FOLDER']
    files = []
    try:
        for filename in os.listdir(docs_path):
            if filename.endswith(('.pdf', '.txt', '.csv')):
                files.append({
                    'name': filename,
                    'path': f'/docs/file/{filename}'
                })
    except Exception as e:
        app.logger.error(f"Error accessing docs folder: {e}")
        return "Error accessing documentation", 500

    # Return a simple HTML page listing all documents
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>HiringHelp Documentation</title>
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                max-width: 800px;
                margin: 40px auto;
                padding: 20px;
                background-color: #f5f7fa;
            }
            h1 {
                color: #433b6b;
                text-align: center;
            }
            .file-list {
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .file-item {
                padding: 10px;
                border-bottom: 1px solid #eee;
            }
            .file-item:last-child {
                border-bottom: none;
            }
            a {
                color: #433b6b;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <h1>📚 Available Candidate's Resume</h1>
        <div class="file-list">
    '''
    
    for file in files:
        html += f'<div class="file-item"><a href="{file["path"]}">{file["name"]}</a></div>'
    
    html += '''
        </div>
    </body>
    </html>
    '''
    
    return html

@app.route('/docs/file/<filename>')
def serve_file(filename):
    if not filename.endswith(('.pdf', '.txt', '.csv')):
        return "File type not allowed", 403
    try:
        return send_from_directory(app.config['DOCS_FOLDER'], filename)
    except Exception as e:
        app.logger.error(f"Error serving file {filename}: {e}")
        return "File not found", 404

@app.route('/api-check')
def api_check():
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        use_openrouter = os.getenv("USE_OPENROUTER")
        
        return jsonify({
            "message": f"OpenRouter API key found (starts with: {api_key[:8]}...), USE_OPENROUTER={use_openrouter}"
        })
    except Exception as e:
        return jsonify({
            "message": f"Error checking API configuration: {str(e)}"
        })

@app.route('/test-api')
def test_api():
    try:
        # First test if the API key is valid
        if not os.environ.get('OPENROUTER_API_KEY'):
            return jsonify({"success": False, "error": "API key is missing"})
            
        # Try making a simple request to the API
        messages = [
            {"role": "user", "content": "Hello, can you hear me?"}
        ]
        
        response = get_chat_completion(messages)
        
        if response and "choices" in response:
            return jsonify({
                "success": True,
                "response": response["choices"][0]["message"]["content"]
            })
        else:
            return jsonify({"success": False, "error": "Invalid response structure"})
            
    except Exception as e:
        return jsonify({
            "success": False, 
            "error": str(e),
            "error_type": type(e).__name__
        })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
