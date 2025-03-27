import gradio as gr
import os
import pandas as pd
from dotenv import load_dotenv
import time
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
                print(f"Error getting embedding: {e}")
                embeddings.append([0.0] * 1536)
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
            print(f"Error getting embedding: {e}")
            return [0.0] * 1536

def chunk_text(text, max_length=50):
    """Split text into chunks at sentence boundaries"""
    sentences = []
    current_sentence = []
    
    for word in text.split():
        current_sentence.append(word)
        if word.endswith(('.', '!', '?', '...')):
            sentences.append(' '.join(current_sentence))
            current_sentence = []
    
    if current_sentence:
        sentences.append(' '.join(current_sentence))
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_length and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if not chunk_text.strip().endswith('.'):
                chunk_text = chunk_text.rstrip('.!?') + '.'
            chunks.append(chunk_text)
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
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

# Initialize the knowledge base
folder_path = 'knowledge_sources'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Created {folder_path} directory as it did not exist")

documents = load_documents(folder_path)
if not documents:
    print("No documents found in knowledge_sources directory")
    documents = [{"content": "No documents available.", "metadata": {"source": "empty", "page": 1}}]

print(f"{len(documents)} documents loaded")
print("FAISS indexing...")
start_time = time.time()

# Create FAISS index
embeddings = OpenRouterEmbeddings(HEADERS)
faiss_index = FAISS.from_texts(
    [doc['content'] for doc in documents], 
    embedding=embeddings,
    metadatas=[doc['metadata'] for doc in documents]
)
print(f"FAISS indexing done in {time.time() - start_time} seconds")

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
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
        print(f"Error getting chat completion: {e}")
        return None

def chat(message, history):
    """Handle chat messages"""
    # Get relevant documents
    docs = faiss_index.similarity_search(message, k=3)
    
    # Prepare context from documents
    context = "\n".join([doc.page_content for doc in docs])
    
    # Prepare messages for the API
    messages = [
        {"role": "system", "content": "You are HiringHelp, an AI assistant that helps with hiring-related questions. Use the provided context to answer questions about candidates and hiring."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}"}
    ]
    
    # Get response from API
    response = get_chat_completion(messages)
    
    if response and "choices" in response:
        answer = response["choices"][0]["message"]["content"]
        
        # Add sources if available
        sources = [{"title": doc.metadata["source"], "page_number": doc.metadata["page"]} for doc in docs]
        if sources:
            answer += "\n\nSources:\n" + "\n".join([f"- {source['title']} (Page {source['page_number']})" for source in sources])
        
        return answer
    else:
        return "I apologize, but I encountered an error while processing your request. Please try again."

# Create Gradio interface
with gr.Blocks(css="styles.css") as demo:
    gr.Markdown("# HiringHelp Chatbot")
    gr.Markdown("Ask me anything about candidates and hiring!")
    
    chatbot = gr.Chatbot(
        value=[["HiringHelp", "Hello, how can I help you today?"]],
        height=600,
        show_label=False
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Your message",
            placeholder="Type your message here...",
            show_label=False,
            container=False
        )
        submit = gr.Button("Send", variant="primary")
    
    # Example questions
    gr.Markdown("### Try these example questions:")
    with gr.Row():
        gr.Button("List all the available candidates")
        gr.Button("Tell me about a candidate named Kristy Natasha Yohanes")
        gr.Button("Which candidate is best for an AI Engineer role?")
    
    # Handle message submission
    submit.click(chat, [msg, chatbot], [chatbot])
    msg.submit(chat, [msg, chatbot], [chatbot])
    
    # Handle example questions
    for btn in demo.children[3].children[0].children:
        btn.click(lambda x: msg.update(x), btn, msg)

# Launch the app
demo.launch() 