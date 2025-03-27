import gradio as gr
import os
import pandas as pd
from dotenv import load_dotenv
import time
from collections import deque
from threading import Lock
import requests
import json
import numpy as np
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.embeddings.base import Embeddings
# from PyPDF2 import PdfReader

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
    def __init__(self, api_url: str, headers: dict, model: str):
        self.api_url = api_url
        self.headers = headers
        self.model = model
        self.dimension = 1536  # Standard embedding dimension

    def _get_embedding(self, text: str) -> List[float]:
        try:
            # Construct a prompt that asks the model to create an embedding
            prompt = f"Please create a numerical embedding representation of this text. Respond with only numbers: {text}"
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 100
                }
            )
            response.raise_for_status()
            
            # Get the response text and convert it to a numerical embedding
            text_response = response.json()["choices"][0]["message"]["content"]
            
            # Generate a deterministic embedding based on the text response
            text_bytes = text_response.encode('utf-8')
            np.random.seed(sum(text_bytes))
            embedding = np.random.normal(0, 1/np.sqrt(self.dimension), self.dimension)
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * self.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts"""
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        return self._get_embedding(text)

# Initialize embeddings with OpenRouter
embeddings = OpenRouterEmbeddings(API_URL, HEADERS, MODEL)

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
    # for filename in os.listdir(folder_path):
    #     file_path = os.path.join(folder_path, filename)
    #     if filename.endswith(".txt"):
    #         with open(file_path, 'r', encoding='utf-8') as file:
    #             text_chunks = chunk_text(file.read())
    #             for chunk in text_chunks:
    #                 documents.append({
    #                     "content": chunk,
    #                     "metadata": {
    #                         "source": filename,
    #                         "page": 1
    #                     }
    #                 })
    #     elif filename.endswith(".pdf"):
    #         reader = PdfReader(file_path)
    #         for page_num, page in enumerate(reader.pages):
    #             text = page.extract_text()
    #             if text:
    #                 text_chunks = chunk_text(text)
    #                 for chunk in text_chunks:
    #                     documents.append({
    #                         "content": chunk,
    #                         "metadata": {
    #                             "source": filename,
    #                             "page": page_num + 1
    #                         }
    #                     })
    #     elif filename.endswith(".csv"):
    #         df = pd.read_csv(file_path)
    #         text_chunks = chunk_text(df.to_string())
    #         for chunk in text_chunks:
    #             documents.append({
    #                 "content": chunk,
    #                 "metadata": {
    #                     "source": filename,
    #                     "page": 1
    #                 }
    #             })
    
    # Process text files from the main directory
    if os.path.exists(folder_path) and os.listdir(folder_path):
        print(f"Processing text files from {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text_content = file.read()
                        text_chunks = chunk_text(text_content)
                        for chunk in text_chunks:
                            documents.append({
                                "content": chunk,
                                "metadata": {
                                    "source": filename,
                                    "page": 1
                                }
                            })
                except Exception as e:
                    print(f"Error processing text file {filename}: {e}")
                    continue
    
    if not documents:
        print("No text files found in the directory")
        return []
    
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
    try:
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
            sources = [{"title": doc.metadata["source"], "page": doc.metadata["page"]} for doc in docs]
            if sources:
                answer += "\n\nSources:\n" + "\n".join([f"- {source['title']} (Page {source['page']})" for source in sources])
            
            # Update history and return
            history.append((message, answer))
            return history, ""  # Return updated history and clear the textbox
        else:
            history.append((message, "I apologize, but I encountered an error while processing your request. Please try again."))
            return history, ""
    except Exception as e:
        print(f"Error in chat: {e}")
        history.append((message, "An error occurred. Please try again."))
        return history, ""

def create_demo():
    with gr.Blocks() as demo:  # Remove css parameter
        gr.Markdown("# 🤖 HiringHelp Chatbot")
        gr.Markdown("""
        Ask me anything about the candidates or hiring-related topics. I'll help you find the right information!
        
        Example questions:
        - List all available candidates
        - Tell me about a specific candidate
        - Which candidate is best for a particular role?
        """)
        
        chatbot = gr.Chatbot(
            show_label=False,
            height=600,
            container=True,
        )
        
        with gr.Row():
            txt = gr.Textbox(
                show_label=False,
                placeholder="Type your message here...",
                container=True
            )
            submit_btn = gr.Button("Send", variant="primary")
        
        with gr.Row():
            clear_btn = gr.Button("Clear Chat")
        
        # Example questions as buttons
        with gr.Row():
            example_questions = [
                "List all available candidates",
                "Tell me about Sleepy Panda",
                "Which candidate is best for a UI/UX Designer role?",
                "What are Tall Giraffe's skills?",
                "Show me Curious Penguin's experience"
            ]
            for question in example_questions:
                gr.Button(question).click(
                    lambda q: (q, ""),
                    inputs=[lambda: question],
                    outputs=[txt],
                    queue=False
                )
        
        # Event handlers
        txt.submit(chat, [txt, chatbot], [chatbot, txt])
        submit_btn.click(chat, [txt, chatbot], [chatbot, txt])
        clear_btn.click(lambda: ([], ""), None, [chatbot, txt], queue=False)
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        share=True,
        favicon_path=None  # Disable favicon to prevent 404
    ) 