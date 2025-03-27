---
title: HiringHelp-Chatbot
emoji: 👨‍💼
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.22.0"
app_file: app.py
pinned: false
---

# HiringHelp Chatbot

A chatbot that helps with hiring-related questions using RAG (Retrieval-Augmented Generation) with Gradio interface.

## Features

- Interactive chat interface using Gradio
- RAG system for retrieving relevant information from candidate documents
- Support for text document formats
- Conversation memory to maintain context
- Real-time responses using OpenRouter API

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```
4. Place your candidate documents in the `knowledge_sources` directory
5. Run the application:
   ```bash
   python app.py
   ```

## Usage

1. Start the application
2. Ask questions about candidates or hiring-related topics
3. The chatbot will retrieve relevant information from the documents and provide answers

## Project Structure

- `app.py`: Main application file
- `requirements.txt`: Python dependencies
- `knowledge_sources/`: Directory containing candidate documents
- `.env`: Environment variables (API keys)

## Dependencies

- gradio
- openai
- python-dotenv
- pandas
- langchain
- faiss-cpu
- requests
- beautifulsoup4

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/natgluons/hiringhelp-chatbot.git
cd hiringhelp-chatbot
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your OpenRouter API key:
```
OPENROUTER_API_KEY=your_api_key_here
```

5. Add your knowledge source documents to the `knowledge_sources` directory.

6. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:7860`.

## Deploying to Hugging Face Spaces

1. Fork this repository to your GitHub account

2. Create a new Space on Hugging Face:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "New Space"
   - Choose "Gradio" as the SDK
   - Name your space (e.g., "hiringhelp-chatbot")

3. Link your GitHub repository:
   - Go to the Space's settings
   - Navigate to the "Repository" section
   - Select your GitHub repository

4. Set up the OpenRouter API key:
   - Go to your Space's settings
   - Navigate to "Repository Secrets"
   - Add your `OPENROUTER_API_KEY` as a secret

The Space will automatically build and deploy your application.

## Project Structure

```
HiringHelp-Chatbot/
├── app.py              # Main Gradio application
├── styles.css          # Custom CSS styles
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (local only)
└── knowledge_sources/ # Directory for knowledge base documents
    └── README.md      # Instructions for adding documents
```

## Overview
HiringHelp Chatbot is an intelligent hiring assistant that uses Retrieval-Augmented Generation (RAG) to match candidates with job positions. Built with LangChain and advanced language models, the chatbot analyzes candidate resumes by first retrieving relevant information from documents and then generating contextual responses. This RAG architecture ensures responses are grounded in actual candidate data rather than hallucinations.

## Model Choices
- **Chat Completion**: Using Qwen-2-7B-Chat (via OpenRouter) for its fast and accurate responses in RAG applications
- **Embeddings**: Using OpenAI's text-embedding-ada-002 for optimal compatibility with LangChain and proven reliability in document retrieval tasks

## Features
- **RAG-Based Analysis**: Uses Retrieval-Augmented Generation to provide accurate, document-grounded responses
- **Resume Analysis**: Processes and analyzes candidate resumes
- **Intelligent Matching**: Uses LangChain and advanced language models to match candidates with job requirements
- **Interactive Chat Interface**: User-friendly web interface for natural conversations
- **Rate-Limited API**: Implements rate limiting (10 requests/minute, 100 requests/day) for stable service
- **Document Management**: Stores and retrieves candidate information from the `knowledge_sources` directory
- **Vector Search**: Uses FAISS for efficient similarity search in candidate documents
- **Secure Environment**: Handles sensitive information through environment variables

## Tech Stack
- **Framework**: LangChain for RAG implementation and document processing
- **Language Models**:
  - Qwen-2-7B-Chat: Primary model for chat completions (via OpenRouter)
  - text-embedding-ada-002: OpenAI's embedding model for document vectorization
- **Vector Database**: FAISS for efficient document retrieval
- **Document Processing**: LangChain for text splitting and embedding
- **Rate Limiting**: Flask-Limiter for API protection
- **Data Storage**: SQLite for persistent storage
- **Containerization**: Docker for deployment

## How RAG Works in This Application
1. **Document Ingestion**:
   - Documents are processed and split into chunks using LangChain's text splitters
   - Each chunk is embedded using OpenAI's text-embedding-ada-002 model
   - Embeddings are stored in a FAISS vector database

2. **Query Processing**:
   - User queries are embedded using the same OpenAI embedding model
   - Relevant document sections are retrieved using vector similarity search
   - Retrieved context is used to generate accurate, grounded responses

3. **Response Generation**:
   - Qwen-2-7B-Chat model receives both the user query and retrieved context
   - Responses are generated based on actual document content
   - The RAG approach ensures responses are factual and verifiable

## Requirements
```
Flask==2.0.3
Werkzeug==2.0.3
openai>=1.0.0
sqlalchemy==1.4.25
python-dotenv==1.0.1
pandas==2.2.0
scikit-learn==1.5.0
langchain-core>=0.1.17
langchain-community>=0.0.13
langchain>=0.1.0
tiktoken
langchain-openai
faiss-cpu==1.7.4
Flask-Limiter>=3.5.0
requests>=2.32.3
aiohttp==3.9.1
beautifulsoup4==4.12.2
```

## Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/natgluons/HiringHelp-Chatbot.git
cd HiringHelp-Chatbot
```

2. Set up environment variables:
Create a `.env` file in the root directory:
```
OPENROUTER_API_KEY=your_api_key_here
```

3. Add candidate documents:
Place candidate documents in the `knowledge_sources` directory.

4. Run with Docker:
```bash
# Build and start the container
docker-compose up --build

# Access the web interface at http://localhost:5000
```

## Docker Commands
```bash
# Start the application
docker-compose up -d

# Stop the application
docker-compose down

# View logs
docker-compose logs -f

# Rebuild and restart
docker-compose up --build -d
```

## Project Structure
```
HiringHelp-Chatbot/
├── api/                    # Main application code
│   ├── index.py           # Flask application and API endpoints
│   └── __init__.py
├── knowledge_sources/      # Directory for candidate documents
├── lib/                    # Helper libraries
├── public/                 # Static files
├── database/              # Database related files
├── docker-compose.yml     # Docker compose configuration
├── Dockerfile             # Docker build instructions
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables
```

## Usage Examples
```
"List all the available candidates"
"Tell me about a candidate named [Name]"
"Which candidate is best for an AI Engineer role?"
```

## Rate Limits
- 10 requests per minute
- 100 requests per day

## Demo
A demo version is available with sample candidate data for testing purposes.

## Security Note
This application handles sensitive information. Always:
- Keep API keys secure
- Use environment variables for secrets
- Review candidate information handling policies
- Monitor rate limits and usage
