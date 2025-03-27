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

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/natgluons/hiringhelp-chatbot)

A Gradio-based chatbot that helps with hiring-related questions by analyzing candidate information and providing intelligent responses.

## Features

- Interactive chat interface built with Gradio
- RAG (Retrieval-Augmented Generation) based responses
- Support for PDF, TXT, and CSV document formats
- Example questions for easy interaction
- Source attribution for responses
- Modern and responsive UI

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

5. Add your knowledge source documents (PDF, TXT, or CSV) to the `knowledge_sources` directory.

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
    ├── sample_candidates.txt  # Sample candidate data
    └── README.md      # Instructions for adding documents
```

## Overview
HiringHelp Chatbot is an intelligent hiring assistant that uses Retrieval-Augmented Generation (RAG) to match candidates with job positions. Built with LangChain and advanced language models, the chatbot analyzes candidate resumes by first retrieving relevant information from documents and then generating contextual responses. This RAG architecture ensures responses are grounded in actual candidate data rather than hallucinations.

## Model Choices
- **Chat Completion**: Using Qwen-2-7B-Chat (via OpenRouter) for its fast and accurate responses in RAG applications
- **Embeddings**: Using OpenAI's text-embedding-ada-002 for optimal compatibility with LangChain and proven reliability in document retrieval tasks

## Features
- **RAG-Based Analysis**: Uses Retrieval-Augmented Generation to provide accurate, document-grounded responses
- **Resume Analysis**: Processes and analyzes candidate resumes in PDF format
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
- **Document Processing**: PyPDF2 for PDF parsing, LangChain for text splitting and embedding
- **Rate Limiting**: Flask-Limiter for API protection
- **Data Storage**: SQLite for persistent storage
- **Containerization**: Docker for deployment

## How RAG Works in This Application
1. **Document Ingestion**:
   - Resumes are processed and split into chunks using LangChain's text splitters
   - Each chunk is embedded using OpenAI's text-embedding-ada-002 model
   - Embeddings are stored in a FAISS vector database

2. **Query Processing**:
   - User queries are embedded using the same OpenAI embedding model
   - Relevant resume sections are retrieved using vector similarity search
   - Retrieved context is used to generate accurate, grounded responses

3. **Response Generation**:
   - Qwen-2-7B-Chat model receives both the user query and retrieved context
   - Responses are generated based on actual resume content
   - The RAG approach ensures responses are factual and verifiable

## Requirements
```
Flask==2.0.3
Werkzeug==2.0.3
openai>=1.0.0
sqlalchemy==1.4.25
python-dotenv==1.0.1
PyPDF2==3.0.1
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

3. Add candidate resumes:
Place PDF resumes in the `knowledge_sources` directory.

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
├── knowledge_sources/      # Directory for candidate resumes
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
