# HiringHelp Chatbot

A chatbot that helps with hiring-related questions using RAG (Retrieval-Augmented Generation) with LangChain.

## How It Works
HiringHelp uses LangChain's RAG implementation to provide accurate, document-grounded responses. The process involves:
1. **Document Processing**: Candidate documents are split into chunks and embedded
2. **Retrieval**: When a query is received, relevant document chunks are retrieved using FAISS vector similarity
3. **Generation**: Retrieved context is combined with the query to generate accurate responses

## Technology
- **API & Model**: Using Qwen-2-7B-Chat via OpenRouter API for its balance of performance and cost-effectiveness in RAG applications, with custom embedding generation for document retrieval.
- **Stack**: LangChain, FAISS, Gradio, Flask-Limiter

## Features
- Interactive chat interface
- Support for text document formats
- Example questions for easy interaction
- Source attribution for responses
- Rate limiting (10 requests/minute, 100 requests/day)
- Vector similarity search for accurate retrieval
- Environment variable configuration

## Deployment Options
1. **Vercel Deployment**: [Live Demo](https://hiring-help-chatbot.vercel.app/)
2. **Hugging Face Spaces**: [Interactive Demo](https://huggingface.co/spaces/natgluons/HiringHelp-Chatbot)
3. **Local Development**: See [docs branch](https://github.com/natgluons/hiringhelp-chatbot/tree/docs) for setup instructions

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
"Which candidate is best for [Role] role?"
```

## Rate Limits
- 10 requests per minute
- 100 requests per day

## Demo
A demo version is available with sample candidate data for testing purposes.