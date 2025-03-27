# 👔 HiringHelp Chatbot

### A chatbot that helps you find the most fitting candidate for the role! Made using **RAG (Retrieval-Augmented Generation)** with **LangChain**.

![image](https://github.com/user-attachments/assets/afbcd76a-d26f-40af-9081-3fe4e7e041ca)

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

## Usage Examples
```
"List all the available candidates"
"Tell me about a candidate named [Name]"
"Which candidate is best for [Role] role?"
```

## Demo Snapshots
A demo version is available with sample candidate data for testing purposes.

### ❔ Ask about a specific candidate
<img src="https://github.com/user-attachments/assets/25ca5927-b981-49b4-8c74-9859ad0fc5cf" width="800px">

### 🏆 Or ask who's best for the role
<img src="https://github.com/user-attachments/assets/9bba78b9-a027-4d8c-b553-688a5850d680" width="800px">

## 👀 Preview all candidate's resume!
<img src="https://github.com/user-attachments/assets/5762a3f8-3df3-4ca1-8151-7fe8b97206cd" width="800px">

## Rate Limits
- 10 requests per minute
- 100 requests per day

## Deployment 
1. **Hugging Face Spaces**: [Interactive Demo](https://huggingface.co/spaces/natgluons/HiringHelp-Chatbot) - active

<img src="https://github.com/user-attachments/assets/25a08529-317d-491b-a376-fec7f224c365" width="800px">

2. **Vercel Deployment**: [Live Demo](https://hiring-help-chatbot.vercel.app/) - inactive API [Preview Only]

<img src="https://github.com/user-attachments/assets/c1a43285-bbec-457a-ab24-ba6cdb0a0bde" width="800px">

3. **Local Development**: See below for setup instructions & local-docs branch for complete script.

---
*❝
Interested in building your own chatbot? Follow this setup instructions below! (◕‿◕)
❞*
---

# Local Development Setup

### Requirements
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
