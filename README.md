# HiringHelp Chatbot: AI-Powered Hiring Assistant

## Overview
HiringHelp Chatbot is an intelligent hiring assistant that helps match candidates with job positions using advanced language models and document retrieval techniques. The chatbot analyzes candidate resumes and provides insights based on your company's requirements.

## Features
- **Resume Analysis**: Processes and analyzes candidate resumes in PDF format
- **Intelligent Matching**: Uses advanced language models to match candidates with job requirements
- **Interactive Chat Interface**: User-friendly web interface for natural conversations
- **Rate-Limited API**: Implements rate limiting (10 requests/minute, 100 requests/day) for stable service
- **Document Management**: Stores and retrieves candidate information from the `knowledge_sources` directory
- **Vector Search**: Uses FAISS for efficient similarity search in candidate documents
- **Secure Environment**: Handles sensitive information through environment variables

## Tech Stack
- **Backend**: Flask (Python)
- **Language Model**: OpenRouter API (supports multiple models)
- **Vector Database**: FAISS for efficient document retrieval
- **Document Processing**: PyPDF2 for PDF parsing
- **Rate Limiting**: Flask-Limiter for API protection
- **Data Storage**: SQLite for persistent storage
- **Containerization**: Docker for deployment

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

## Deployment on Hugging Face Spaces

1. Create a new Space on Hugging Face:
   - Go to huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Docker" as the SDK
   - Set the visibility as needed

2. Configure the Space:
   - Add your `OPENROUTER_API_KEY` to the Space's secrets
   - Link your GitHub repository
   - Enable Docker build

3. The Space will automatically build and deploy your Docker container.

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

## License
[Add your license information here]

## Contact
[Add your contact information here]
