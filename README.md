# OpenAI Chatbot with RAG (Retrieval-Augmented Generation): Retrieve Information from CV

## Overview
This project is an OpenAI-powered chatbot that uses Retrieval-Augmented Generation (RAG) to enhance responses with relevant information from provided documents. The chatbot is deployed using Docker and Kubernetes, and it's designed to be accessed via a web interface.

## Features
- Natural Language Processing: Utilizes OpenAI's GPT-3.5 model to generate responses.
- Document Retrieval: Enhances responses by retrieving relevant information from a set of documents stored in the `knowledge_sources` folder.
- Web Interface: Provides a simple web interface for user interaction.
- Dockerized Deployment: Containerized using Docker for easy deployment.
- Kubernetes: Supports deployment on Kubernetes for scalable and reliable service.

## Requirements
- Python 3.9+
- Flask==2.0.3
- Werkzeug==2.0.3
- openai==1.38.0
- sqlalchemy==1.4.25
- python-dotenv==1.0.1
- PyPDF2==3.0.1
- pandas==2.2.0
- scikit-learn==1.5.0
- Docker
- Kubernetes

## Docker Desktop Setup

1. Install Docker Desktop:
   - Download Docker Desktop from https://www.docker.com/products/docker-desktop
   - Install and start Docker Desktop
   - Wait for Docker Desktop to finish starting (you'll see the whale icon in your system tray)

2. Verify Docker Installation:
```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker-compose --version
```

3. Start Docker Desktop:
   - Open Docker Desktop application
   - Wait for it to fully start (the whale icon should stop animating)
   - You can verify it's running by opening a terminal and running:
```bash
docker ps
```

## Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/natgluons/RAG-Chatbot.git
cd RAG-Chatbot
```

2. Set up environment variables:
Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. Set up knowledge sources:
Create a `knowledge_sources` folder in the root directory and add your documents (PDF, TXT, etc.) that you want the chatbot to use for answering questions.

4. Database Setup:
The project uses SQLite as its database. The database file (`database.db`) will be automatically created when you first run the application. No additional setup is required.

5. Run the application using Docker:
```bash
# Build and start the containers
docker-compose up --build

# The web interface will be available at http://localhost:5000
```

To stop the application:
```bash
docker-compose down
```

To restart the application:
```bash
# Restart without rebuilding
docker-compose restart

# Restart with rebuilding (if you made changes to requirements.txt or code)
docker-compose up --build -d
```

To view logs after restart:
```bash
docker-compose logs -f
```

## Project Structure
- `knowledge_sources/`: Place your documents here (PDF, TXT, etc.)
- `backend/`: Contains the Flask application and RAG implementation
- `database/`: Contains database-related files
- `database.db`: SQLite database file (created automatically)
- `index.html`: Frontend interface
- `styles.css`: Frontend styling

## Docker Deployment

## Access the Web Interface
Open your browser and navigate to `http://34.71.245.123/` to interact with the chatbot. Ask anything related to my CV, background, professional, and academic experience. This is the minimum viable product (MVP) under development; the final version will be hosted on a domain website, to be announced later.

*this service is currently offline due to cost considerations (Why does Kubernetes cost so much!?)

![ragchatbot](https://github.com/user-attachments/assets/4570bf02-735f-4f92-94f8-b803e6859997)
