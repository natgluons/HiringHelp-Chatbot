from flask import Flask, request, jsonify, send_from_directory
import os
import requests

# Load environment variables only in development
if os.environ.get('ENV') != 'production':
    from dotenv import load_dotenv
    load_dotenv()

# Configure OpenRouter API
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
MODEL_NAME = os.environ.get('OPENROUTER_MODEL', 'qwen/qwen-2-7b-instruct:free')

app = Flask(__name__, static_folder='..', static_url_path='')
app.config['DOCS_FOLDER'] = 'knowledge_sources'

def get_candidate_names():
    """Extract candidate names from TXT filenames"""
    return sorted({
        f[3:-4]  # Remove "CV_" prefix and ".txt" extension
        for f in os.listdir(app.config['DOCS_FOLDER'])
        if f.startswith("CV_") and f.endswith(".txt")
    })

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        # Handle candidate listing
        if "list all" in prompt.lower() and "candidate" in prompt.lower():
            candidates = get_candidate_names()
            response = "Here are all available candidates:\n\n"
            for i, name in enumerate(candidates, 1):
                response += f"{i}. {name}\n"
            
            # Return in the correct message format
            return jsonify({
                "response": response,
                "sources": []
            })
        
        # Handle empty prompt
        if not prompt.strip():
            return jsonify({
                "response": "Please enter a question to continue",
                "sources": []
            })

        # Query OpenRouter API
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://your-domain.com",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        response.raise_for_status()
        
        return jsonify({
            "response": response.json()['choices'][0]['message']['content'],
            "sources": []
        })
        
    except requests.exceptions.HTTPError as e:
        return jsonify({
            "response": f"API Error: {str(e)}",
            "sources": []
        }), 500
    except Exception as e:
        return jsonify({
            "response": "Error processing your request",
            "sources": []
        }), 500

@app.route('/')
def index():
    return send_from_directory('..', 'index.html')

@app.route('/app.js')
def serve_js():
    return send_from_directory('..', 'app.js')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
