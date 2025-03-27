// Add initial greeting message when page loads
window.addEventListener('load', () => {
    const chatbox = document.getElementById('chatbox');
    chatbox.innerHTML = `
        <p class="hiringhelp-message">HiringHelp: Hello, how can I help you today?</p>
        <div class="example-prompts-container">
            <p>Try these example questions:</p>
            <div class="example-prompt" onclick="listCandidates()">
                "List all the available candidates"
            </div>
            <div class="example-prompt" onclick="usePrompt('Tell me about a candidate named Kristy Natasha Yohanes')">
                "Tell me about a candidate named Kristy Natasha Yohanes"
            </div>
            <div class="example-prompt" onclick="usePrompt('Which candidate is best for an AI Engineer role?')">
                "Which candidate is best for an AI Engineer role?"
            </div>
        </div>
    `;

    // Add event listeners after the DOM is loaded
    document.addEventListener('DOMContentLoaded', () => {
        const sendButton = document.getElementById('sendButton');
        const userInput = document.getElementById('userInput');

        if (sendButton) {
            sendButton.addEventListener('click', sendMessage);
        }

        if (userInput) {
            userInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        }
    });
});

// Function to use selected prompt
function usePrompt(prompt) {
    const userInput = document.getElementById('userInput');
    userInput.value = prompt;
    document.querySelector('.example-prompts-container').remove();
}

// Function to handle listing candidates
async function listCandidates() {
    const userInput = "List all the available candidates";
    const chatbox = document.getElementById('chatbox');
    
    // Remove example prompts if they still exist
    const examplePrompts = document.querySelector('.example-prompts-container');
    if (examplePrompts) {
        examplePrompts.remove();
    }

    chatbox.innerHTML += `<p class="user-message">You: ${userInput}</p>`;

    // Show loading indicator and start timer
    const loadingMessage = document.createElement('div');
    loadingMessage.className = 'loading-message';
    loadingMessage.innerHTML = `
        <div class="spinner"></div>
        <span id="generation-time">Generating response... (0s)</span>
    `;
    chatbox.appendChild(loadingMessage);
    const startTime = Date.now();
    const timerInterval = setInterval(() => {
        const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
        document.getElementById('generation-time').textContent = `Generating response... (${elapsedSeconds}s)`;
    }, 1000);

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: userInput })
        });

        // Clear timer and remove loading indicator
        clearInterval(timerInterval);
        loadingMessage.remove();

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        // Add the HiringHelp's response
        chatbox.innerHTML += `<p class="hiringhelp-message">HiringHelp: ${data.response}</p>`;

    } catch (error) {
        // Clear timer and remove loading indicator
        clearInterval(timerInterval);
        loadingMessage.remove();

        const errorMessage = document.createElement('p');
        errorMessage.className = 'error-message';
        errorMessage.textContent = `Error: ${error.message}`;
        chatbox.appendChild(errorMessage);
    }

    // Scroll to bottom
    chatbox.scrollTop = chatbox.scrollHeight;
}

document.getElementById('sendButton').addEventListener('click', async () => {
    const userInput = document.getElementById('userInput').value;
    if (userInput.trim() === '') return;

    // Remove example prompts if they still exist
    const examplePrompts = document.querySelector('.example-prompts-container');
    if (examplePrompts) {
        examplePrompts.remove();
    }

    const chatbox = document.getElementById('chatbox');
    chatbox.innerHTML += `<p class="user-message">You: ${userInput}</p>`;

    // Show loading indicator and start timer
    const loadingMessage = document.createElement('div');
    loadingMessage.className = 'loading-message';
    loadingMessage.innerHTML = `
        <div class="spinner"></div>
        <span id="generation-time">Generating response... (0s)</span>
    `;
    chatbox.appendChild(loadingMessage);
    const startTime = Date.now();
    const timerInterval = setInterval(() => {
        const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
        document.getElementById('generation-time').textContent = `Generating response... (${elapsedSeconds}s)`;
    }, 1000);

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: userInput })
        });

        // Clear timer and remove loading indicator
        clearInterval(timerInterval);
        loadingMessage.remove();

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        // Add the HiringHelp's response
        chatbox.innerHTML += `<p class="hiringhelp-message">HiringHelp: ${data.response}</p>`;

        // If there are sources, add them
        if (data.sources && data.sources.length > 0) {
            let sourcesHtml = '<p class="sources">Sources:<br>';
            data.sources.forEach(source => {
                sourcesHtml += `- ${source.title} (Page ${source.page_number})<br>`;
            });
            sourcesHtml += '</p>';
            chatbox.innerHTML += sourcesHtml;
        }

        // Update rate limit info if available
        if (data.rate_limit) {
            const rateLimitInfo = document.getElementById('rate-limit-info');
            rateLimitInfo.textContent = `Rate limit: ${data.rate_limit.remaining}/${data.rate_limit.limit} requests per minute`;
        }
    } catch (error) {
        // Clear timer and remove loading indicator
        clearInterval(timerInterval);
        loadingMessage.remove();

        const errorMessage = document.createElement('p');
        errorMessage.className = 'error-message';
        errorMessage.textContent = `Error: ${error.message}`;
        chatbox.appendChild(errorMessage);
    }

    document.getElementById('userInput').value = '';
    chatbox.scrollTop = chatbox.scrollHeight;
});

// Test API connection button
document.getElementById('testApiButton').addEventListener('click', async () => {
    const resultDiv = document.getElementById('apiTestResult');
    resultDiv.textContent = "Testing API connection...";
    
    try {
        const response = await fetch('/test-api');
        const data = await response.json();
        
        if (data.success) {
            resultDiv.innerHTML = `<span style="color: green;">✓ API works! Response: "${data.response}"</span>`;
        } else {
            resultDiv.innerHTML = `<span style="color: red;">✗ API error: ${data.error}</span>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<span style="color: red;">✗ Connection error: ${error.message}</span>`;
    }
});
