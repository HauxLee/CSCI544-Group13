document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    const connectionArea = document.getElementById('connection-area');
    const chatArea = document.getElementById('chat-area');
    const dbIdInput = document.getElementById('db-id-input');
    const connectButton = document.getElementById('connect-button');
    const connectionStatus = document.getElementById('connection-status');
    const connectedDbId = document.getElementById('connected-db-id');
    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const agentStatus = document.getElementById('agent-status');

    let currentDbId = null; // Store the connected DB ID

    // --- Connection Logic ---
    connectButton.addEventListener('click', handleConnect);
    dbIdInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleConnect();
        }
    });

    function handleConnect() {
        const dbId = dbIdInput.value.trim();
        if (!dbId) {
            connectionStatus.textContent = 'Please enter a Database ID.';
            return;
        }

        connectionStatus.textContent = `Attempting to connect to "${dbId}"...`;
        connectButton.disabled = true;
        dbIdInput.disabled = true;

        console.log('Initiating fetch to /api/connect...'); // Log start

        fetch('/api/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ db_id: dbId })
        })
        .then(response => {
            console.log('Raw Response Status:', response.status); // Log response status
            console.log('Raw Response OK:', response.ok); // Log ok status (true for 200-299)

            // Clone response to log body safely, as response.json() consumes it
            return response.clone().text().then(text => {
                console.log('Raw Response Body Text:', `"${text}"`); // Log raw text body (added quotes for clarity)
                if (!response.ok) {
                     // Throw error to be caught by .catch if status is not 2xx
                     throw new Error(`HTTP error! status: ${response.status}, body: ${text}`);
                 }
                 // Check if body is empty before parsing
                 if (!text) {
                    console.log('Response body is empty, assuming failure.');
                    // Treat empty body as failure for this specific API
                    return { success: false, error: 'Received empty response from server.' };
                 }
                 // Attempt to parse JSON only if body is not empty
                 try {
                     return JSON.parse(text); // Manually parse after logging
                 } catch (parseError) {
                     console.error('JSON parsing error:', parseError);
                     throw new Error(`Failed to parse JSON response: ${text}`);
                 }
             });
        })
        .then(data => {
            console.log('Parsed Data:', data); // Log the parsed data
            if (data && data.success === true) { // Be explicit checking for true
                console.log('Processing successful connection...'); // Log success path
                currentDbId = dbId;
                connectedDbId.textContent = dbId;
                // --- UI Switch on Success ---
                connectionArea.classList.add('hidden');
                chatArea.classList.remove('hidden');
                userInput.disabled = false;
                sendButton.disabled = false;
                userInput.focus();
                addMessageToChat(`Connected to '${dbId}'! How can I help you?`, 'agent-message');
                connectionStatus.textContent = ''; // Clear status
            } else {
                // Connection failed according to backend response or processing
                const errorMsg = data ? data.error : 'Unknown error: Backend response did not indicate success or was unparseable.';
                console.log('Processing failed connection:', errorMsg); // Log failure path
                connectionStatus.textContent = `Connection failed: ${errorMsg}`;
                // --- Reset UI Elements on Failure ---
                connectButton.disabled = false;
                dbIdInput.disabled = false;
                // Explicitly reset UI visibility on failure
                chatArea.classList.add('hidden');
                connectionArea.classList.remove('hidden');
            }
        })
        .catch(error => {
            console.error('Fetch caught error:', error); // Log any error during fetch/parsing
            connectionStatus.textContent = `Connection error: ${error.message || error}. Check console.`;
             // --- Reset UI Elements on Error ---
            connectButton.disabled = false;
            dbIdInput.disabled = false;
             // Explicitly reset UI visibility on error
             chatArea.classList.add('hidden');
            connectionArea.classList.remove('hidden');
        });

        // Removed the simulation blocks completely now
    }


    // --- Chat Logic ---
    sendButton.addEventListener('click', handleSendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !sendButton.disabled) {
            handleSendMessage();
        }
    });

    function handleSendMessage() {
        const messageText = userInput.value.trim();
        if (!messageText) return;

        addMessageToChat(messageText, 'user-message');
        userInput.value = ''; // Clear input field
        userInput.disabled = true;
        sendButton.disabled = true;
        agentStatus.textContent = 'Agent is thinking...';

        // --- !!! SIMULATION: Replace with actual fetch to your backend /chat endpoint !!! ---
        console.log(`Frontend: Sending message to agent: "${messageText}"`);
        // Example backend endpoint: POST /api/chat with body { "message": messageText, "db_id": currentDbId }
        // You might also need to send a session/thread ID
        fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: messageText, db_id: currentDbId /*, thread_id: your_thread_id */ })
        })
        .then(response => response.json())
        .then(data => {
            if (data.response) {
                addMessageToChat(data.response, 'agent-message');
            } else {
                 addMessageToChat(`Error: ${data.error || 'Agent did not respond.'}`, 'agent-message');
            }
        })
        .catch(error => {
            console.error('Chat error:', error);
            addMessageToChat('Error communicating with agent. Check console.', 'agent-message');
        })
        .finally(() => {
             agentStatus.textContent = '';
             userInput.disabled = false;
             sendButton.disabled = false;
             userInput.focus();
        });

        // --- Simulated Agent Response (Remove this block when using actual fetch) ---
        // setTimeout(() => {
        //     const simulatedResponse = `Okay, I received your message: "${messageText}". I am a simulated agent, so I can't process it for real, but in a real application, I would have generated and executed SQL based on this.`;
        //     console.log(`Frontend: Simulating agent response.`);
        //     addMessageToChat(simulatedResponse, 'agent-message');
        //     agentStatus.textContent = '';
        //     userInput.disabled = false;
        //     sendButton.disabled = false;
        //     userInput.focus();
        // }, 2000); // Simulate agent thinking time
        // --- End Simulated Agent Response ---
    }

    // Helper function to add messages to the chat display
    function addMessageToChat(text, className) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', className);
        messageElement.textContent = text; // Use textContent for security
        chatHistory.appendChild(messageElement);
        // Scroll to the bottom
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});