import os
import uuid
import json
from flask import Flask, request, jsonify, session, send_from_directory
from flask_session import Session # For server-side sessions
from flask_cors import CORS # For handling cross-origin requests from frontend
from dotenv import load_dotenv

# --- Import from your core_code ---
# Adjust paths if your app.py is not in the same directory as core_code

from agent import graph as agent_graph # Rename to avoid conflict
from agent import execution_tools # Import the original tool list
from tools import get_user_input_tool # Import the specific tool
from database.db_manager import DatabaseManager #
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage


# --- Configuration ---
load_dotenv() # Load environment variables (like OPENAI_API_KEY)

# --- Flask App Setup ---
app = Flask(__name__, static_folder='static', static_url_path='')
# IMPORTANT: Set a strong secret key for session security
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'e7d14e23e99c4f0f874a8e02145cb9d8718edbbf4a3d78ccf0f12cbb9b1e3f25')
# Configure server-side sessions (using filesystem by default)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False # Session expires when browser closes
Session(app)
CORS(app) # Allow requests from your frontend origin (adjust in production)

# --- Modify Agent Tools for Web Context ---
# Remove the get_user_input_tool as the frontend handles input
web_execution_tools = [tool for tool in execution_tools if tool.name != get_user_input_tool.name]
# Re-bind the tools to the LLM used in your agent (assuming llm is accessible or re-initialized)
# You might need to access the llm instance from agent.py or re-create it here
# This step depends on how agent.py is structured. If the llm is part of the graph,
# you might need to modify how the graph is compiled or passed around.
# For simplicity, let's assume the graph loaded from agent.py already has the correct tools *bound*
# If not, you would modify the agent definition (e.g., execution_assistant_runnable)
# like: web_assistant_runnable = execution_assistant_prompt | llm.bind_tools(web_execution_tools)
# And potentially recompile the graph if needed.

# --- API Endpoints ---

@app.route('/api/connect', methods=['POST'])
def connect_db():
    """
    Endpoint to validate the database path and initialize the session.
    """
    data = request.get_json()
    db_id_path = data.get('db_id')

    if not db_id_path:
        return jsonify({"success": False, "error": "Missing 'db_id' (database path)"}), 400

    # --- Validation ---
    # Basic check: Does the file exist?
    # Use the normalization logic potentially from DatabaseManager if needed
    normalized_path = os.path.normpath(db_id_path)
    if not os.path.isabs(normalized_path):
         # Assuming relative paths aren't typically used for DB IDs here
         # Or resolve relative to a known base directory if needed
         normalized_path = os.path.abspath(normalized_path) # Make absolute if needed

    if not os.path.exists(normalized_path) or not os.path.isfile(normalized_path):
         print(f"Connection attempt failed: Path not found or not a file - {normalized_path}")
         return jsonify({"success": False, "error": f"Database path not found or invalid: {db_id_path}"}), 400

    # More robust check: Try initializing DatabaseManager (optional, adds overhead)
    # try:
    #     # Temporarily set env var for the check if necessary
    #     os.environ['TARGET_DB_PATH'] = normalized_path
    #     temp_manager = DatabaseManager()
    #     temp_manager.disconnect() # Check connection and close
    #     del os.environ['TARGET_DB_PATH']
    # except Exception as e:
    #     print(f"Connection attempt failed: DBManager init error - {e}")
    #     if 'TARGET_DB_PATH' in os.environ: del os.environ['TARGET_DB_PATH']
    #     return jsonify({"success": False, "error": f"Failed to initialize database connection: {e}"}), 500

    # --- Store in Session ---
    session['db_id_path'] = normalized_path
    session['thread_id'] = str(uuid.uuid4())
    session['messages'] = [] # Initialize conversation history

    print(f"Session initialized for DB: {normalized_path}, Thread: {session['thread_id']}")
    return jsonify(success=True)

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Endpoint to handle user messages and interact with the agent.
    """
    if 'db_id_path' not in session or 'thread_id' not in session:
        return jsonify({"error": "Not connected. Please connect to a database first."}), 400

    data = request.json
    user_message_content = data.get('message')

    if not user_message_content:
        return jsonify({"error": "Missing 'message' in request"}), 400

    db_path = session['db_id_path']
    thread_id = session['thread_id']
    messages_history = session.get('messages', []) # Get history from session

    # Add new user message to history
    messages_history.append(HumanMessage(content=user_message_content).dict()) # Store as dict for JSON serialization

    # Prepare agent input state (convert dicts back to BaseMessage objects)
    agent_input_messages = [
        HumanMessage(**msg) if msg.get('type') == 'human'
        else AIMessage(**msg) if msg.get('type') == 'ai'
        else ToolMessage(**msg) if msg.get('type') == 'tool'
        # Add other types if needed
        else BaseMessage(**msg) # Fallback, might need adjustment
        for msg in messages_history
    ]

    config = {
        "configurable": {
            "thread_id": thread_id
        },
        "recursion_limit": 50 # Match your agent config
    }

    agent_response_content = "Sorry, I encountered an error."
    final_state_messages = []

    # --- Execute Agent ---
    # Use the workaround: Set/Unset TARGET_DB_PATH for the agent call
    original_env_value = os.environ.get('TARGET_DB_PATH')
    try:
        os.environ['TARGET_DB_PATH'] = db_path
        print(f"Set TARGET_DB_PATH for agent call: {db_path}") # Debug

        final_state = None
        # IMPORTANT: Ensure agent_graph uses the modified 'web_execution_tools' if you rebound them
        stream = agent_graph.stream({"messages": agent_input_messages}, config=config, stream_mode="values")

        for event in stream:
            final_state = event # Keep track of the last state

        # Process the final state to get the agent's response message(s)
        if final_state and final_state.get("messages"):
            all_final_messages = final_state["messages"]
            # Find the last AIMessage that is not a tool call (simplistic approach)
            last_ai_response = None
            for msg in reversed(all_final_messages):
                # Recreate object to check type/attributes
                if isinstance(msg, dict) and msg.get('type') == 'ai':
                    ai_msg_obj = AIMessage(**msg)
                    if not ai_msg_obj.tool_calls and ai_msg_obj.content and "DECISION ANALYSIS" not in ai_msg_obj.content:
                         last_ai_response = ai_msg_obj.content
                         break
                elif isinstance(msg, AIMessage): # If it's already an object
                     if not msg.tool_calls and msg.content and "DECISION ANALYSIS" not in msg.content:
                         last_ai_response = msg.content
                         break

            if last_ai_response:
                agent_response_content = last_ai_response
            else:
                 # Handle cases where the last message might be a tool call or decision analysis
                 agent_response_content = "Agent finished processing, but no final text response found."
                 # You might want to inspect final_state['messages'][-1] content more closely

            # Store the full message history from the final state back into the session
            # Convert objects back to dicts for storage
            final_state_messages = [msg.dict() if hasattr(msg, 'dict') else msg for msg in all_final_messages]

    except Exception as e:
        print(f"Error during agent execution: {e}")
        # Consider logging the full traceback
        agent_response_content = f"An error occurred: {e}"
        # Keep the history up to the point of failure
        final_state_messages = messages_history
    finally:
        # --- CRITICAL: Unset/Restore Environment Variable ---
        if original_env_value is None:
            if 'TARGET_DB_PATH' in os.environ:
                del os.environ['TARGET_DB_PATH']
                print("Unset TARGET_DB_PATH") # Debug
        else:
            os.environ['TARGET_DB_PATH'] = original_env_value
            print(f"Restored TARGET_DB_PATH to: {original_env_value}") # Debug

    # --- Update Session History ---
    session['messages'] = final_state_messages
    session.modified = True # Mark session as modified

    # --- Return Response ---
    return jsonify({"response": agent_response_content})


@app.route('/')
def index():
    return app.send_static_file('index.html')

# --- Run the App ---
if __name__ == '__main__':
    # Use waitress or gunicorn for production instead of Flask's dev server
    # Example: waitress-serve --host 127.0.0.1 --port 5001 app:app
    app.run(port=5000, debug=True) # Use port 5001 to avoid conflict if frontend is on 5000