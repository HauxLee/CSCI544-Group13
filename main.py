# # main.py
# import uuid
# import json
# import os
# from datetime import datetime

# from agent import graph

# # Configure initial parameters
# thread_id = str(uuid.uuid4())
# config = {
#     "configurable": {
#         "passenger_id": "3442 587242",
#         "thread_id": thread_id
#     },
#     "recursion_limit": 100
# }

# # Ensure the 'logs' directory exists
# if not os.path.exists('logs'):
#     os.makedirs('logs')

# # Event stream handling function with JSON logging
# def event_handler(event: dict, _printed: set, log_data: list, max_length=5000):
#     message = event.get("messages")
#     current_state = None  # Initialize current state

#     if message:
#         if isinstance(message, list):
#             message = message[-1]
#         # Get message type and name
#         message_type = message.type
#         assistant_name = getattr(message, 'name', None)

#         # Infer current state based on message_type and assistant_name
#         if message_type == 'ai':
#             current_state = 'Execution Assistant'
#         elif message_type == 'tool':
#             current_state = 'Execution_Tools'
#         elif message_type == 'human':
#             current_state = 'User Input'
#         else:
#             current_state = 'Unknown State'

#         # Adjust message_type display format
#         if message_type == 'ai':
#             message_type = 'AI'
#         elif message_type == 'human':
#             message_type = 'Human'
#         elif message_type == 'tool':
#             message_type = 'Tool'
#         else:
#             message_type = message_type.capitalize()

#         # Build log entry without assistant_name
#         log_entry = {
#             "timestamp": datetime.now().isoformat(),
#             "state": current_state,
#             "message_id": message.id,
#             "message_type": message_type,
#             "message_content": message.content,
#             "message_tool_calls": getattr(message, 'tool_calls', None),
#         }

#         # Add to printed set and print message if not already printed
#         if message.id not in _printed:
#             if message_type != 'Human':  # Skip printing Human messages
#                 msg_repr = message.pretty_repr(html=True)
#                 if len(msg_repr) > max_length:
#                     msg_repr = msg_repr[:max_length] + " ... (truncated)"
#                 print(msg_repr)
#             _printed.add(message.id)

#     else:
#         # If there's no message, still build the log entry
#         log_entry = {
#             "timestamp": datetime.now().isoformat(),
#             "state": None,
#             "message_id": None,
#             "message_type": None,
#             "message_content": None,
#             "message_tool_calls": None,
#         }

#     log_data.append(log_entry)

# # Initialize questions
# initial_questions = [
#     "[START_CONVERSATION]",
# ]

# # Process event stream and output results
# _printed = set()
# log_data = []

# for question in initial_questions:
#     print("\n================================== Human Message =================================\n")
#     print(question)
#     events = graph.stream(
#         {"messages": ("user", question)},
#         config,
#         stream_mode="values",
#     )
#     for event in events:
#         event_handler(event, _printed, log_data)

# # Generate a unique filename with timestamp
# timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
# log_filename = f"agent_log_{timestamp_str}.json"
# log_filepath = os.path.join('logs', log_filename)

# # Write log data to JSON file
# with open(log_filepath, 'w', encoding='utf-8') as f:
#     json.dump(log_data, f, ensure_ascii=False, indent=4)
# main.py
# 修改 main.py

import uuid
import json
import os
from datetime import datetime

from agent import graph
from langchain_core.messages import HumanMessage

# Configure initial parameters
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        "passenger_id": "3442 587242",
        "thread_id": thread_id
    },
    "recursion_limit": 100
}

# Ensure the 'logs' directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# Generate a unique log filename with timestamp for this session
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"agent_log_{timestamp_str}.json"
log_filepath = os.path.join('logs', log_filename)

# Function to save logs to disk
def save_logs(log_data, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"\n❌ Error saving logs: {e}")

# Event stream handling function with JSON logging
def event_handler(event: dict, _printed: set, log_data: list, max_length=5000):
    message = event.get("messages")
    current_state = None

    if message:
        if isinstance(message, list):
            message = message[-1]
        
        message_type = message.type
        message_content = message.content if hasattr(message, 'content') else ""
        
        # Skip duplicate messages by content
        content_hash = hash(message_content)
        if content_hash in _printed:
            return message
        
        # Build log entry for internal logging
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "state": current_state,
            "message_id": message.id,
            "message_type": message_type,
            "message_content": message_content,
            "message_tool_calls": getattr(message, 'tool_calls', None),
        }
        log_data.append(log_entry)

        # Only print user-facing messages with improved formatting
        if message_type == 'ai':
            # Skip internal messages
            if ("DECISION ANALYSIS" not in message_content and 
                "COMPLETE TASK" not in message_content):
                content = message_content
                # Format the response for better readability
                if "```sql" in content:
                    parts = content.split("```sql")
                    explanation = parts[0].strip()
                    sql_query = parts[1].split("```")[0].strip()
                    
                    if explanation:
                        print("\n🤖 Assistant:", explanation)
                    print("\n📝 Generated SQL:")
                    print("─" * 40)
                    print(sql_query)
                    print("─" * 40)
                    
                    # If there's content after the SQL query
                    if len(parts[1].split("```")) > 1:
                        remaining = parts[1].split("```")[1].strip()
                        if remaining:
                            print("\n🤖 Assistant:", remaining)
                else:
                    print("\n🤖 Assistant:", content.strip())
                _printed.add(content_hash)
        elif message_type == 'tool' and message_content:
            try:
                # Parse tool response and format it nicely
                if message_content.startswith("{") or message_content.startswith("["):
                    # Skip showing raw JSON data - let the AI interpret it
                    pass
                else:
                    # Only show non-technical tool messages
                    if not message_content.startswith("Name:"):
                        print("\n💡 Info:", message_content)
                _printed.add(content_hash)
            except:
                pass  # Ignore any parsing errors

        return message
    else:
        # Log empty events
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "state": None,
            "message_id": None,
            "message_type": None,
            "message_content": None,
            "message_tool_calls": None,
        }
        log_data.append(log_entry)
        return None

# Print welcome message with improved formatting
print("\n" + "─" * 80)
print("🤖 Welcome to Database Assistant!")
print("I can help you explore and analyze your database. Just ask me anything!")
print("─" * 80 + "\n")

# Process event stream and output results
_printed = set()
log_data = []

try:
    conversation_state = None
    
    while True:
        # Get user input with a clean prompt
        user_input = input("\n👤 You: ")
        if not user_input.strip():
            continue
            
        # Check for exit commands
        if user_input.strip().lower() in ['exit', 'quit']:
            print("\n👋 Goodbye! Thanks for using Database Assistant.")
            break
            
        # Add user message to logs
        user_log_entry = {
            "timestamp": datetime.now().isoformat(),
            "state": "User Input",
            "message_id": f"user_{uuid.uuid4()}",
            "message_type": "Human",
            "message_content": user_input,
            "message_tool_calls": None,
        }
        log_data.append(user_log_entry)
            
        # Create or update conversation state
        if conversation_state and "messages" in conversation_state:
            messages_history = conversation_state["messages"]
            conversation_state = {
                "messages": messages_history + [HumanMessage(content=user_input)]
            }
        else:
            conversation_state = {"messages": [HumanMessage(content=user_input)]}
        
        # Show typing indicator
        print("\n🤖 Assistant is thinking...", end="\r")
        
        # Process the conversation
        events = graph.stream(
            conversation_state,
            config,
            stream_mode="values",
        )
        
        # Clear typing indicator
        print(" " * 30, end="\r")  # Clear the thinking message
        
        # Handle events
        complete_task_detected = False
        for event in events:
            message = event_handler(event, _printed, log_data)
            conversation_state = event
            
        # Save logs after each user interaction
        save_logs(log_data, log_filepath)

except KeyboardInterrupt:
    print("\n\n👋 Goodbye! Thanks for using Database Assistant.")
except Exception as e:
    print(f"\n❌ An error occurred: {e}")
finally:
    # Final log save before exiting
    save_logs(log_data, log_filepath)
    print(f"\nConversation logs saved to: {log_filepath}")