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

# Event stream handling function with JSON logging
def event_handler(event: dict, _printed: set, log_data: list, max_length=5000):
    message = event.get("messages")
    current_state = None  # Initialize current state

    if message:
        if isinstance(message, list):
            message = message[-1]
        # Get message type and name
        message_type = message.type
        assistant_name = getattr(message, 'name', None)

        # Infer current state based on message_type and assistant_name
        if message_type == 'ai':
            current_state = 'Execution Assistant'
        elif message_type == 'tool':
            current_state = 'Execution_Tools'
        elif message_type == 'human':
            current_state = 'User Input'
        else:
            current_state = 'Unknown State'

        # Adjust message_type display format
        if message_type == 'ai':
            message_type = 'AI'
        elif message_type == 'human':
            message_type = 'Human'
        elif message_type == 'tool':
            message_type = 'Tool'
        else:
            message_type = message_type.capitalize()

        # Build log entry without assistant_name
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "state": current_state,
            "message_id": message.id,
            "message_type": message_type,
            "message_content": message.content,
            "message_tool_calls": getattr(message, 'tool_calls', None),
        }

        # Add to printed set and print message if not already printed
        if message.id not in _printed:
            # Skip printing Human messages as they are handled separately
            if message_type != 'Human':
                msg_repr = message.pretty_repr(html=True)
                if len(msg_repr) > max_length:
                    msg_repr = msg_repr[:max_length] + " ... (truncated)"
                print(msg_repr)
            _printed.add(message.id)

        log_data.append(log_entry)
        return message  # Return the message for checking COMPLETE TASK
    else:
        # If there's no message, still build the log entry
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

# Print initial welcome message directly (without using the agent)
print("\n================================== AI Message =================================\n")
print("Hello! I'm your Database QA Agent. What can I do for you today?")

# Process event stream and output results
_printed = set()
log_data = []

try:
    # Start conversation loop
    conversation_state = None
    
    while True:
        # Get user input
        user_input = input("\n(Send a message to the Agent): ")
        if not user_input.strip():  # Skip empty inputs
            continue
            
        # Print human message
        print(f"\n================================== Human Message =================================\n")
        print(user_input)
        
        # Create or update conversation state
        if conversation_state and "messages" in conversation_state:
            # Keep conversation history and add new user message
            messages_history = conversation_state["messages"]
            conversation_state = {
                "messages": messages_history + [HumanMessage(content=user_input)]
            }
        else:
            # First message or no previous state
            conversation_state = {"messages": [HumanMessage(content=user_input)]}
        
        # Process the conversation
        events = graph.stream(
            conversation_state,
            config,
            stream_mode="values",
        )
        
        # Handle events
        complete_task_detected = False
        
        for event in events:
            message = event_handler(event, _printed, log_data)
            conversation_state = event  # Store the latest state
            
            # Check if COMPLETE TASK was detected
            if message and hasattr(message, 'content') and "COMPLETE TASK" in message.content:
                complete_task_detected = True
        
        # No need to handle COMPLETE TASK specially since we always prompt for new input
        
except KeyboardInterrupt:
    print("\n\nConversation ended by user.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    # Generate a unique filename with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"agent_log_{timestamp_str}.json"
    log_filepath = os.path.join('logs', log_filename)

    # Write log data to JSON file
    with open(log_filepath, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nLog saved to {log_filepath}")