# run_cosql_test.py

import json
import os
import sys
import uuid
import re
import random
from datetime import datetime
# Ensure Union is imported for Python 3.9 compatibility
from typing import List, Dict, Any, Tuple, Union

# Assuming your agent and tools are importable from the current structure
# Make sure 'agent.py' has NOT been modified to remove get_user_input_tool
from agent import graph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

# --- Configuration ---
COSQL_JSON_PATH = "/Users/jostarchen/NotCloud/cosql_dataset/cosql_all_info_dialogs.json"
COSQL_DB_BASE_PATH = "/Users/jostarchen/NotCloud/cosql_dataset/database"
OUTPUT_RESULTS_FILE = "cosql_test_results.json"
LOG_FILE = "logs/cosql_test_runner.log"

# --- !!! TESTING CONFIGURATION !!! ---
# Set MAX_DIALOGUES_TO_TEST to a number (e.g., 20) for cost-effective initial testing.
# Set to None to run on the entire dataset.
MAX_DIALOGUES_TO_TEST = 5 # <-- EDIT THIS VALUE FOR TESTING
HISTORY_LIMIT = 10 # Max number of recent messages (Human/AI non-tool-call) to pass to agent
# --- END TESTING CONFIGURATION ---

# --- Ensure Log Directory Exists ---
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# --- Logging Function ---
def log(message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # Added milliseconds
    log_entry = f"{timestamp} - {message}"
    print(log_entry) # Print to console
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    except Exception as e:
        print(f"{timestamp} - ERROR: Failed to write to log file: {e}")

# --- Helper to Parse Agent Stream ---
# Uses Union for Python 3.9 compatibility
def parse_agent_stream_for_results(events: List[Dict[str, Any]]) -> Tuple[Union[str, None], Union[Any, None], Union[str, None]]:
    """
    Parses the stream of events from LangGraph to find generated SQL,
    its execution result, and the final NL response.
    """
    generated_sql = None
    sql_execution_result = None
    final_nl_response = None
    last_ai_message_content = None
    tool_call_id_map: Dict[str, str] = {} # Maps tool_call_id to generated SQL

    for event in events:
        messages = event.get("messages", [])
        if not messages: continue
        if not isinstance(messages, list): messages = [messages]
        last_message = messages[-1] if messages else None
        if not last_message: continue

        # 1. Look for AI message making a tool call to execute_sql_query
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            for tc in last_message.tool_calls:
                # Record the *first* SQL query attempt in the turn
                if tc.get('name') == 'execute_sql_query':
                    if generated_sql is None and 'query' in tc.get('args', {}):
                        generated_sql = tc['args']['query']
                        tool_call_id_map[tc['id']] = generated_sql # Map ID to SQL
                        log(f"DEBUG: Parser found potential generated SQL: {generated_sql} (Tool Call ID: {tc['id']})")
                # Note: We don't break here in case other tool calls happen in the same message

        # 2. Look for ToolMessage containing the result for the execute_sql_query call
        if isinstance(last_message, ToolMessage):
             if last_message.tool_call_id in tool_call_id_map: # Check if it matches a logged SQL call
                 sql_execution_result = last_message.content
                 log(f"DEBUG: Parser found SQL execution result for call ID {last_message.tool_call_id}: {sql_execution_result}")
                 try:
                     # Attempt to parse if result is a JSON string (common for list/dict results)
                     parsed_result = json.loads(sql_execution_result)
                     sql_execution_result = parsed_result
                 except (json.JSONDecodeError, TypeError):
                      pass # Keep as string if not parsable JSON

        # 3. Look for the final AI message (likely the NL response)
        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
            # Consider this the final response if it follows some tool activity or is the only AI msg
            if generated_sql is not None or sql_execution_result is not None or not tool_call_id_map:
                 final_nl_response = last_message.content
                 # log(f"DEBUG: Parser found potential final NL response: {final_nl_response}") # Less verbose
            # Keep track of the last seen AI message content regardless
            last_ai_message_content = last_message.content

    # Fallback: If no specific NL response was identified after potential tool use,
    # use the very last AI message content encountered.
    if final_nl_response is None and last_ai_message_content is not None:
        final_nl_response = last_ai_message_content
        log(f"DEBUG: Parser using last AI message as NL response (fallback): {final_nl_response}")

    # If get_user_input_tool was called, the tool itself (modified version) should return a specific string.
    # Check if the *result* of a tool call contains that string.
    if isinstance(sql_execution_result, str) and "User input unavailable" in sql_execution_result:
        log("DEBUG: Detected that user input was requested by the agent.")
        # Override NL response to indicate input request. SQL fields remain None or as parsed before request.
        final_nl_response = "AGENT_REQUESTED_USER_INPUT"


    return generated_sql, sql_execution_result, final_nl_response

# --- Main Test Execution ---
def run_cosql_tests():
    log("Starting CoSQL Test Run...")
    if MAX_DIALOGUES_TO_TEST is not None:
        log(f"--- RUNNING LIMITED TEST: Processing a maximum of {MAX_DIALOGUES_TO_TEST} dialogues ---")
    else:
        log("--- RUNNING FULL TEST: Processing all dialogues ---")

    # Load CoSQL data
    try:
        with open(COSQL_JSON_PATH, 'r', encoding='utf-8') as f:
            cosql_data = json.load(f)
        log(f"Successfully loaded CoSQL data ({len(cosql_data)} dialogues) from: {COSQL_JSON_PATH}")
    except FileNotFoundError:
        log(f"ERROR: CoSQL JSON file not found at {COSQL_JSON_PATH}")
        return
    except json.JSONDecodeError as e:
        log(f"ERROR: Failed to parse CoSQL JSON file: {e}")
        return
    except Exception as e:
        log(f"ERROR: An unexpected error occurred loading CoSQL data: {e}")
        return

    all_results = []
    processed_dialogue_count = 0

    # --- START: Dialogue Sampling Logic ---
    all_dialogue_ids = list(cosql_data.keys())
    total_dialogues = len(all_dialogue_ids)

    if MAX_DIALOGUES_TO_TEST is not None and MAX_DIALOGUES_TO_TEST > 0 and MAX_DIALOGUES_TO_TEST < total_dialogues:
        log(f"Sampling {MAX_DIALOGUES_TO_TEST} dialogues randomly from {total_dialogues} total dialogues.")
        random.shuffle(all_dialogue_ids)
        dialogue_ids_to_process = all_dialogue_ids[:MAX_DIALOGUES_TO_TEST]
    elif MAX_DIALOGUES_TO_TEST is not None and MAX_DIALOGUES_TO_TEST <= 0:
        log("Warning: MAX_DIALOGUES_TO_TEST is set to 0 or negative. No dialogues will be processed.")
        dialogue_ids_to_process = []
    else: # Process all if None or >= total
        log("Processing all available dialogues.")
        dialogue_ids_to_process = all_dialogue_ids
    # --- END: Dialogue Sampling Logic ---

    # Iterate through the selected dialogue IDs
    for dialogue_id in dialogue_ids_to_process:
        dialogue_data = cosql_data[dialogue_id]
        processed_dialogue_count += 1

        log(f"\n--- Processing Dialogue {processed_dialogue_count}/{len(dialogue_ids_to_process)} | ID: {dialogue_id} ---")
        db_id = dialogue_data.get('db_id')
        if not db_id:
            log(f"WARNING: Missing 'db_id' for dialogue {dialogue_id}. Skipping.")
            continue

        db_path = os.path.join(COSQL_DB_BASE_PATH, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            log(f"WARNING: Database file not found at {db_path} for dialogue {dialogue_id}. Skipping.")
            continue

        conversation_history: List[BaseMessage] = []
        turns = dialogue_data.get('turns', [])

        # Iterate through turns in the dialogue
        for turn_index, turn_data in enumerate(turns):
            is_user = turn_data.get('isUser', False)
            text = turn_data.get('text', '')
            turn_id = turn_data.get('_id', f'turn_{turn_index}')

            if is_user:
                current_user_message = HumanMessage(content=text)
                conversation_history.append(current_user_message)
                log(f"Running agent for user turn {turn_index}...")

                # Set environment variables for this agent run
                os.environ['TARGET_DB_PATH'] = db_path
                os.environ['RUNNING_NON_INTERACTIVE'] = 'true' # For tools.py modification

                thread_id = str(uuid.uuid4())
                config = { "configurable": {"thread_id": thread_id}, "recursion_limit": 50 }

                # Prepare history slice for API call (limited and filtered)
                history_slice = conversation_history[-HISTORY_LIMIT:]
                filtered_history_slice = []
                for msg in history_slice:
                    if isinstance(msg, ToolMessage):
                        continue # Skip Tool messages
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        continue # Skip AI messages that are just tool calls
                    filtered_history_slice.append(msg)

                current_state = {"messages": filtered_history_slice}
                log(f"DEBUG: Passing {len(filtered_history_slice)} messages (after filtering Tool/AI-ToolCall messages) to agent.")

                events = []
                final_state = None
                generated_sql = None
                sql_execution_result = None
                agent_nl_response = None

                try:
                    stream = graph.stream(current_state, config, stream_mode="values")
                    for event in stream:
                         events.append(event)
                         final_state = event

                    # Parse results from the collected events
                    generated_sql, sql_execution_result, agent_nl_response = parse_agent_stream_for_results(events) # Removed flag

                    # Add agent's actual response messages back to the FULL history for context
                    if final_state and final_state.get("messages"):
                        # Determine index based on potentially different lengths
                        input_msg_count_sent = len(filtered_history_slice)
                        new_messages = []
                        if len(final_state["messages"]) >= input_msg_count_sent:
                             # Simple assumption check
                             match = True
                             for i in range(input_msg_count_sent):
                                 if i >= len(final_state["messages"]) or \
                                    final_state["messages"][i].content != filtered_history_slice[i].content:
                                     match = False
                                     break
                             if match:
                                 new_messages = final_state["messages"][input_msg_count_sent:]
                             else:
                                 log(f"WARN: History mismatch detected adding agent response @ T{turn_index}. Adding last message only.")
                                 # Add only the very last message from the agent's final state
                                 new_messages = final_state["messages"][-1:]
                        else:
                             log(f"WARN: Final state shorter than input @ T{turn_index}. Adding last message only.")
                             new_messages = final_state["messages"][-1:] # Add just the last one

                        if new_messages:
                            # Filter ToolMessages before adding to history if needed? Or keep? Keep for now.
                            conversation_history.extend(new_messages)
                            # log(f"DEBUG: Added {len(new_messages)} agent message(s) back to history.")


                except Exception as e:
                    log(f"ERROR: Agent execution failed for dialogue {dialogue_id}, turn {turn_index}: {e}")
                    agent_nl_response = f"ERROR: Agent Exception - {type(e).__name__}: {e}"
                finally:
                    # CRITICAL: Unset environment variables
                    if 'TARGET_DB_PATH' in os.environ: del os.environ['TARGET_DB_PATH']
                    if 'RUNNING_NON_INTERACTIVE' in os.environ: del os.environ['RUNNING_NON_INTERACTIVE']

                # --- Store Results for this User Turn ---
                turn_result = {
                    "dialogue_id": dialogue_id,
                    "turn_index_user": turn_index,
                    "user_query": text,
                    "db_id": db_id,
                    "ground_truth_sql": turns[turn_index + 1].get('rawSql') if turn_index + 1 < len(turns) and turns[turn_index + 1].get('isSql') else None,
                    "ground_truth_sql_result": turns[turn_index + 1].get('sql_result') if turn_index + 1 < len(turns) and turns[turn_index + 1].get('isSql') else None,
                    "ground_truth_nl_response": turns[turn_index + 2].get('text') if turn_index + 2 < len(turns) and not turns[turn_index + 2].get('isSql') else None,
                    "agent_generated_sql": generated_sql,
                    "agent_sql_execution_result": sql_execution_result,
                    "agent_nl_response": agent_nl_response,
                }
                all_results.append(turn_result)

            else:
                # If it's an assistant turn, add its ground truth NL response to history
                if not is_user and not turn_data.get('isSql') and text:
                   assistant_message = AIMessage(content=text)
                   conversation_history.append(assistant_message)
                # We could also add the assistant's SQL and Tool Result if available and needed for context,
                # but let's keep it simpler for now.
                pass

    # Save all results
    try:
        timestamp_str = datetime.now().strftime("%m%d_%H%M%S")
        log_output_path = f"logs/run_cosql_test_{timestamp_str}.json"
        with open(log_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        log(f"Successfully saved {len(all_results)} results to: {log_output_path}")
    except Exception as e:
        log(f"ERROR: Failed to save results to JSON file: {e}")

    log(f"CoSQL Test Run Finished. Processed {processed_dialogue_count} dialogues.")


if __name__ == "__main__":
    # --- Check Prerequisites ---
    if not os.path.exists(COSQL_JSON_PATH):
         log(f"FATAL ERROR: CoSQL JSON input file not found at {COSQL_JSON_PATH}. Please check the path.")
         sys.exit(1)
    if not os.path.exists(COSQL_DB_BASE_PATH) or not os.path.isdir(COSQL_DB_BASE_PATH):
         log(f"FATAL ERROR: CoSQL database base directory not found or is not a directory at {COSQL_DB_BASE_PATH}. Please check the path.")
         sys.exit(1)

    run_cosql_tests()