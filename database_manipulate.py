from database import DatabaseManager
from database.base_db_manager import (
    ConnectionError,
    AuthenticationError,
    QueryError,
    ConfigError,
    DataError
)
from database.init_db import initialize_database
import os
from datetime import datetime

def test_database_operations():
    """Test various database operations"""

    # First, initialize the database
    print("Initializing database...")
    config_path = os.path.join('config', 'database_config.yaml')
    initialize_database(config_path)

    # Create a database manager instance
    print("Creating database manager...")
    db_manager = DatabaseManager()

    try:
        # Test 1: Log a conversation
        print("\nTest 1: Logging a conversation...")
        db_manager.log_conversation(
            "What's the weather like?",
            "The weather is sunny today!"
        )
        print("Successfully logged conversation")

        # Test 2: Retrieve conversation history
        print("\nTest 2: Retrieving conversation history...")
        history = db_manager.get_conversation_history(limit=5)
        print(f"Retrieved {len(history)} conversations:")
        for conv in history:
            print(f"Timestamp: {conv['timestamp']}")
            print(f"User: {conv['user_input']}")
            print(f"Agent: {conv['agent_response']}")
            print("-" * 50)

        # Test 3: Direct query execution (SQLite only)
        print("\nTest 3: Direct query execution...")
        if db_manager.db_type == 'sqlite':
            result = db_manager.execute_query("""
                SELECT COUNT(*) as count
                FROM conversations
            """)
            print(f"Total conversations in database: {result[0]['count']}")

        # Test 4: Batch insert
        print("\nTest 4: Batch insert...")
        batch_data = [
            {
                'timestamp': datetime.now().isoformat(),
                'user_input': f'Test question {i}',
                'agent_response': f'Test response {i}'
            }
            for i in range(1, 4)
        ]
        db_manager.insert_many('conversations', batch_data)
        print("Successfully inserted batch data")

    except ConnectionError as e:
        print(f"Connection failed: {e}")
        print(f"Details: {e.details}")
    except AuthenticationError as e:
        print(f"Authentication failed: {e}")
    except QueryError as e:
        print(f"Query failed: {e}")
    except DataError as e:
        print(f"Data operation failed: {e}")
    except ConfigError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if db_manager.manager:
            db_manager.disconnect()
            print("\nDatabase connection closed")

if __name__ == "__main__":
    db_path = os.path.join('database', 'data', 'default.sqlite')
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database at {db_path}")

    # Run tests
    test_database_operations()