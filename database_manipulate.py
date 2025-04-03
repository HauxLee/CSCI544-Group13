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
from tools import get_database_schema, execute_sql_query

def init_database_connection(config_path=None):
    """Initialize database and create a connection.

    Args:
        config_path (str, optional): Path to database config file.
            Defaults to 'config/database_config.yaml'

    Returns:
        DatabaseManager: Initialized database manager instance
    """
    if config_path is None:
        config_path = os.path.join('config', 'database_config.yaml')

    print("Initializing database...")
    initialize_database(config_path)

    print("Creating database manager...")
    return DatabaseManager(config_path)

def test_basic_operations(db_manager):
    """Test basic database operations like logging and retrieving conversations.

    Args:
        db_manager (DatabaseManager): Database manager instance
    """
    print("\nTest 1: Logging a conversation...")
    db_manager.log_conversation(
        "What's the weather like?",
        "The weather is sunny today!"
    )
    print("Successfully logged conversation")

    print("\nTest 2: Retrieving conversation history...")
    history = db_manager.get_conversation_history(limit=5)
    print(f"Retrieved {len(history)} conversations:")
    for conv in history:
        print(f"Timestamp: {conv['timestamp']}")
        print(f"User: {conv['user_input']}")
        print(f"Agent: {conv['agent_response']}")
        print("-" * 50)

def test_query_operations(db_manager):
    """Test SQL query operations.

    Args:
        db_manager (DatabaseManager): Database manager instance
    """
    print("\nTest 3: Direct query execution...")
    if db_manager.db_type == 'sqlite':
        result = db_manager.execute_query("""
            SELECT COUNT(*) as count
            FROM conversations
        """)
        print(f"Total conversations in database: {result[0]['count']}")

def test_batch_operations(db_manager):
    """Test batch insert operations.

    Args:
        db_manager (DatabaseManager): Database manager instance
    """
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

def test_schema_operations(db_manager):
    """Test database schema operations using tools.py functions.

    Args:
        db_manager (DatabaseManager): Database manager instance
    """
    print("\nTest Schema Operations:")
    try:
        # Test get_database_schema
        print("Testing get_database_schema...")
        schema = get_database_schema()

        print("\nDatabase Schema:")
        for table_name, table_info in schema['tables'].items():
            print(f"\nTable: {table_name}")
            for column in table_info['columns']:
                print(f"  - Column: {column['column_name']}")
                print(f"    Type: {column['data_type']}")
                print(f"    Primary Key: {column['is_primary_key']}")
                print(f"    Nullable: {not column['not_null']}")
                if column['default_value']:
                    print(f"    Default: {column['default_value']}")
    except Exception as e:
        print(f"Schema operation failed: {e}")

def test_query_tool_operations(db_manager):
    """Test SQL query operations using tools.py functions.

    Args:
        db_manager (DatabaseManager): Database manager instance
    """
    print("\nTest Query Tool Operations:")
    try:
        # Test 1: Simple SELECT query
        print("\n1. Testing simple SELECT query...")
        result = execute_sql_query("SELECT COUNT(*) as count FROM conversations")
        print(f"Total conversations: {result[0]['count']}")

        # Test 2: Query with conditions
        print("\n2. Testing query with WHERE clause...")
        result = execute_sql_query("""
            SELECT * FROM conversations
            WHERE user_input LIKE '%weather%'
            LIMIT 3
        """)
        print(f"Found {len(result)} conversations about weather:")
        for row in result:
            print(f"User: {row['user_input']}")
            print(f"Agent: {row['agent_response']}")
            print("-" * 30)

        # Test 3: Query with ordering
        print("\n3. Testing query with ORDER BY...")
        result = execute_sql_query("""
            SELECT timestamp, user_input
            FROM conversations
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        print("Latest 5 conversations:")
        for row in result:
            print(f"Time: {row['timestamp']}")
            print(f"Question: {row['user_input']}")
            print("-" * 30)

    except Exception as e:
        print(f"Query operation failed: {e}")

def test_database_operations():
    """Main function to test various database operations"""
    db_manager = None
    try:
        # Initialize database and get manager
        db_manager = init_database_connection()

        # Run different test suites
        test_basic_operations(db_manager)
        test_query_operations(db_manager)
        test_batch_operations(db_manager)

        # Add new test functions for tools.py
        test_schema_operations(db_manager)
        test_query_tool_operations(db_manager)

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
        if db_manager and db_manager.manager:
            db_manager.disconnect()
            print("\nDatabase connection closed")

if __name__ == "__main__":
    test_database_operations()

