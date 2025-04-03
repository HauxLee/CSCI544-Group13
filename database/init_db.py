from .db_manager import DatabaseManager
import os

def initialize_sqlite_schema(db_manager):
    """Initialize SQLite database schema"""
    schemas = [
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_input TEXT NOT NULL,
            agent_response TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
        ON conversations(timestamp)
        """
    ]

    for schema in schemas:
        db_manager.execute_query(schema)

def initialize_mongodb_schema(db_manager):
    """Initialize MongoDB database schema (if needed)"""
    # MongoDB is schemaless, but we can create indexes
    if db_manager.db_type == 'mongodb':
        db_manager.db.conversations.create_index('timestamp')

def initialize_database(config_path):
    """Initialize the database based on configuration"""
    db_manager = DatabaseManager(config_path)
    try:
        if db_manager.db_type == 'sqlite':
            initialize_sqlite_schema(db_manager)
        else:
            initialize_mongodb_schema(db_manager)

        print(f"Successfully initialized {db_manager.db_type} database")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        db_manager.disconnect()

if __name__ == "__main__":
    initialize_database()