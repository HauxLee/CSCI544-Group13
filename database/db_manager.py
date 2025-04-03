import yaml
import os
from datetime import datetime
from typing import Dict, Union, List, Optional
from .base_db_manager import BaseManager, DatabaseError
from .sqlite_db_manager import SQLiteManager
from .mongo_db_manager import MongoDBManager

class DatabaseManager:
    """
    A factory class for creating database manager instances.
    This class only handles initialization of database connections and doesn't expose specific database operations.
    """
    def __init__(self, config_path: str = os.path.join('config', 'database_config.yaml')):
        self.config_path = config_path
        self.manager: Optional[Union[SQLiteManager, MongoDBManager]] = None
        self.db_type: Optional[str] = None
        self._initialize_config()
        self.manager = self.create_manager(self.config_path)

    def _initialize_config(self) -> None:
        """Initialize configuration file if it doesn't exist"""
        if not os.path.exists(self.config_path):
            # Create default config directory
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            # Default configuration
            default_config = {
                'database': {
                    'type': 'sqlite',
                    'sqlite': {
                        'path': os.path.join('database', 'data', 'default.sqlite')
                    },
                    'mongodb': {
                        'uri': 'mongodb://localhost:27017',
                        'database': 'mydatabase'
                    }
                }
            }

            # Create data directory for SQLite
            os.makedirs(os.path.join('database', 'data'), exist_ok=True)

            # Write default config
            with open(self.config_path, 'w') as file:
                yaml.safe_dump(default_config, file)

    def create_manager(self, config_path: str = 'config/database_config.yaml') -> Union[SQLiteManager, MongoDBManager]:
        """
        Creates and returns a specific database manager instance based on configuration.

        Args:
            config_path (str): Path to the database configuration file

        Returns:
            Union[SQLiteManager, MongoDBManager]: An instance of the appropriate database manager

        Raises:
            DatabaseError: If configuration loading fails or database type is unsupported
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)['database']
        except Exception as e:
            raise DatabaseError(f"Failed to load database config: {e}")

        db_type = config['type']
        self.db_type = db_type

        if db_type == 'sqlite':
            raw_path = config['sqlite']['path']
            sqlite_path = self._normalize_path(raw_path)
            os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
            return SQLiteManager(sqlite_path)
        elif db_type == 'mongodb':
            return MongoDBManager(
                config['mongodb']['uri'],
                config['mongodb']['database']
            )
        else:
            raise DatabaseError(f"Unsupported database type: {db_type}")

    def _normalize_path(self, path: str) -> str:
        """
        Normalize a file path to be OS-independent and absolute.
        If path is relative, convert it to absolute path based on the project root.
        """
        normalized = os.path.normpath(path)

        if not os.path.isabs(normalized):
            normalized = os.path.abspath(normalized)

        return normalized

    @staticmethod
    def create_sqlite_manager(db_path: str) -> SQLiteManager:
        """
        Creates a SQLite database manager instance.

        Args:
            db_path (str): Path to the SQLite database file

        Returns:
            SQLiteManager: An instance of SQLite database manager
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return SQLiteManager(db_path)

    @staticmethod
    def create_mongodb_manager(uri: str, db_name: str) -> MongoDBManager:
        """
        Creates a MongoDB database manager instance.

        Args:
            uri (str): MongoDB connection URI
            db_name (str): Name of the database

        Returns:
            MongoDBManager: An instance of MongoDB database manager
        """
        return MongoDBManager(uri, db_name)

    def __getattr__(self, name):
        if self.manager is None:
            raise DatabaseError("Database manager not initialized")
        return getattr(self.manager, name)

    def log_conversation(self, user_input: str, agent_response: str) -> None:
        """Utility method to log conversations"""
        if self.manager is None:
            raise DatabaseError("Database manager not initialized")

        data = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'agent_response': agent_response
        }

        self.manager.insert_one('conversations', data)

    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Utility method to retrieve conversation history"""
        if self.manager is None:
            raise DatabaseError("Database manager not initialized")

        if isinstance(self.manager, SQLiteManager):
            return self.manager.execute_query(
                "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
        else:  # mongodb
            return list(self.manager.db['conversations'].find().sort('timestamp', -1).limit(limit))




