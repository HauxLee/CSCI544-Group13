from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class DatabaseError(Exception):
    """Custom exception for database operations"""

    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

class ConnectionError(DatabaseError):
    """Raised when database connection fails"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "DB_CONN_001", details)

class AuthenticationError(DatabaseError):
    """Raised when database authentication fails"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "DB_AUTH_001", details)

class QueryError(DatabaseError):
    """Raised when query execution fails"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "DB_QUERY_001", details)

class ConfigError(DatabaseError):
    """Raised when configuration is invalid or missing"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "DB_CONFIG_001", details)

class DataError(DatabaseError):
    """Raised when data manipulation fails"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "DB_DATA_001", details)

class BaseManager(ABC):
    @abstractmethod
    def connect(self):
        """
        Establish connection to the database.

        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If authentication fails
            ConfigError: If configuration is invalid
        """
        pass

    @abstractmethod
    def disconnect(self):
        """
        Close database connection.

        Raises:
            ConnectionError: If disconnection fails
        """
        pass

    @abstractmethod
    def execute_query(self, query: str, params: Any = None) -> List[Any]:
        """
        Execute a database query.

        Args:
            query: SQL query string
            params: Query parameters

        Raises:
            QueryError: If query execution fails
            ConnectionError: If connection is lost
            DataError: If data manipulation fails
        """
        pass

    @abstractmethod
    def insert_one(self, table: str, data: Dict) -> Any:
        """
        Insert a single record.

        Args:
            table: Target table name
            data: Record data

        Raises:
            DataError: If insertion fails
            QueryError: If query is invalid
            ConnectionError: If connection is lost
        """
        pass

    @abstractmethod
    def insert_many(self, table: str, data: List[Dict]) -> Any:
        """
        Insert multiple records.

        Args:
            table: Target table name
            data: List of records

        Raises:
            DataError: If batch insertion fails
            QueryError: If query is invalid
            ConnectionError: If connection is lost
        """
        pass

    @abstractmethod
    def find(self, table: str, conditions: Dict = None) -> List[Dict]:
        """
        Find records matching conditions.

        Args:
            table: Target table name
            conditions: Search conditions

        Raises:
            QueryError: If query is invalid
            ConnectionError: If connection is lost
        """
        pass
