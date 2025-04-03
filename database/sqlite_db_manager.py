import sqlite3
import os
from typing import Any, Dict, List, Optional
from .base_db_manager import BaseManager, DatabaseError

class SQLiteManager(BaseManager):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None
        self.connect()

    def connect(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise DatabaseError(f"SQLite connection error: {e}")

    def disconnect(self):
        if self.connection:
            self.connection.close()

    def execute_query(self, query: str, params: Any = None) -> List[Any]:
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.connection.commit()
            return cursor.fetchall()
        except sqlite3.Error as e:
            self.connection.rollback()
            raise DatabaseError(f"SQLite query execution error: {e}")

    def insert_one(self, table: str, data: Dict) -> Any:
        self.assert_table_exists(table)
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        return self.execute_query(query, tuple(data.values()))

    def insert_many(self, table: str, data: List[Dict]) -> Any:
        if not data:
            return None
        self.assert_table_exists(table)
        columns = ', '.join(data[0].keys())
        placeholders = ', '.join(['?' for _ in data[0]])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        cursor = self.connection.cursor()
        try:
            cursor.executemany(query, [tuple(d.values()) for d in data])
            self.connection.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            self.connection.rollback()
            raise DatabaseError(f"SQLite batch insert error: {e}")

    def find(self, table: str, conditions: Dict = None) -> List[Dict]:
        self.assert_table_exists(table)
        query = f"SELECT * FROM {table}"
        if conditions:
            where_clause = ' AND '.join([f"{k} = ?" for k in conditions.keys()])
            query += f" WHERE {where_clause}"
            return self.execute_query(query, tuple(conditions.values()))
        return self.execute_query(query)

    def get_table_names(self) -> List[str]:
        """
        Returns a list of all table names in the SQLite database
        """
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        result = self.execute_query(query)
        return [row["name"] for row in result]

    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """
        Returns schema details (column name and type) for the given table
        """
        query = f"PRAGMA table_info({table_name})"
        result = self.execute_query(query)
        return [
            {
                "column_name": row["name"],
                "data_type": row["type"],
                "not_null": bool(row["notnull"]),
                "default_value": row["dflt_value"],
                "is_primary_key": bool(row["pk"])
            }
            for row in result
        ]

    def check_table_exists(self, table_name: str) -> bool:
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = self.execute_query(query, (table_name,))
        return len(result) > 0

    def assert_table_exists(self, table_name: str):
        if not self.check_table_exists(table_name):
            raise DatabaseError(f"The table '{table_name}' does not exist. Please initialize the schema first.")

    def update(self, table: str, data: Dict, conditions: Dict) -> int:
        """
        Update records in the specified table that match the conditions.

        Args:
            table (str): Name of the table to update
            data (Dict): Dictionary of column-value pairs to update
            conditions (Dict): Dictionary of column-value pairs for WHERE clause

        Returns:
            int: Number of rows affected

        Raises:
            DatabaseError: If update operation fails
        """
        self.assert_table_exists(table)

        try:
            # Build SET clause
            set_clause = ', '.join([f"{k} = ?" for k in data.keys()])

            # Build WHERE clause
            where_clause = ' AND '.join([f"{k} = ?" for k in conditions.keys()])

            # Combine parameters for both SET and WHERE clauses
            params = tuple(list(data.values()) + list(conditions.values()))

            # Construct the full query
            query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"

            cursor = self.connection.cursor()
            cursor.execute(query, params)
            self.connection.commit()

            return cursor.rowcount
        except sqlite3.Error as e:
            self.connection.rollback()
            raise DatabaseError(f"SQLite update error: {e}")

    def delete(self, table: str, conditions: Dict) -> int:
        """
        Delete records from the specified table that match the conditions.

        Args:
            table (str): Name of the table to delete from
            conditions (Dict): Dictionary of column-value pairs for WHERE clause

        Returns:
            int: Number of rows affected

        Raises:
            DatabaseError: If delete operation fails
        """
        self.assert_table_exists(table)

        try:
            # Build WHERE clause
            where_clause = ' AND '.join([f"{k} = ?" for k in conditions.keys()])

            # Construct the full query
            query = f"DELETE FROM {table} WHERE {where_clause}"

            cursor = self.connection.cursor()
            cursor.execute(query, tuple(conditions.values()))
            self.connection.commit()

            return cursor.rowcount
        except sqlite3.Error as e:
            self.connection.rollback()
            raise DatabaseError(f"SQLite delete error: {e}")
