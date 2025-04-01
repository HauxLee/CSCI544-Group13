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
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        return self.execute_query(query, tuple(data.values()))

    def insert_many(self, table: str, data: List[Dict]) -> Any:
        if not data:
            return None
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
        query = f"SELECT * FROM {table}"
        if conditions:
            where_clause = ' AND '.join([f"{k} = ?" for k in conditions.keys()])
            query += f" WHERE {where_clause}"
            return self.execute_query(query, tuple(conditions.values()))
        return self.execute_query(query)