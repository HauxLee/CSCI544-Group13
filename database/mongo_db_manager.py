from typing import Any, Dict, List, Optional
from .base_db_manager import BaseManager, DatabaseError

try:
    from pymongo import MongoClient
    from pymongo.database import Database as MongoDatabase
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

class MongoDBManager(BaseManager):
    def __init__(self, uri: str, db_name: str):
        if not MONGODB_AVAILABLE:
            raise ImportError("pymongo is not installed. Please install it to use MongoDB.")
        self.uri = uri
        self.db_name = db_name
        self.client: Optional[MongoClient] = None
        self.db: Optional[MongoDatabase] = None
        self.connect()

    def connect(self):
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
        except Exception as e:
            raise DatabaseError(f"MongoDB connection error: {e}")

    def disconnect(self):
        if self.client:
            self.client.close()

    def execute_query(self, query: str, params: Any = None) -> List[Any]:
        raise NotImplementedError("Direct query execution is not supported for MongoDB")

    def insert_one(self, collection: str, data: Dict) -> Any:
        try:
            result = self.db[collection].insert_one(data)
            return result.inserted_id
        except Exception as e:
            raise DatabaseError(f"MongoDB insert error: {e}")

    def insert_many(self, collection: str, data: List[Dict]) -> Any:
        try:
            result = self.db[collection].insert_many(data)
            return result.inserted_ids
        except Exception as e:
            raise DatabaseError(f"MongoDB batch insert error: {e}")

    def find(self, collection: str, conditions: Dict = None) -> List[Dict]:
        try:
            return list(self.db[collection].find(conditions or {}))
        except Exception as e:
            raise DatabaseError(f"MongoDB find error: {e}")