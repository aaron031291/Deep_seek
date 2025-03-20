import sqlite3
from datetime import datetime

class AdaptiveMemory:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        
    def create_tables(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS memory_store (
                id INTEGER PRIMARY KEY,
                content TEXT,
                importance FLOAT,
                last_accessed TIMESTAMP,
                access_count INTEGER
            )
        ''')
        
    def store(self, content, importance=0.5):
        self.conn.execute('''
            INSERT INTO memory_store (content, importance, last_accessed, access_count)
            VALUES (?, ?, ?, ?)
        ''', (content, importance, datetime.now(), 1))
        self.conn.commit()
        
    def retrieve(self, query):
        cursor = self.conn.execute('''
            SELECT content FROM memory_store 
            WHERE content LIKE ? 
            ORDER BY importance DESC, access_count DESC
            LIMIT 10
        ''', (f'%{query}%',))
        return cursor.fetchall()
