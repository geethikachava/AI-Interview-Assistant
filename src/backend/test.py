import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = "interview_data.db"
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = sqlite3.connect("interview_data.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM users").fetchall()
    for row in rows:
        print(dict(row))

if __name__ == "__main__":
    init_db()