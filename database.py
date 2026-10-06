import sqlite3
from typing import List, Dict, Any, Optional

DB_FILE = "tasks.db"


def get_connection(db_file: str = DB_FILE) -> sqlite3.Connection:
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_file: str = DB_FILE) -> None:
    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                priority TEXT NOT NULL CHECK(priority IN ('Low', 'Medium', 'High', 'Urgent')),
                deadline TEXT,
                category TEXT NOT NULL CHECK(category IN ('Work', 'Study', 'Personal', 'Shopping', 'Health', 'Finance', 'General')),
                status TEXT NOT NULL CHECK(status IN ('Pending', 'In Progress', 'Completed')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()


def insert_tasks(tasks: List[Dict[str, Any]], db_file: str = DB_FILE) -> int:
    if not tasks:
        return 0

    init_db(db_file)

    query = """
        INSERT INTO tasks (title, description, priority, deadline, category, status)
        VALUES (:title, :description, :priority, :deadline, :category, :status)
    """

    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.executemany(query, tasks)
        conn.commit()
        return cursor.rowcount


def get_all_tasks(db_file: str = DB_FILE) -> List[Dict[str, Any]]:
    init_db(db_file)
    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks ORDER BY id ASC")
        return [dict(row) for row in cursor.fetchall()]


def get_tasks_by_status(status: str, db_file: str = DB_FILE) -> List[Dict[str, Any]]:
    init_db(db_file)
    with get_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE status = ? ORDER BY id ASC", (status,))
        return [dict(row) for row in cursor.fetchall()]