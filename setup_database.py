# setup_database.py
import os
import sqlite3

def delete_database(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted existing database at {db_path}")
    else:
        print(f"No existing database found at {db_path}")

def setup_database():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY,
        filename_original TEXT,
        filename_server TEXT,
        model_name TEXT,
        prediction TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    conn.close()

if __name__ == "__main__":
    db_path = 'predictions.db'
    
    # Delete the existing database
    # delete_database(db_path)

    setup_database()
