# backend/services/newsletter_service.py

import re
from backend.database import get_db_connection
import sqlite3

def validate_email(email):
    """
    Validate the email address using a regex pattern.
    """
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email)

def save_email(email):
    """
    Save the email to the database.
    """
    if not validate_email(email):
        raise ValueError("Invalid email address")

    conn = get_db_connection()
    if conn is None:
        raise RuntimeError("Database connection failed")

    try:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO subscriptions (email) VALUES (?)', (email,))
        conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError("Email already subscribed")
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error: {e}")
    finally:
        conn.close()
