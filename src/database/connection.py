"""
Database Connection Management.

Responsible for creating and managing PostgreSQL connection engines and sessions.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config.settings import DATABASE_URL

def get_engine():
    """
    Create and return a SQLAlchemy engine using DATABASE_URL.
    """
    return create_engine(DATABASE_URL, pool_pre_ping=True)

def get_session():
    """
    Return a new SQLAlchemy database session instance.
    """
    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def test_connection(engine=None):
    """
    Execute a simple SELECT 1 query to test database connection.
    Returns (True, "Success message") or (False, "Error message").
    """
    if engine is None:
        engine = get_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1;"))
            row = result.fetchone()
            if row and row[0] == 1:
                return True, "Database connection successful."
            return False, f"Unexpected response from database: {row}"
    except Exception as e:
        return False, f"Database connection failed: {str(e)}"

