import sqlite3
import logging
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Default row limit to prevent runaway queries
DEFAULT_ROW_LIMIT = 1000
# Query timeout: abort after N SQLite VM steps (~5 seconds of work)
QUERY_TIMEOUT_STEPS = 5_000_000


class DBManager:
    def __init__(self, db_path: str = 'data/ecommerce.db'):
        self.db_path = db_path
        self._conn = None

    @property
    def connection(self) -> sqlite3.Connection:
        """Lazy, persistent connection. Reused across calls."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            # Set a progress handler as a query timeout guard
            self._conn.set_progress_handler(self._timeout_handler, QUERY_TIMEOUT_STEPS)
            logger.info(f"Opened persistent connection to {self.db_path}")
        return self._conn

    @staticmethod
    def _timeout_handler():
        """Returning non-zero aborts the currently running query."""
        return 1

    def execute_query(self, query: str, row_limit: int = DEFAULT_ROW_LIMIT) -> List[Dict[str, Any]]:
        """Execute a read query with a safety row limit."""
        # Append LIMIT if not already present
        q_upper = query.strip().rstrip(';').upper()
        if 'LIMIT' not in q_upper:
            query = query.strip().rstrip(';') + f" LIMIT {row_limit};"

        try:
            df = pd.read_sql_query(query, self.connection)
            return df.to_dict(orient='records')
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e).lower():
                raise TimeoutError(f"Query timed out after exceeding step limit: {query}")
            raise

    def get_schema_info(self) -> str:
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        schema_text = ""
        for table in tables:
            table_name = table[0]
            schema_text += f"\nTable: {table_name}\nColumns:\n"
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            for col in columns:
                schema_text += f"  - {col[1]} ({col[2]})\n"
        return schema_text

    def get_full_schema(self) -> dict:
        """
        Get full schema as a dictionary for vector store initialization.
        Returns: {table_name: {'columns': ['col TYPE', ...], 'foreign_keys': [...]}, ...}
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        schema_dict = {}
        for table in tables:
            table_name = table[0]
            
            # Get columns
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            col_list = [f"{col[1]} {col[2]}" for col in columns]
            
            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            fks = cursor.fetchall()
            fk_list = [f"{fk[3]} -> {fk[2]}.{fk[4]}" for fk in fks]
            
            schema_dict[table_name] = {
                'columns': col_list,
                'foreign_keys': fk_list
            }
        
        return schema_dict

    def get_table_names(self) -> List[str]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [t[0] for t in cursor.fetchall()]

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("Database connection closed.")

    def __del__(self):
        self.close()
