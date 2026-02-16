import time
import logging
from typing import Any, Dict, List, Tuple
from utils.db_manager import DBManager

logger = logging.getLogger(__name__)


class Executor:
    def __init__(self, db_manager: DBManager):
        self.db_manager = db_manager

    def execute_sql(self, sql: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Executes the SQL query with timing.
        Returns: (results, error_message)
        """
        try:
            start = time.time()
            results = self.db_manager.execute_query(sql)
            elapsed = time.time() - start
            logger.info(f"SQL executed in {elapsed:.3f}s — {len(results)} rows returned")
            return results, ""
        except TimeoutError as e:
            logger.warning(f"Query timed out: {sql}")
            return [], str(e)
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return [], str(e)
