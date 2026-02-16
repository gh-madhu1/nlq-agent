"""
SQL Evaluator — deterministic self-evaluation of generated SQL.

Validates that generated SQL only references tables and columns that
actually exist in the database schema. No LLM involved — pure code
analysis for zero hallucination risk.
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Set

logger = logging.getLogger(__name__)


class SQLEvaluator:
    """
    Deterministic SQL evaluator that checks generated SQL against the real schema.

    Catches errors like:
    - Referencing non-existent tables
    - Referencing non-existent columns
    - Using columns from wrong tables
    - Missing GROUP BY for aggregate queries
    """

    def __init__(self, schema_info: Dict[str, Dict[str, Any]]):
        self.schema_info = schema_info

        # Build lookup maps
        self._table_names: Set[str] = set(schema_info.keys())
        self._table_columns: Dict[str, Set[str]] = {}
        self._all_columns: Set[str] = set()

        for table, info in schema_info.items():
            cols = set()
            for col_def in info.get("columns", []):
                col_name = col_def.split()[0]
                cols.add(col_name)
                self._all_columns.add(col_name)
            self._table_columns[table] = cols

    def evaluate(self, sql: str) -> Tuple[bool, str]:
        """
        Evaluate SQL against the schema.

        Returns: (is_valid, error_detail)
        If invalid, error_detail explains what's wrong so the generator can fix it.
        """
        sql_upper = sql.upper().strip()

        # 1. Check referenced tables exist
        table_error = self._check_tables(sql)
        if table_error:
            return False, table_error

        # 2. Check referenced columns exist
        column_error = self._check_columns(sql)
        if column_error:
            return False, column_error

        # 3. Check GROUP BY consistency
        groupby_error = self._check_group_by(sql)
        if groupby_error:
            return False, groupby_error

        return True, ""

    def _check_tables(self, sql: str) -> str:
        """Verify all referenced tables exist in schema."""
        # Extract table names after FROM and JOIN keywords
        pattern = r'\b(?:FROM|JOIN)\s+(\w+)\b'
        referenced = re.findall(pattern, sql, re.IGNORECASE)

        for table in referenced:
            if table.lower() not in {t.lower() for t in self._table_names}:
                # Could be a subquery alias — skip common SQL keywords
                if table.upper() in ('SELECT', 'WHERE', 'GROUP', 'ORDER', 'HAVING',
                                     'LIMIT', 'UNION', 'DISTINCT', 'AS', 'ON', 'AND',
                                     'OR', 'NOT', 'IN', 'EXISTS', 'BETWEEN', 'LIKE',
                                     'NULL', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END'):
                    continue
                available = ", ".join(sorted(self._table_names))
                return (f"Table '{table}' does not exist. "
                        f"Available tables: {available}")

        return ""

    def _check_columns(self, sql: str) -> str:
        """Verify referenced columns belong to the correct tables."""
        # Build alias → table mapping from the SQL
        alias_map = self._extract_aliases(sql)

        # Find table.column or alias.column references
        qualified_pattern = r'\b(\w+)\.(\w+)\b'
        matches = re.findall(qualified_pattern, sql, re.IGNORECASE)

        for table_or_alias, column in matches:
            # Skip SQL keywords and aggregate functions
            if table_or_alias.upper() in ('COUNT', 'SUM', 'AVG', 'MAX', 'MIN',
                                           'GROUP', 'ORDER', 'CASE', 'WHEN'):
                continue

            # Resolve alias to table name
            actual_table = alias_map.get(table_or_alias.lower(),
                                          table_or_alias.lower())

            # Find the matching table (case-insensitive)
            matched_table = None
            for t in self._table_names:
                if t.lower() == actual_table:
                    matched_table = t
                    break

            if matched_table is None:
                # Might be a subquery alias, skip
                continue

            # Check if column exists in that table
            table_cols = {c.lower() for c in self._table_columns.get(matched_table, set())}
            if column.lower() not in table_cols and column != '*':
                available_cols = ", ".join(sorted(self._table_columns.get(matched_table, set())))
                return (f"Column '{column}' does not exist in table '{matched_table}'. "
                        f"Available columns in {matched_table}: {available_cols}")

        return ""

    def _check_group_by(self, sql: str) -> str:
        """Check that aggregate functions have corresponding GROUP BY."""
        sql_upper = sql.upper()

        # Check for aggregate functions
        has_aggregate = bool(re.search(
            r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql_upper
        ))

        if not has_aggregate:
            return ""

        # If there's an aggregate, check if non-aggregated columns need GROUP BY
        has_group_by = 'GROUP BY' in sql_upper

        # Extract SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper, re.DOTALL)
        if not select_match:
            return ""

        select_clause = select_match.group(1)

        # If there are non-aggregated columns in SELECT but no GROUP BY
        # Check for plain columns (not inside aggregate functions)
        # Remove aggregated expressions first
        cleaned = re.sub(r'(COUNT|SUM|AVG|MAX|MIN)\s*\([^)]*\)', '', select_clause)
        # Check if remaining has column references
        has_plain_columns = bool(re.search(r'\b\w+\.\w+\b', cleaned))

        if has_plain_columns and not has_group_by:
            return ("Query uses aggregate functions with non-aggregated columns "
                    "but is missing a GROUP BY clause.")

        return ""

    def _extract_aliases(self, sql: str) -> Dict[str, str]:
        """Extract table alias mappings from SQL."""
        alias_map: Dict[str, str] = {}

        # Match "table_name alias" or "table_name AS alias"
        patterns = [
            r'\b(?:FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?(\w+)\b',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, sql, re.IGNORECASE):
                table = match.group(1).lower()
                alias = match.group(2).lower()

                # Skip if alias is a SQL keyword
                if alias.upper() in ('ON', 'WHERE', 'SET', 'JOIN', 'LEFT', 'RIGHT',
                                      'INNER', 'OUTER', 'CROSS', 'GROUP', 'ORDER',
                                      'HAVING', 'LIMIT', 'AS', 'AND', 'OR', 'NOT'):
                    continue

                if table in {t.lower() for t in self._table_names}:
                    alias_map[alias] = table

        return alias_map

    def get_correction_hint(self, sql: str, error: str) -> str:
        """Generate a detailed correction hint for the generator."""
        hints = [f"ERROR: {error}"]

        # Add the full schema as reference
        hints.append("\nCorrect schema reference:")
        for table, cols in self._table_columns.items():
            col_list = ", ".join(sorted(cols))
            hints.append(f"  {table}: {col_list}")

        return "\n".join(hints)
