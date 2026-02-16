import re
import logging
from typing import Tuple, List, Optional, Dict, Any, Set

logger = logging.getLogger(__name__)

# Forbidden keywords for a read-only agent
_FORBIDDEN_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
    "TRUNCATE", "GRANT", "REVOKE", "CREATE", "EXEC",
}


class Validator:
    """
    SQL Validator — validates syntax, safety, and schema correctness.
    
    Combines syntax/safety checks with schema validation to ensure
    generated SQL is both safe and references valid tables/columns.
    """
    
    def __init__(self, schema_info: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Args:
            schema_info: Database schema for validation. If provided, enables schema checks.
        """
        self.schema_info = schema_info or {}
        
        # Build lookup maps for schema validation
        self._table_names: Set[str] = set(self.schema_info.keys())
        self._table_columns: Dict[str, Set[str]] = {}
        self._all_columns: Set[str] = set()
        
        for table, info in self.schema_info.items():
            cols = set()
            for col_def in info.get("columns", []):
                col_name = col_def.split()[0]
                cols.add(col_name)
                self._all_columns.add(col_name)
            self._table_columns[table] = cols
    
    def validate_sql(self, sql: str, known_tables: Optional[List[str]] = None) -> Tuple[bool, str]:
        """
        Validate generated SQL for syntax, safety, and schema correctness.
        
        Returns: (is_valid, error_message)
        """
        sql_stripped = sql.strip()

        # 1. Reject empty SQL
        if not sql_stripped:
            return False, "Generated SQL is empty."

        # 2. Check SQL length (reasonable bounds)
        if len(sql_stripped) < 10:
            return False, "SQL is too short to be valid."
        if len(sql_stripped) > 2000:
            return False, "SQL is suspiciously long."

        # 3. Must start with SELECT (case-insensitive)
        if not sql_stripped.upper().lstrip().startswith("SELECT"):
            return False, "Only SELECT statements are allowed."

        # 4. Reject multiple statements (semicolon not at the end)
        body = sql_stripped.rstrip(';')
        if ';' in body:
            return False, "Multiple SQL statements are not allowed."

        # 5. Check for forbidden keywords using word-boundary regex
        sql_upper = sql_stripped.upper()
        for keyword in _FORBIDDEN_KEYWORDS:
            pattern = rf'\b{keyword}\b'
            if re.search(pattern, sql_upper):
                return False, f"SQL contains forbidden keyword: {keyword}"

        # 6. Basic syntax checks
        # Check for balanced parentheses
        if sql_stripped.count('(') != sql_stripped.count(')'):
            return False, "Unbalanced parentheses in SQL."

        # Must have FROM clause
        if 'FROM' not in sql_upper:
            return False, "SQL missing FROM clause."

        # 7. Validate table names if known_tables provided
        if known_tables:
            # Extract table names from FROM and JOIN clauses
            table_pattern = r'\b(?:FROM|JOIN)\s+(\w+)\b'
            referenced_tables = re.findall(table_pattern, sql_stripped, re.IGNORECASE)
            for table in referenced_tables:
                if table.lower() not in [t.lower() for t in known_tables]:
                    return False, f"Unknown table referenced: {table}. Known tables: {', '.join(known_tables)}"
        
        # 8. Schema validation (if schema_info provided)
        if self.schema_info:
            is_valid_schema, schema_error = self._validate_schema(sql_stripped)
            if not is_valid_schema:
                return False, schema_error

        return True, ""
    
    # ------------------------------------------------------------------
    # Schema validation methods (merged from evaluator)
    # ------------------------------------------------------------------
    
    def _validate_schema(self, sql: str) -> Tuple[bool, str]:
        """Validate SQL against the schema."""
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
        alias_map = self._extract_aliases(sql)
        qualified_pattern = r'\b(\w+)\.(\w+)\b'
        matches = re.findall(qualified_pattern, sql, re.IGNORECASE)
        
        for table_or_alias, column in matches:
            # Skip SQL keywords and aggregate functions
            if table_or_alias.upper() in ('COUNT', 'SUM', 'AVG', 'MAX', 'MIN',
                                          'GROUP', 'ORDER', 'CASE', 'WHEN'):
                continue
            
            # Resolve alias to table name
            actual_table = alias_map.get(table_or_alias.lower(), table_or_alias.lower())
            
            # Find the matching table (case-insensitive)
            matched_table = None
            for t in self._table_names:
                if t.lower() == actual_table:
                    matched_table = t
                    break
            
            if matched_table is None:
                continue  # Might be a subquery alias
            
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
        has_aggregate = bool(re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql_upper))
        
        if not has_aggregate:
            return ""
        
        has_group_by = 'GROUP BY' in sql_upper
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper, re.DOTALL)
        if not select_match:
            return ""
        
        select_clause = select_match.group(1)
        cleaned = re.sub(r'(COUNT|SUM|AVG|MAX|MIN)\s*\([^)]*\)', '', select_clause)
        has_plain_columns = bool(re.search(r'\b\w+\.\w+\b', cleaned))
        
        if has_plain_columns and not has_group_by:
            return ("Query uses aggregate functions with non-aggregated columns "
                    "but is missing a GROUP BY clause.")
        return ""
    
    def _extract_aliases(self, sql: str) -> Dict[str, str]:
        """Extract table alias mappings from SQL."""
        alias_map: Dict[str, str] = {}
        pattern = r'\b(?:FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?(\w+)\b'
        
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
        
        if self.schema_info:
            hints.append("\nCorrect schema reference:")
            for table, cols in self._table_columns.items():
                col_list = ", ".join(sorted(cols))
                hints.append(f"  {table}: {col_list}")
        
        return "\n".join(hints)
