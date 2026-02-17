"""
SQL Generator — plan-driven SQL generation with strict plan adherence.
"""

import re
import logging
from typing import Dict, Any, Optional

from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class SQLGenerator:
    """
    Plan-driven SQL generator that strictly follows the deterministic plan.
    """

    def __init__(self, llm_client: LLMClient, schema_info: Dict[str, Dict[str, Any]] = None):
        self.llm_client = llm_client
        self.schema_info = schema_info or {}

    def generate_sql(self, query: str, schema_context: str,
                     plan: Optional[Dict[str, Any]] = None,
                     error_feedback: str = "") -> str:
        """Generate SQL from natural language, strictly following the query plan."""

        if not plan:
            response = self.llm_client.call(
                prompt=f"Write a SQLite SELECT query for: {query}"
            )
        else:
            # Build schema for ONLY the tables in the plan
            schema_lines = []
            for table in plan.get("tables", []):
                if table in self.schema_info:
                    cols = self.schema_info[table].get("columns", [])
                    fks = self.schema_info[table].get("foreign_keys", [])
                    
                    # Build column list with types: "col_name (TYPE)"
                    col_parts = []
                    for col_def in cols:
                        parts = col_def.split()
                        col_name = parts[0]
                        col_type = parts[1] if len(parts) > 1 else "TEXT"
                        col_parts.append(f"{col_name} ({col_type})")
                    
                    schema_lines.append(f"{table}: {', '.join(col_parts)}")
                    
                    # Add foreign keys if present
                    if fks:
                        schema_lines.append(f"  FK: {', '.join(fks)}")
            
            schema_str = "\n".join(schema_lines)
            
            # Build FROM clause
            tables = plan.get("tables", [])
            if not tables:
                from_clause = ""
            elif len(tables) == 1:
                from_clause = f"FROM {tables[0]}"
            else:
                # Build JOINs from plan
                from_clause = f"FROM {tables[0]}"
                for join in plan.get("joins", []):
                    from_clause += f" JOIN {join['to']} ON {join['on']}"
            
            # Build WHERE clause
            where_parts = []
            for w in plan.get("where_conditions", []):
                where_parts.append(f"{w['column']} {w['op']} {w['value']}")
            where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
            
            # Build GROUP BY
            group_by = plan.get("group_by", [])
            group_clause = f"GROUP BY {', '.join(group_by)}" if group_by else ""
            
            # Build ORDER BY
            order_by = plan.get("order_by", [])
            if order_by:
                order_parts = [f"{o['column']} {o['direction']}" for o in order_by]
                order_clause = f"ORDER BY {', '.join(order_parts)}"
            else:
                order_clause = ""
            
            # Build LIMIT
            limit_clause = f"LIMIT {plan['limit']}" if plan.get("limit") else ""
            
            # Construct the template
            template_parts = ["SELECT <columns>", from_clause, where_clause, group_clause, order_clause, limit_clause]
            template = " ".join([p for p in template_parts if p]).strip()
            
            # Get suggested columns from plan
            suggested_columns = plan.get("select_columns", ["*"])
            columns_hint = ", ".join(suggested_columns) if suggested_columns != ["*"] else "appropriate columns"

            # Construct the system instructions
            system_prompt = f"""You are a specialized SQL Generator. Generate ONLY a valid SQLite SELECT query. No explanations or comments.

SCHEMA (use ONLY these tables and columns):
{schema_str}

CRITICAL RULES - FOLLOW EXACTLY:
1. Do NOT add any WHERE clause unless shown in REQUIRED STRUCTURE below
2. Do NOT add any JOIN unless shown in REQUIRED STRUCTURE below  
3. Do NOT add any GROUP BY unless shown in REQUIRED STRUCTURE below
4. Do NOT add any ORDER BY unless shown in REQUIRED STRUCTURE below
5. Use ONLY the tables and columns from the SCHEMA above
6. Use table.column format for all columns (e.g., users.email, products.name)
7. If specific columns are suggested, use ONLY those columns
8. Do NOT use SELECT * if specific columns are available
"""
            # Construct the user instructions and placeholders
            user_prompt = f"""USER QUESTION: "{query}"

REQUIRED STRUCTURE (DO NOT MODIFY):
{template}

YOUR TASK:
Generate the complete SQLite SQL query. Use these columns: {columns_hint}
"""
            if error_feedback:
                user_prompt += f"\n\nPREVIOUS ERROR:\n{error_feedback}\nGenerate a DIFFERENT, correct query."

            response = self.llm_client.call(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=200,
                stop=["\n\n", "Question:", "SCHEMA:", "RULES:"]
            )

        sql = self._extract_sql(response)
        logger.info(f"Generated SQL: {sql}")
        return sql

    def _extract_sql(self, response: str) -> str:
        """Extract clean SQL from LLM response."""
        text = response.strip()

        # Remove markdown fences
        text = re.sub(r'```\w*', '', text)

        # Find SQL: marker
        sql_match = re.search(r'SQL:\s*(.*)', text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            text = sql_match.group(1).strip()

        # Find SELECT statement
        select_match = re.search(r'(SELECT\s+.+)', text, re.DOTALL | re.IGNORECASE)
        if select_match:
            text = select_match.group(1)
        elif "SELECT" not in text.upper():
            # No SQL found
            return text.strip()

        # Clean up
        text = self._clean_sql(text)
        return text

    def _clean_sql(self, sql: str) -> str:
        """Clean up model output."""
        sql = sql.strip()

        # Remove junk
        sql = re.sub(r'[!?]{2,}', '', sql)
        sql = re.sub(r'```\w*', '', sql)

        # Stop at non-SQL content
        lines = []
        for line in sql.split("\n"):
            stripped = line.strip()
            if not stripped:
                if lines:
                    break
                continue
            if stripped.startswith(("--", "Note", "Reasoning", "This query",
                                    "This will", "Explanation", "#", "Q:", "`")):
                if lines:
                    break
                continue
            lines.append(line)

        sql = " ".join(lines).strip()
        sql = sql.rstrip(';').strip()
        return sql
