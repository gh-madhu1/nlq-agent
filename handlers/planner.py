"""
Deterministic Query Planner.

Uses keyword matching and schema analysis to identify which tables,
columns, joins, and aggregations a query needs — with zero LLM calls.
JOIN paths are resolved via the foreign key graph.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Set

logger = logging.getLogger(__name__)


# Aggregation keyword patterns
_AGG_PATTERNS = {
    "COUNT": [r'\bhow many\b', r'\bcount\b', r'\bnumber of\b', r'\btotal number\b'],
    "SUM":   [r'\btotal\b(?!.*number)', r'\bsum\b', r'\boverall\b', r'\bcombined\b'],
    "AVG":   [r'\baverage\b', r'\bavg\b', r'\bmean\b'],
    "MAX":   [r'\bmost expensive\b', r'\bhighest\b', r'\bmax\b', r'\blargest\b', r'\bgreatest\b'],
    "MIN":   [r'\bcheapest\b', r'\blowest\b', r'\bmin\b', r'\bsmallest\b', r'\bleast\b'],
}

# Filter keyword patterns → (operator, value_extractor_hint)
_FILTER_PATTERNS = [
    (r'\bunder\s+\$?(\d+(?:\.\d+)?)', "<", None),
    (r'\bless than\s+\$?(\d+(?:\.\d+)?)', "<", None),
    (r'\bbelow\s+\$?(\d+(?:\.\d+)?)', "<", None),
    (r'\bover\s+\$?(\d+(?:\.\d+)?)', ">", None),
    (r'\bmore than\s+\$?(\d+(?:\.\d+)?)', ">", None),
    (r'\babove\s+\$?(\d+(?:\.\d+)?)', ">", None),
    (r'\bbetween\s+\$?(\d+(?:\.\d+)?)\s+and\s+\$?(\d+(?:\.\d+)?)', "BETWEEN", None),
    (r'\blast\s+(\d+)\s+days?\b', "DATE", "days"),
    (r'\blast\s+(\d+)\s+weeks?\b', "DATE", "weeks"),
    (r'\blast\s+(\d+)\s+months?\b', "DATE", "months"),
]

# Order keywords
_ORDER_PATTERNS = [
    (r'\bmost expensive\b', "price", "DESC"),
    (r'\bcheapest\b', "price", "ASC"),
    (r'\bhighest\b', "price", "DESC"),
    (r'\blowest\b', "price", "ASC"),
    (r'\bnewest\b', "order_date", "DESC"),
    (r'\boldest\b', "order_date", "ASC"),
    (r'\brecent\b', "order_date", "DESC"),
    (r'\blatest\b', "order_date", "DESC"),
]


class QueryPlanner:
    """
    Deterministic query planner — no LLM calls.

    Analyzes the user query against the database schema using keyword
    matching and produces a structured plan with tables, joins, columns,
    filters, aggregations, and ordering.
    """

    def __init__(self, schema: Dict[str, Dict[str, Any]], llm_client=None):
        """
        Args:
            schema: {table_name: {'columns': ['col TYPE', ...], 'foreign_keys': [...]}}
            llm_client: Unused, kept for backward compatibility.
        """
        self.schema = schema

        # Build lookup structures
        self._table_names = list(schema.keys())
        self._columns: Dict[str, List[str]] = {}
        self._col_types: Dict[str, Dict[str, str]] = {}
        self._fk_graph: Dict[str, List[Dict]] = {}

        for table, info in schema.items():
            cols, types = [], {}
            for col_def in info.get("columns", []):
                parts = col_def.split()
                col_name = parts[0]
                col_type = parts[1] if len(parts) > 1 else "TEXT"
                cols.append(col_name)
                types[col_name] = col_type.upper()
            self._columns[table] = cols
            self._col_types[table] = types

            fks = []
            for fk_str in info.get("foreign_keys", []):
                m = re.match(r"(\w+)\s*->\s*(\w+)\.(\w+)", fk_str)
                if m:
                    fks.append({
                        "from_col": m.group(1),
                        "to_table": m.group(2),
                        "to_col": m.group(3),
                    })
            self._fk_graph[table] = fks

        # Build keyword → table mapping for quick lookup
        self._keyword_table_map = self._build_keyword_map()

        # Build schema description for generator prompt
        self._schema_desc = self._build_schema_description()

        logger.info(f"QueryPlanner (deterministic) initialized with tables: {self._table_names}")

    # ------------------------------------------------------------------
    # Schema description (used by generator)
    # ------------------------------------------------------------------

    def _build_schema_description(self) -> str:
        """Build a complete schema description string."""
        lines = []
        for table, info in self.schema.items():
            cols = info.get("columns", [])
            fks = info.get("foreign_keys", [])
            lines.append(f"Table: {table}")
            lines.append(f"  Columns: {', '.join(cols)}")
            if fks:
                lines.append(f"  Foreign Keys: {', '.join(fks)}")
        return "\n".join(lines)

    def get_schema_description(self) -> str:
        """Return the full schema description for generator prompts."""
        return self._schema_desc

    def _build_schema_description_from(self, schema: Dict[str, Dict[str, Any]]) -> str:
        """Build schema description from a specific schema dict (e.g., retrieved schema)."""
        lines = []
        for table, info in schema.items():
            cols = info.get("columns", [])
            fks = info.get("foreign_keys", [])
            lines.append(f"Table: {table}")
            lines.append(f"  Columns: {', '.join(cols)}")
            if fks:
                lines.append(f"  Foreign Keys: {', '.join(fks)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Keyword → table mapping
    # ------------------------------------------------------------------

    def _build_keyword_map(self) -> Dict[str, str]:
        """Map lowercase keywords to table names for quick lookup."""
        kw_map = {}

        # Table name variants → table
        for table in self._table_names:
            kw_map[table.lower()] = table
            # Singular/plural handling
            if table.endswith("s"):
                kw_map[table[:-1].lower()] = table  # "user" → users
            else:
                kw_map[(table + "s").lower()] = table  # "product" → products (if table is singular)

        # Column name → table (for disambiguation)
        for table, cols in self._columns.items():
            for col in cols:
                col_lower = col.lower()
                # Avoid overwriting more specific mappings
                if col_lower not in kw_map:
                    kw_map[col_lower] = table

        # Special domain-specific aliases
        kw_map.update({
            "customer": "users",
            "customers": "users",
            "buyer": "users",
            "buyers": "users",
            "people": "users",
            "person": "users",
            "purchase": "orders",
            "purchases": "orders",
            "bought": "orders",
            "ordered": "orders",
            "spending": "orders",
            "spent": "orders",
            "sales": "orders",
            "revenue": "orders",
            "item": "order_items",
            "goods": "products",
            "electronics": "products",
            "books": "products",
            "clothing": "products",
            "sports": "products",
            "inventory": "products",
        })

        return kw_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_plan(self, query: str, retrieved_schema: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze the query and produce a structured plan.
        
        Args:
            query: User's natural language query
            retrieved_schema: Optional filtered schema from vector retrieval.
                            If provided, planning will be constrained to these tables only.

        Returns:
            {
                "tables": [...],
                "joins": [{\"from\": ..., \"to\": ..., \"on\": ...}],
                "select_columns": [...],
                "where_conditions": [...],
                "group_by": [...],
                "order_by": [...],
                "limit": int | None,
                "aggregation": str | None,
                "schema_context": str,  # Full schema for generator
            }
        """
        query_lower = query.lower()
        
        # Use retrieved schema if provided, otherwise use full schema
        active_schema = retrieved_schema if retrieved_schema else self.schema
        
        # Temporarily override schema for this planning session
        original_schema = self.schema
        original_table_names = self._table_names
        
        if retrieved_schema:
            self.schema = retrieved_schema
            self._table_names = list(retrieved_schema.keys())
            logger.info(f"Using retrieved schema with {len(self._table_names)} tables: {self._table_names}")

        # 1. Identify required tables
        tables = self._identify_tables(query_lower)
        logger.info(f"Identified tables: {tables}")

        # 2. Resolve JOINs
        joins = self._resolve_joins(tables)

        # 3. Detect aggregation
        aggregation = self._detect_aggregation(query_lower)

        # 4. Detect filters
        where_conditions = self._detect_filters(query_lower, tables)

        # 5. Detect ordering
        order_by = self._detect_ordering(query_lower)

        # 6. Detect LIMIT
        limit = self._detect_limit(query_lower)

        # 7. Detect GROUP BY needs
        group_by = self._detect_group_by(query_lower, tables, aggregation)

        # 8. Build select columns
        select_columns = self._determine_select(query_lower, tables, aggregation)
        
        # Build schema context for generator using active schema
        schema_desc = self._build_schema_description_from(active_schema)

        plan = {
            "tables": tables,
            "joins": joins,
            "select_columns": select_columns,
            "where_conditions": where_conditions,
            "group_by": group_by,
            "order_by": order_by,
            "limit": limit,
            "aggregation": aggregation,
            "schema_context": schema_desc,
        }
        
        # Restore original schema
        if retrieved_schema:
            self.schema = original_schema
            self._table_names = original_table_names

        logger.info(f"Deterministic plan: tables={tables}, agg={aggregation}, "
                     f"filters={len(where_conditions)}, order={order_by}")
        return plan

    # ------------------------------------------------------------------
    # Table identification
    # ------------------------------------------------------------------

    def _identify_tables(self, query: str) -> List[str]:
        """Identify required tables from query keywords."""
        tables: Set[str] = set()
        words = re.findall(r'\b\w+\b', query)

        for word in words:
            if word in self._keyword_table_map:
                tables.add(self._keyword_table_map[word])

        # If user mentions "bought", "purchased", etc. we need order_items + products
        purchase_words = {"bought", "purchased", "purchase", "purchases", "ordered"}
        if purchase_words & set(words):
            tables.update({"orders", "order_items", "products"})

        # If user mentions a specific user name, we need users + orders
        name_patterns = [r'\bwhat did (\w+)', r'\bshow (\w+)\'?s?\b.*order',
                         r'\bhow many.*\border.*\b(\w+)']
        for pattern in name_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                tables.update({"users", "orders"})
                break

        # If "sales by category" or "revenue by category" → need products + orders + order_items
        if ("category" in query or "categories" in query) and \
           any(w in query for w in ["sales", "revenue", "total", "sum"]):
            tables.update({"products", "orders", "order_items"})

        # If nothing matched, default to all tables
        if not tables:
            tables = set(self._table_names)

        # Ensure required bridge tables are included
        tables = self._ensure_bridge_tables(tables)

        return sorted(tables)

    def _ensure_bridge_tables(self, tables: Set[str]) -> Set[str]:
        """Add bridge tables needed to connect the identified tables."""
        # If we have products and orders but not order_items, we need order_items
        if "products" in tables and "orders" in tables and "order_items" not in tables:
            tables.add("order_items")
        # If we have users and products, we need orders + order_items
        if "users" in tables and "products" in tables:
            tables.add("orders")
            tables.add("order_items")
        return tables

    # ------------------------------------------------------------------
    # Aggregation detection
    # ------------------------------------------------------------------

    def _detect_aggregation(self, query: str) -> Optional[str]:
        """Detect aggregation type from query keywords."""
        # Special case: 'which X has the most Y' → COUNT, not MAX
        if re.search(r'\bwhich\b.*\bmost\b', query) and not re.search(r'\bmost expensive\b', query):
            return "COUNT"

        for agg_type, patterns in _AGG_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return agg_type
        return None

    # ------------------------------------------------------------------
    # Filter detection
    # ------------------------------------------------------------------

    def _detect_filters(self, query: str, tables: List[str]) -> List[Dict[str, str]]:
        """Detect WHERE conditions from query keywords."""
        filters = []

        # Price filters
        for pattern, op, hint in _FILTER_PATTERNS:
            match = re.search(pattern, query)
            if not match:
                continue

            if op == "BETWEEN":
                filters.append({
                    "column": "products.price",
                    "op": "BETWEEN",
                    "value": f"{match.group(1)} AND {match.group(2)}"
                })
            elif op == "DATE":
                days = int(match.group(1))
                if hint == "weeks":
                    days *= 7
                elif hint == "months":
                    days *= 30
                filters.append({
                    "column": "orders.order_date",
                    "op": ">=",
                    "value": f"date('now', '-{days} days')"
                })
            elif op in ("<", ">"):
                # Determine which column (price context vs quantity)
                if any(w in query for w in ["stock", "inventory", "quantity"]):
                    col = "products.stock_quantity"
                else:
                    col = "products.price"
                filters.append({"column": col, "op": op, "value": match.group(1)})

        # Category filter
        categories = ["electronics", "home & kitchen", "books", "sports & outdoors", "clothing"]
        for cat in categories:
            if cat.lower() in query:
                filters.append({
                    "column": "products.category",
                    "op": "=",
                    "value": f"'{cat.title()}'" if '&' not in cat else f"'{cat.replace('&', '&').title()}'"
                })

        # Email filter
        email_match = re.search(r"email.*(?:contains|like|with)\s+['\"]?([^'\"]+)['\"]?", query)
        if email_match:
            filters.append({
                "column": "users.email",
                "op": "LIKE",
                "value": f"'%{email_match.group(1)}%'"
            })
        elif "example.com" in query:
            filters.append({
                "column": "users.email",
                "op": "LIKE",
                "value": "'%@example.com'"
            })

        # Name filter (e.g., "What did Alice purchase?")
        name_match = re.search(r'\b(?:what did|show|for)\s+(\w+)\b', query)
        if name_match:
            name = name_match.group(1)
            # Exclude common words, table names, SQL terms, prepositions
            skip_words = {
                "the", "all", "each", "every", "me", "total", "my", "our",
                "users", "user", "products", "product", "orders", "order",
                "items", "item", "sales", "revenue", "inventory",
                "category", "categories", "price", "stock", "quantity",
                "that", "this", "those", "which", "what", "how",
            }
            if name.lower() not in skip_words and name[0].isupper():
                filters.append({
                    "column": "users.name",
                    "op": "LIKE",
                    "value": f"'%{name}%'"
                })

        # Order ID filter (e.g., "order 1", "order #5", "order ID 10")
        order_id_match = re.search(r'\border\s*(?:#|id)?\s*(\d+)\b', query, re.IGNORECASE)
        if order_id_match:
            filters.append({
                "column": "orders.order_id",
                "op": "=",
                "value": order_id_match.group(1)
            })

        return filters

    # ------------------------------------------------------------------
    # Ordering detection
    # ------------------------------------------------------------------

    def _detect_ordering(self, query: str) -> List[Dict[str, str]]:
        """Detect ORDER BY from query keywords."""
        for pattern, column, direction in _ORDER_PATTERNS:
            if re.search(pattern, query):
                return [{"column": column, "direction": direction}]
        return []

    # ------------------------------------------------------------------
    # Limit detection
    # ------------------------------------------------------------------

    def _detect_limit(self, query: str) -> Optional[int]:
        """Detect LIMIT from query keywords."""
        limit_match = re.search(r'\btop\s+(\d+)\b', query)
        if limit_match:
            return int(limit_match.group(1))

        # "first N" pattern
        first_match = re.search(r'\bfirst\s+(\d+)\b', query)
        if first_match:
            return int(first_match.group(1))

        # "which X has the most Y" → LIMIT 1
        if re.search(r'\bwhich\b.*\bmost\b', query) or re.search(r'\bmost\b.*\bwhich\b', query):
            return 1

        return None

    # ------------------------------------------------------------------
    # GROUP BY detection
    # ------------------------------------------------------------------

    def _detect_group_by(self, query: str, tables: List[str],
                          aggregation: Optional[str]) -> List[str]:
        """Detect GROUP BY columns based on query context."""
        if not aggregation:
            return []

        # "by category" → GROUP BY products.category
        if "by category" in query or "per category" in query or "each category" in query:
            return ["products.category"]

        # "which category has the most X" → GROUP BY products.category
        if re.search(r'which\s+category', query) and "most" in query:
            return ["products.category"]

        # "by user" / "per user" / "each user"
        if "by user" in query or "per user" in query or "each user" in query:
            return ["users.name"]

        # "by product" / "per product"
        if "by product" in query or "per product" in query or "each product" in query:
            return ["products.name"]

        # If aggregation + "each" → try to find what to group by
        if "each" in query:
            for table in tables:
                if table.lower() in query:
                    # Group by the name column of that table
                    if "name" in self._columns.get(table, []):
                        return [f"{table}.name"]

        return []

    # ------------------------------------------------------------------
    # SELECT column determination
    # ------------------------------------------------------------------

    def _determine_select(self, query: str, tables: List[str],
                           aggregation: Optional[str]) -> List[str]:
        """
        Determine which columns to SELECT based on query intent.
        
        Returns specific columns when identifiable, avoiding SELECT * when possible.
        """
        columns = []
        
        # If aggregation, determine what to aggregate
        if aggregation:
            # Check if grouping by something
            if "by category" in query or "per category" in query or "each category" in query:
                columns.append("products.category")
            elif "by user" in query or "per user" in query or "each user" in query:
                columns.append("users.name")
            elif "by product" in query or "per product" in query:
                columns.append("products.name")
            
            # Add the aggregation
            if aggregation == "COUNT":
                if "which" in query and "most" in query:
                    # "which category has the most products" → need category + count
                    if "products.category" not in columns:
                        columns.append("products.category")
                    columns.append("COUNT(*)")
                else:
                    columns.append("COUNT(*)")
            elif aggregation == "SUM":
                # Sum of what? Look for price, amount, quantity
                if "price" in query or "revenue" in query or "sales" in query:
                    if "order_items" in tables:
                        columns.append("SUM(order_items.quantity * products.price)")
                    elif "products" in tables:
                        columns.append("SUM(products.price)")
                elif "quantity" in query:
                    columns.append("SUM(order_items.quantity)")
                else:
                    columns.append("SUM(*)")
            elif aggregation == "AVG":
                if "price" in query:
                    columns.append("AVG(products.price)")
                else:
                    columns.append("AVG(*)")
            elif aggregation == "MAX":
                if "price" in query:
                    columns.append("MAX(products.price)")
                else:
                    columns.append("MAX(*)")
            elif aggregation == "MIN":
                if "price" in query:
                    columns.append("MIN(products.price)")
                else:
                    columns.append("MIN(*)")
            
            return columns if columns else ["*"]
        
        # No aggregation - determine columns from query keywords
        
        # Email queries
        if "email" in query:
            if "users" in tables:
                columns.append("users.email")
        
        # Name queries
        if re.search(r'\bnames?\b', query):
            if "users" in tables:
                columns.append("users.name")
            if "products" in tables and "product" in query:
                columns.append("products.name")
        
        # Price queries
        if "price" in query or "cost" in query:
            if "products" in tables:
                columns.append("products.price")
        
        # Category queries
        if "categor" in query:
            if "products" in tables:
                columns.append("products.category")
        
        # Stock/inventory queries
        if "stock" in query or "inventory" in query or "quantity" in query:
            if "products" in tables:
                columns.append("products.stock_quantity")
        
        # Order date queries
        if "date" in query or "when" in query:
            if "orders" in tables:
                columns.append("orders.order_date")
        
        # Product details queries (show/list products)
        if re.search(r'\b(show|list|display|get)\b.*\bproducts?\b', query):
            if "products" in tables:
                # Include key product columns
                columns.extend(["products.name", "products.price", "products.category"])
        
        # User details queries
        if re.search(r'\b(show|list|display|get)\b.*\b(users?|customers?)\b', query):
            if "users" in tables:
                columns.extend(["users.name", "users.email"])
        
        # Order details queries
        if re.search(r'\b(show|list|display|get)\b.*\borders?\b', query):
            if "orders" in tables:
                columns.extend(["orders.order_id", "orders.order_date"])
        
        # "What did X purchase/buy/order" queries
        if re.search(r'\bwhat did\b.*\b(purchase|buy|order)', query):
            if "products" in tables:
                columns.append("products.name")
            if "orders" in tables:
                columns.append("orders.order_date")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_columns = []
        for col in columns:
            if col not in seen:
                seen.add(col)
                unique_columns.append(col)
        
        # If we identified specific columns, return them
        if unique_columns:
            return unique_columns
        
        # Fallback: if single table and simple query, select key columns
        if len(tables) == 1:
            table = tables[0]
            if table == "users":
                return ["users.name", "users.email"]
            elif table == "products":
                return ["products.name", "products.price", "products.category"]
            elif table == "orders":
                return ["orders.order_id", "orders.user_id", "orders.order_date"]
        
        # Last resort: SELECT *
        return ["*"]

    # ------------------------------------------------------------------
    # Deterministic JOIN resolution
    # ------------------------------------------------------------------

    def _resolve_joins(self, tables: List[str]) -> List[Dict]:
        """Build JOIN clauses by following foreign keys between identified tables."""
        if len(tables) <= 1:
            return []

        joins = []
        connected = {tables[0]}
        remaining = set(tables[1:])

        # BFS: connect tables through foreign keys
        changed = True
        while remaining and changed:
            changed = False
            for table in list(remaining):
                # Check if table has FK pointing to a connected table
                for fk in self._fk_graph.get(table, []):
                    if fk["to_table"] in connected:
                        joins.append({
                            "from": table,
                            "to": fk["to_table"],
                            "on": f"{table}.{fk['from_col']} = {fk['to_table']}.{fk['to_col']}",
                        })
                        connected.add(table)
                        remaining.discard(table)
                        changed = True
                        break

                # Check reverse: connected table has FK pointing to this table
                if table in remaining:
                    for conn_tbl in list(connected):
                        for fk in self._fk_graph.get(conn_tbl, []):
                            if fk["to_table"] == table:
                                joins.append({
                                    "from": conn_tbl,
                                    "to": table,
                                    "on": f"{conn_tbl}.{fk['from_col']} = {table}.{fk['to_col']}",
                                })
                                connected.add(table)
                                remaining.discard(table)
                                changed = True
                                break
                        if table not in remaining:
                            break

        # Bridge tables for unconnected
        if remaining:
            for table in list(remaining):
                bridge = self._find_bridge(table, connected)
                if bridge:
                    joins.extend(bridge)
                    connected.add(table)
                    remaining.discard(table)

        return joins

    def _find_bridge(self, target: str, connected: set) -> List[Dict]:
        """Find a bridge table to connect target to already-connected tables."""
        for bridge_table in self._table_names:
            if bridge_table in connected or bridge_table == target:
                continue
            bridge_to_connected = None
            bridge_to_target = None

            for fk in self._fk_graph.get(bridge_table, []):
                if fk["to_table"] in connected:
                    bridge_to_connected = fk
                if fk["to_table"] == target:
                    bridge_to_target = fk

            for fk in self._fk_graph.get(target, []):
                if fk["to_table"] == bridge_table:
                    bridge_to_target = {
                        "from_col": fk["from_col"],
                        "to_table": bridge_table,
                        "to_col": fk["to_col"],
                    }

            if bridge_to_connected and bridge_to_target:
                return [
                    {
                        "from": bridge_table,
                        "to": bridge_to_connected["to_table"],
                        "on": f"{bridge_table}.{bridge_to_connected['from_col']} = {bridge_to_connected['to_table']}.{bridge_to_connected['to_col']}",
                    },
                    {
                        "from": target,
                        "to": bridge_table,
                        "on": f"{target}.{bridge_to_target['from_col']} = {bridge_table}.{bridge_to_target['to_col']}",
                    },
                ]
        return []
