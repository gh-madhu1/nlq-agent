"""
Tests for the refactored NL2SQL Agent.

Tests cover:
1. Validator — syntax and safety checks
2. IntentSafetyClassifier — keyword-based classification
3. QueryPlanner — deterministic planning
4. SQLEvaluator — schema-aware SQL validation
5. Agent integration — full pipeline with mocked LLM and DB
"""

import unittest
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────
# Validator Tests
# ─────────────────────────────────────────────────────────────────────
class TestValidator(unittest.TestCase):
    """Test cases for the SQL validator."""

    def setUp(self):
        from handlers.validator import Validator
        self.validator = Validator()

    def test_valid_select(self):
        is_valid, err = self.validator.validate_sql("SELECT * FROM users;")
        self.assertTrue(is_valid)
        self.assertEqual(err, "")

    def test_rejects_drop(self):
        is_valid, err = self.validator.validate_sql("DROP TABLE users;")
        self.assertFalse(is_valid)
        self.assertIn("SELECT", err)

    def test_rejects_delete(self):
        is_valid, err = self.validator.validate_sql("SELECT * FROM users; DELETE FROM users;")
        self.assertFalse(is_valid)
        self.assertIn("Multiple", err)

    def test_no_false_positive_on_column_names(self):
        """'updated_at' should NOT trigger the 'UPDATE' keyword check."""
        is_valid, err = self.validator.validate_sql("SELECT updated_at FROM orders;")
        self.assertTrue(is_valid)

    def test_rejects_empty(self):
        is_valid, err = self.validator.validate_sql("")
        self.assertFalse(is_valid)

    def test_rejects_insert(self):
        is_valid, err = self.validator.validate_sql("INSERT INTO users VALUES (1, 'test', 'test@test.com');")
        self.assertFalse(is_valid)

    def test_validates_known_tables(self):
        is_valid, err = self.validator.validate_sql(
            "SELECT * FROM nonexistent_table;",
            known_tables=["users", "products"]
        )
        self.assertFalse(is_valid)
        self.assertIn("nonexistent_table", err)


# ─────────────────────────────────────────────────────────────────────
# Intent Classifier Tests
# ─────────────────────────────────────────────────────────────────────
class TestIntentClassifier(unittest.TestCase):
    """Test keyword-based intent and safety classification."""

    def setUp(self):
        from handlers.intent_safety import IntentSafetyClassifier
        self.classifier = IntentSafetyClassifier()

    def test_data_query_detected(self):
        result = self.classifier.classify("How many products are there?")
        self.assertTrue(result["is_data_query"])
        self.assertTrue(result["is_safe"])

    def test_non_data_query_detected(self):
        result = self.classifier.classify("Hello, how are you doing today?")
        self.assertFalse(result["is_data_query"])

    def test_unsafe_query_detected(self):
        result = self.classifier.classify("DROP TABLE users")
        self.assertFalse(result["is_safe"])

    def test_sql_injection_detected(self):
        result = self.classifier.classify("Show users; DELETE FROM users")
        self.assertFalse(result["is_safe"])

    def test_data_keywords_recognized(self):
        queries = [
            "Show all products",
            "List users",
            "What is the total revenue?",
            "Find orders by Alice",
            "How many items were sold?",
        ]
        for q in queries:
            result = self.classifier.classify(q)
            self.assertTrue(result["is_data_query"], f"'{q}' should be classified as data query")


# ─────────────────────────────────────────────────────────────────────
# Planner Tests
# ─────────────────────────────────────────────────────────────────────
class TestQueryPlanner(unittest.TestCase):
    """Test deterministic query planner."""

    def setUp(self):
        from handlers.planner import QueryPlanner
        self.schema = {
            "users": {
                "columns": ["user_id INTEGER", "name TEXT", "email TEXT"],
                "foreign_keys": []
            },
            "products": {
                "columns": ["product_id INTEGER", "name TEXT", "category TEXT", "price REAL", "stock_quantity INTEGER"],
                "foreign_keys": []
            },
            "orders": {
                "columns": ["order_id INTEGER", "user_id INTEGER", "order_date TEXT", "total_amount REAL"],
                "foreign_keys": ["user_id -> users.user_id"]
            },
            "order_items": {
                "columns": ["item_id INTEGER", "order_id INTEGER", "product_id INTEGER", "quantity INTEGER", "price_at_purchase REAL"],
                "foreign_keys": ["order_id -> orders.order_id", "product_id -> products.product_id"]
            }
        }
        self.planner = QueryPlanner(self.schema)

    def test_identifies_products_table(self):
        plan = self.planner.create_plan("Show all products")
        self.assertIn("products", plan["tables"])

    def test_identifies_users_table(self):
        plan = self.planner.create_plan("List all users")
        self.assertIn("users", plan["tables"])

    def test_detects_count_aggregation(self):
        plan = self.planner.create_plan("How many products are there?")
        self.assertEqual(plan["aggregation"], "COUNT")

    def test_detects_sum_aggregation(self):
        plan = self.planner.create_plan("What is the total revenue?")
        self.assertEqual(plan["aggregation"], "SUM")

    def test_detects_avg_aggregation(self):
        plan = self.planner.create_plan("What is the average price?")
        self.assertEqual(plan["aggregation"], "AVG")

    def test_resolves_joins(self):
        plan = self.planner.create_plan("What did Alice purchase?")
        self.assertTrue(len(plan["joins"]) > 0)

    def test_detects_group_by_category(self):
        plan = self.planner.create_plan("Show total sales by category")
        self.assertIn("products.category", plan["group_by"])

    def test_includes_bridge_tables(self):
        plan = self.planner.create_plan("Show products ordered by each user")
        self.assertIn("order_items", plan["tables"])

    def test_returns_schema_context(self):
        plan = self.planner.create_plan("Show products")
        self.assertIn("schema_context", plan)
        self.assertTrue(len(plan["schema_context"]) > 0)

    def test_most_products_adds_limit_and_group_by(self):
        """'Which category has the most products' should get GROUP BY + LIMIT 1."""
        plan = self.planner.create_plan("Which category has the most products?")
        self.assertIn("products.category", plan["group_by"])
        self.assertEqual(plan["limit"], 1)

    def test_inventory_value_uses_only_products(self):
        """Inventory value query should only use products table, no extra joins."""
        plan = self.planner.create_plan("What's the total value of all inventory (price × stock)?")
        self.assertEqual(plan["tables"], ["products"])
        self.assertEqual(plan["joins"], [])

    def test_no_name_filter_on_common_words(self):
        """'by users whose email ...' should NOT trigger a name filter for 'users'."""
        plan = self.planner.create_plan("List all products purchased by users whose email contains 'example.com'")
        name_filters = [f for f in plan["where_conditions"] if f["column"] == "users.name"]
        self.assertEqual(len(name_filters), 0, "Should not have a name filter for 'users'")


# ─────────────────────────────────────────────────────────────────────
# Evaluator Tests
# ─────────────────────────────────────────────────────────────────────
class TestSQLEvaluator(unittest.TestCase):
    """Test deterministic SQL evaluator."""

    def setUp(self):
        from handlers.evaluator import SQLEvaluator
        self.schema = {
            "users": {
                "columns": ["user_id INTEGER", "name TEXT", "email TEXT"],
                "foreign_keys": []
            },
            "products": {
                "columns": ["product_id INTEGER", "name TEXT", "category TEXT", "price REAL"],
                "foreign_keys": []
            },
            "orders": {
                "columns": ["order_id INTEGER", "user_id INTEGER", "order_date TEXT", "total_amount REAL"],
                "foreign_keys": ["user_id -> users.user_id"]
            },
            "order_items": {
                "columns": ["item_id INTEGER", "order_id INTEGER", "product_id INTEGER", "quantity INTEGER", "price_at_purchase REAL"],
                "foreign_keys": ["order_id -> orders.order_id", "product_id -> products.product_id"]
            }
        }
        self.evaluator = SQLEvaluator(self.schema)

    def test_valid_simple_query(self):
        is_valid, err = self.evaluator.evaluate("SELECT users.name FROM users")
        self.assertTrue(is_valid, f"Expected valid but got: {err}")

    def test_valid_join_query(self):
        sql = "SELECT users.name, orders.total_amount FROM users JOIN orders ON orders.user_id = users.user_id"
        is_valid, err = self.evaluator.evaluate(sql)
        self.assertTrue(is_valid, f"Expected valid but got: {err}")

    def test_rejects_nonexistent_table(self):
        is_valid, err = self.evaluator.evaluate("SELECT * FROM fake_table")
        self.assertFalse(is_valid)
        self.assertIn("fake_table", err)

    def test_rejects_nonexistent_column(self):
        is_valid, err = self.evaluator.evaluate("SELECT users.fake_column FROM users")
        self.assertFalse(is_valid)
        self.assertIn("fake_column", err)

    def test_rejects_column_in_wrong_table(self):
        # quantity is in order_items, not orders
        is_valid, err = self.evaluator.evaluate("SELECT orders.quantity FROM orders")
        self.assertFalse(is_valid)
        self.assertIn("quantity", err)

    def test_valid_with_alias(self):
        sql = "SELECT u.name FROM users u"
        is_valid, err = self.evaluator.evaluate(sql)
        self.assertTrue(is_valid, f"Expected valid but got: {err}")

    def test_provides_correction_hint(self):
        _, err = self.evaluator.evaluate("SELECT orders.quantity FROM orders")
        hint = self.evaluator.get_correction_hint("SELECT orders.quantity FROM orders", err)
        self.assertIn("Correct schema reference", hint)
        self.assertIn("orders", hint)


# ─────────────────────────────────────────────────────────────────────
# Agent Integration Tests (mocked LLM + mocked DB)
# ─────────────────────────────────────────────────────────────────────
class TestNL2SQLAgent(unittest.TestCase):
    """Integration tests for the full agent pipeline with mocks."""

    def _make_agent(self, llm_responses, db_schema, db_results):
        """Helper to create an agent with mocked dependencies."""
        mock_llm = MagicMock()
        mock_llm.call.side_effect = llm_responses

        with patch('agent.DBManager') as MockDBManager:
            mock_db = MockDBManager.return_value
            mock_db.get_full_schema.return_value = db_schema
            mock_db.execute_query.return_value = db_results

            from agent import NL2SQLAgent
            agent = NL2SQLAgent(llm_client=mock_llm, db_path=":memory:")
            return agent

    def test_unsafe_query_rejected(self):
        """Unsafe queries should be blocked without any SQL generation."""
        schema = {
            "users": {"columns": ["user_id INTEGER", "name TEXT"], "foreign_keys": []}
        }
        agent = self._make_agent([], schema, [])
        response = agent.run("DROP TABLE users; --")
        self.assertIn("Unsafe", response)

    def test_non_data_query_handled(self):
        """Non-data queries should be handled as general chat."""
        schema = {
            "users": {"columns": ["user_id INTEGER", "name TEXT"], "foreign_keys": []}
        }
        agent = self._make_agent(
            None,  # Not using side_effect
            schema, []
        )
        # Use return_value so any call to LLM returns the same thing
        agent.llm_client.call.return_value = "I'm a helpful database assistant."
        response = agent.run("Hello, how are you?")
        self.assertIn("helpful", response)


if __name__ == "__main__":
    unittest.main()
