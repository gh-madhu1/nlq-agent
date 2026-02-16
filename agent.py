"""
NL2SQL Agent — Natural Language to SQL with deterministic planning.

Flow: Classify → Retrieve Schema → Plan → Generate → Validate → Execute
Max 3 retries for SQL generation with error feedback.
"""

import re
import logging
from typing import Generator, Dict, Any

from handlers.intent_safety import IntentSafetyClassifier as IntentClassifier
from handlers.query_refiner import QueryRefiner
from handlers.retriever import SchemaRetriever
from handlers.planner import QueryPlanner
from handlers.generator import SQLGenerator
from handlers.validator import Validator
from handlers.executor import Executor as SQLExecutor
from utils.llm_client import LLMClient
from utils.db_manager import DBManager
from utils.vector_store import SchemaVectorStore

logger = logging.getLogger(__name__)


class NL2SQLAgent:
    """
    Natural Language to SQL Agent with deterministic planning and self-evaluation.

    Flow:
    1. Refine (conditional) — clean up query if needed (LLM, only for ambiguous queries)
    2. Classify — intent & safety check (keyword-based, no LLM)
    3. Retrieve Schema — vector search for relevant tables (no LLM)
    4. Plan — identify tables, joins, filters, aggregation (deterministic, no LLM)
    5. Generate — produce SQL (LLM with strict plan adherence)
    6. Validate — syntax, safety, and schema checks (deterministic, no LLM)
    7. Execute — run SQL against database

    Max 3 retries with detailed error feedback on each failure.
    """

    MAX_RETRIES = 3

    def __init__(
        self,
        llm_client: LLMClient,
        db_path: str,
        max_retries: int = 3
    ):
        self.llm_client = llm_client
        self.db_path = db_path
        self.max_retries = max_retries

        # Initialize core dependencies
        self.db_manager = DBManager(db_path)

        # Get schema (used by multiple handlers)
        self.schema_info = self.db_manager.get_full_schema()
        self.table_names = list(self.schema_info.keys())
        
        # Initialize vector store for schema retrieval
        self.vector_store = SchemaVectorStore()
        self.vector_store.initialize_from_schema(self.schema_info)

        # Initialize handlers (order matches the pipeline)
        self.refiner = QueryRefiner(llm_client, self.schema_info)  # Conditional use
        self.classifier = IntentClassifier(llm_client)
        self.retriever = SchemaRetriever(self.vector_store, self.schema_info)
        self.planner = QueryPlanner(self.schema_info)
        self.generator = SQLGenerator(llm_client, self.schema_info)
        self.validator = Validator(self.schema_info)  # Now includes schema validation
        self.executor = SQLExecutor(self.db_manager)

        logger.info(f"NL2SQLAgent initialized: tables={self.table_names}, max_retries={max_retries}")

    def process_query(self, user_query: str) -> Generator[Dict[str, Any], None, None]:
        """
        Process a natural language query and yield SSE events.

        Each event has: {"event": str, "data": Any, "log": str}
        """
        yield {
            "event": "start",
            "data": user_query,
            "log": f"Processing query: {user_query}"
        }
        logger.info(f"Processing query: {user_query}")
        
        # ── Step 1: Conditional Query Refinement ──────────────────────
        # Only refine if query needs it (saves LLM calls)
        refined_query = user_query
        if self._needs_refinement(user_query):
            yield {"event": "refining", "data": {}, "log": "Refining query..."}
            try:
                refined_query = self.refiner.refine(user_query)
                if refined_query != user_query:
                    yield {
                        "event": "query_refined",
                        "data": {"original": user_query, "refined": refined_query},
                        "log": f"Refined: '{user_query}' → '{refined_query}'"
                    }
                else:
                    yield {"event": "query_refined", "data": {"refined": refined_query}, "log": "Query unchanged after refinement"}
            except Exception as e:
                logger.warning(f"Refinement failed: {e}, using original query")
                refined_query = user_query
        else:
            logger.info("Query is clean, skipping refinement")

        # ── Step 2: Intent & Safety Classification ────────────────────
        classification = self.classifier.classify(refined_query)
        yield {
            "event": "intent_classification",
            "data": classification,
            "log": f"Intent: data={classification['is_data_query']}, safe={classification['is_safe']}"
        }

        if not classification["is_safe"]:
            yield {"event": "error", "data": "Unsafe query detected.", "log": "Unsafe query"}
            return

        if not classification["is_data_query"]:
            response = self.llm_client.call(refined_query)
            yield {"event": "final_answer", "data": response, "log": "Handled as general chat"}
            return

        # ── Step 2: Schema Retrieval (Vector Search) ──────────────────
        yield {"event": "schema_retrieval", "data": {}, "log": "Retrieving relevant schema..."}
        retrieved_schema = self.retriever.get_relevant_schema_dict(refined_query, k=5)
        retrieved_tables = list(retrieved_schema.keys())
        yield {
            "event": "schema_retrieved",
            "data": {"tables": retrieved_tables, "count": len(retrieved_tables)},
            "log": f"Retrieved {len(retrieved_tables)} relevant tables: {retrieved_tables}"
        }

        # ── Step 3: Deterministic Planning ────────────────────────────
        yield {"event": "planning", "data": {}, "log": "Planning query..."}
        plan = self.planner.create_plan(refined_query, retrieved_schema=retrieved_schema)
        yield {
            "event": "query_plan",
            "data": {k: v for k, v in plan.items() if k != "schema_context"},
            "log": f"Plan: tables={plan['tables']}, agg={plan.get('aggregation')}"
        }

        # Use the schema context from the plan (based on retrieved schema)
        schema_context = plan.get("schema_context", "")

        # ── Step 4-6: Generate → Validate → Execute loop ──────────────
        error_feedback = ""
        failed_sqls = []

        for attempt in range(1, self.max_retries + 1):
            attempt_label = f"Attempt {attempt}/{self.max_retries}"

            try:
                # 4. Generate SQL
                yield {
                    "event": "sql_generation_start",
                    "data": {"attempt": attempt},
                    "log": f"Generating SQL ({attempt_label})..."
                }
                sql = self.generator.generate_sql(
                    refined_query, schema_context, plan, error_feedback
                )
                yield {"event": "sql_generated", "data": sql, "log": f"Generated: {sql}"}
                logger.info(f"{attempt_label}: Generated SQL: {sql}")

                # Check for repeated failures
                if sql in failed_sqls:
                    error_feedback = (
                        f"You generated the same SQL that already failed: {sql}\n"
                        f"Generate a COMPLETELY DIFFERENT query."
                    )
                    yield {
                        "event": "evaluation_failed",
                        "data": "Duplicate SQL detected",
                        "log": "Same SQL as previous failed attempt"
                    }
                    continue

                # 5. Syntax, safety, and schema validation (all in one)
                is_valid, validation_error = self.validator.validate_sql(
                    sql, known_tables=self.table_names
                )
                if not is_valid:
                    failed_sqls.append(sql)
                    # Get detailed correction hint
                    correction_hint = self.validator.get_correction_hint(sql, validation_error)
                    error_feedback = correction_hint
                    yield {
                        "event": "validation_failed",
                        "data": validation_error,
                        "log": f"Validation failed: {validation_error}"
                    }
                    logger.warning(f"{attempt_label}: Validation failed: {validation_error}")
                    continue

                yield {
                    "event": "validation_passed",
                    "data": "SQL passed all validation checks",
                    "log": "Validation passed"
                }

                # 6. Execute SQL
                results, exec_error = self.executor.execute_sql(sql)
                if exec_error:
                    failed_sqls.append(sql)
                    # Build specific error feedback for the generator
                    if "no such table" in exec_error.lower():
                        error_feedback = (
                            f"Execution error: {exec_error}\n"
                            f"Available tables: {', '.join(self.table_names)}"
                        )
                    elif "no such column" in exec_error.lower():
                        error_feedback = self.validator.get_correction_hint(sql, exec_error)
                    else:
                        error_feedback = f"Execution error: {exec_error}\nSimplify the query."
                    yield {
                        "event": "execution_failed",
                        "data": exec_error,
                        "log": f"Execution failed: {exec_error}"
                    }
                    logger.warning(f"{attempt_label}: Execution failed: {exec_error}")
                    continue

                yield {
                    "event": "execution_success",
                    "data": results,
                    "log": f"Executed successfully. Rows: {len(results)}"
                }

                # ── Return results directly ────────────────────────
                answer = {
                    "sql": sql,
                    "rows": results,
                    "row_count": len(results),
                }
                yield {"event": "final_answer", "data": answer, "log": "Answer ready"}
                logger.info(f"Query completed successfully: {user_query}")
                return

            except Exception as e:
                logger.error(f"Error in {attempt_label}: {str(e)}", exc_info=True)
                error_feedback = f"Internal error: {str(e)}\nGenerate a simpler query."
                continue

        # All attempts exhausted
        error_msg = {
            "message": f"Unable to generate valid SQL after {self.max_retries} attempts.",
            "suggestion": "Try rephrasing your question more simply.",
            "examples": [
                "Try: 'Show all products'",
                "Try: 'How many orders are there?'",
                "Try: 'What did Alice purchase?'",
                "Try: 'Show products under $50'",
            ]
        }
        yield {
            "event": "error",
            "data": error_msg,
            "log": f"Max retries ({self.max_retries}) reached"
        }

    def _needs_refinement(self, query: str) -> bool:
        """
        Heuristic to determine if query needs LLM refinement.
        
        Returns True if query appears ambiguous, has typos, or is poorly formed.
        """
        words = query.split()
        
        # Very short queries (< 3 words) might need clarification
        if len(words) < 3:
            return True
        
        # Check for obvious typos or non-words
        # Simple heuristic: if >30% of words are not common or schema terms
        uncommon_count = 0
        common_words = {
            "show", "list", "get", "find", "what", "how", "many", "all", "the",
            "from", "in", "by", "for", "with", "and", "or", "to", "of", "is",
            "are", "was", "were", "has", "have", "had", "total", "count", "sum",
            "average", "max", "min", "most", "least", "under", "over", "between",
            "users", "products", "orders", "items", "email", "name", "price",
            "category", "stock", "quantity", "date", "amount", "purchased", "bought",
        }
        
        for word in words:
            clean_word = word.lower().strip("?!.,;:")
            if len(clean_word) > 2 and clean_word not in common_words:
                uncommon_count += 1
        
        if uncommon_count > len(words) * 0.3:
            return True
        
        # Check for heavy abbreviations/slang
        slang_patterns = [r'\bppl\b', r'\bcat\b', r'\bqty\b', r'\bamt\b', r'\binv\b']
        for pattern in slang_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        return False

    def run(self, user_query: str) -> str:
        """Backward compatibility wrapper for non-streaming usage."""
        final_answer = None
        for event in self.process_query(user_query):
            if event["event"] == "final_answer":
                final_answer = event["data"]
            elif event["event"] == "error":
                return f"Error: {event['data']}"
        return final_answer or "No response generated."
