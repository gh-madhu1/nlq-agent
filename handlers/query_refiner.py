"""
Query Refiner — normalizes messy user input into clean, structured questions.

Handles typos, abbreviations, informal language, and maps user terms
to actual database schema names for better downstream SQL generation.
"""

import re
import logging
from typing import Dict, List, Any

from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class QueryRefiner:
    """
    Refines user queries before SQL generation.

    Two-stage approach:
    1. Fast keyword mapping (deterministic) — maps slang/abbreviations to schema terms
    2. LLM refinement (if needed) — cleans grammar and clarifies intent
    """

    # Common abbreviations / slang → schema terms
    TERM_MAP = {
        "ppl": "users",
        "people": "users",
        "customers": "users",
        "customer": "users",
        "buyers": "users",
        "buyer": "users",
        "stuff": "products",
        "items": "products",
        "goods": "products",
        "purchases": "orders",
        "bought": "ordered",
        "buy": "order",
        "cost": "price",
        "costs": "price",
        "amt": "amount",
        "qty": "quantity",
        "num": "number",
        "cats": "categories",
        "cat": "category",
        "inv": "inventory",
        "stock": "stock_quantity",
    }

    def __init__(self, llm_client: LLMClient, schema_info: Dict[str, Dict[str, Any]]):
        self.llm_client = llm_client
        self.schema_info = schema_info

        # Build a set of known schema terms for matching
        self._schema_terms = set()
        for table, info in schema_info.items():
            self._schema_terms.add(table.lower())
            for col_def in info.get("columns", []):
                col_name = col_def.split()[0].lower()
                self._schema_terms.add(col_name)

    def refine(self, user_query: str) -> str:
        """
        Refine a user query into a clean, structured question.

        Returns the refined query string.
        """
        original = user_query.strip()
        if not original:
            return original

        # Stage 1: Fast deterministic keyword mapping
        refined = self._apply_term_mapping(original)

        # Stage 2: If query looks messy (short, has typos, no verb), use LLM
        if self._needs_llm_refinement(original):
            try:
                refined = self._llm_refine(refined)
                logger.info(f"Query refined: '{original}' → '{refined}'")
            except Exception as e:
                logger.warning(f"LLM refinement failed, using keyword-mapped query: {e}")
        else:
            logger.info(f"Query clean enough, skipping LLM refinement: '{refined}'")

        return refined

    def _apply_term_mapping(self, query: str) -> str:
        """Replace common abbreviations/slang with schema-accurate terms."""
        result = query
        for slang, proper in self.TERM_MAP.items():
            # Word-boundary replacement to avoid partial matches
            pattern = rf'\b{re.escape(slang)}\b'
            result = re.sub(pattern, proper, result, flags=re.IGNORECASE)
        return result

    def _needs_llm_refinement(self, query: str) -> bool:
        """Heuristic: does this query need LLM help to clean up?"""
        words = query.split()

        # Very short or fragmented queries
        if len(words) <= 2:
            return True

        # Contains obvious typos (words not in dictionary or schema)
        misspelling_signals = 0
        for word in words:
            w = re.sub(r'[^a-zA-Z]', '', word).lower()
            if len(w) > 3 and w not in self._schema_terms and not self._is_common_word(w):
                misspelling_signals += 1

        # If more than 30% of words look wrong, refine
        if misspelling_signals > len(words) * 0.3:
            return True

        return False

    @staticmethod
    def _is_common_word(word: str) -> bool:
        """Check if a word is a common English word (simple allowlist)."""
        common = {
            "show", "me", "the", "all", "get", "find", "list", "what", "how",
            "many", "much", "is", "are", "was", "were", "do", "does", "did",
            "from", "in", "by", "for", "with", "and", "or", "not", "to", "of",
            "each", "every", "per", "total", "average", "count", "sum", "max",
            "min", "most", "least", "top", "bottom", "first", "last", "recent",
            "new", "old", "between", "under", "over", "above", "below", "less",
            "more", "than", "greater", "highest", "lowest", "cheapest", "expensive",
            "who", "which", "where", "when", "that", "this", "there", "their",
            "has", "have", "had", "been", "being", "a", "an", "it", "its",
            "purchased", "ordered", "placed", "made", "sold", "bought", "spent",
            "number", "name", "email", "date", "amount", "price", "quantity",
            "category", "product", "user", "order", "item", "items",
            "products", "users", "orders", "sales", "revenue", "inventory",
            # Conversational / greeting words (avoid LLM refinement for casual chat)
            "hello", "hi", "hey", "thanks", "thank", "please", "sorry",
            "yes", "no", "okay", "sure", "great", "good", "fine",
            "tell", "help", "can", "could", "would", "should", "will",
            "about", "just", "only", "also", "too", "very", "really",
            "doing", "going", "today", "yesterday", "tomorrow", "now",
            "right", "well", "like", "want", "need", "know", "think",
            "see", "look", "give", "take", "some", "any", "other",
            "why", "because", "but", "if", "then", "so", "up", "out",
            "on", "off", "at", "into", "as", "no", "be", "my", "your",
        }
        return word in common

    def _llm_refine(self, query: str) -> str:
        """Use LLM to clean up a messy query."""
        # Build table summary for context
        tables_summary = ", ".join(self.schema_info.keys())

        prompt = f"""Rewrite this database question to be clear and grammatically correct.

Database tables: {tables_summary}

Original question: "{query}"

Rules:
1. Fix any spelling or grammar errors
2. Make the question clear and specific
3. Keep the same meaning — do NOT change what is being asked
4. Use proper database terms (users, products, orders, order_items)
5. Return ONLY the rewritten question, nothing else

Rewritten question:"""

        response = self.llm_client.call(
            prompt,
            temperature=0.0,
            max_tokens=100,
            stop=["\n\n", "Original:", "Rules:"]
        )

        refined = response.strip().strip('"').strip("'")

        # Sanity check: if LLM returned something too different or too long, use original
        if len(refined) > len(query) * 3 or len(refined) < 5:
            return query

        return refined
