"""
Intent & Safety Classifier — keyword-based, no LLM calls.

Fast and reliable classification of user queries:
- Is this a data/database query?
- Is it safe (no SQL injection)?
"""

import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class IntentSafetyClassifier:
    """
    Keyword-based intent and safety classifier.

    Removed LLM call — keyword matching is faster and more reliable
    for this binary classification task.
    """

    # Keywords that strongly indicate a data query
    DATA_KEYWORDS = {
        # Question words in data context
        "how many", "count", "total", "sum", "average", "avg",
        "maximum", "minimum", "max", "min",
        # Action words
        "list", "show", "get", "find", "display", "fetch",
        "what", "which", "who", "when",
        # Domain terms
        "products", "product", "users", "user", "orders", "order",
        "items", "item", "customers", "customer",
        "email", "name", "price", "category", "stock",
        "revenue", "sales", "purchases", "bought", "purchased",
        "inventory", "quantity", "amount",
        # Aggregation/analysis words
        "each", "per", "by", "group", "top", "most",
        "cheapest", "expensive", "highest", "lowest",
        "recent", "latest", "last",
    }

    # Patterns that indicate unsafe intent (SQL injection, etc.)
    UNSAFE_PATTERNS = [
        r'\bdrop\b', r'\bdelete\b', r'\bupdate\b', r'\binsert\b',
        r'\balter\b', r'\btruncate\b', r'\bgrant\b', r'\brevoke\b',
        r'\bcreate\b', r'\bexec\b',
        r'--',          # SQL comment injection
        r';\s*\w',      # Multiple statements
        r'\bunion\b',   # UNION injection
    ]

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Unused, kept for backward compatibility.
        """
        pass

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify query intent and safety.

        Returns:
            {
                "is_data_query": bool,
                "is_safe": bool,
                "confidence": float,  # 0.0-1.0
            }
        """
        is_safe = self._is_safe(query)
        is_data_query, confidence = self._is_data_query(query)

        result = {
            "is_data_query": is_data_query,
            "is_safe": is_safe,
            "confidence": confidence,
        }

        logger.info(f"Classification: data_query={is_data_query} (conf={confidence:.2f}), safe={is_safe}")
        return result

    def _is_data_query(self, query: str) -> tuple:
        """
        Determine if query is asking for data from the database.

        Returns: (is_data_query, confidence)
        """
        query_lower = query.lower()
        words = set(re.findall(r'\b\w+\b', query_lower))

        # Count matching data keywords
        matches = 0
        matched_keywords = []
        for keyword in self.DATA_KEYWORDS:
            if ' ' in keyword:
                # Multi-word keyword — check as substring
                if keyword in query_lower:
                    matches += 2  # Multi-word matches are stronger signals
                    matched_keywords.append(keyword)
            else:
                if keyword in words:
                    matches += 1
                    matched_keywords.append(keyword)

        # Calculate confidence based on match density
        if matches >= 3:
            return True, min(0.95, 0.5 + matches * 0.1)
        elif matches >= 1:
            return True, 0.3 + matches * 0.15
        else:
            return False, 0.1

    def _is_safe(self, query: str) -> bool:
        """Check if query contains unsafe SQL injection patterns."""
        query_lower = query.lower()

        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, query_lower):
                logger.warning(f"Unsafe pattern detected: {pattern} in query: {query}")
                return False

        return True
