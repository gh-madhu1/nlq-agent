from typing import List, Dict, Any
from utils.llm_client import LLMClient


class AnswerFormatter:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def format_response(self, query: str, results: List[Dict[str, Any]], sql: str = None) -> str:
        if not results:
            return "No results found for your query."

        # For simple single-value results, skip the LLM entirely
        if len(results) == 1 and len(results[0]) == 1:
            key, value = next(iter(results[0].items()))
            label = str(key).replace('_', ' ').title()
            return f"**{label}:** {value}"

        # Truncate results to prevent context overflow
        max_preview = 20
        results_preview = str(results[:max_preview])
        overflow_note = ""
        if len(results) > max_preview:
            overflow_note = f"\n(Showing {max_preview} of {len(results)} total records.)"

        prompt = f"""You are a data analyst. Convert the database results below into a clear, complete answer using Markdown formatting.

Rules:
- Use **bold** for key values and numbers
- Use bullet points (- item) for lists of 3+ items
- Use a markdown table if the result has multiple rows and columns
- Always finish your answer completely — do not cut off mid-sentence
- Be concise but thorough — cover all the data returned

User Question: "{query}"
SQL Query: {sql}
Results: {results_preview}{overflow_note}

Markdown Answer:"""

        response = self.llm_client.call(prompt)
        return response
