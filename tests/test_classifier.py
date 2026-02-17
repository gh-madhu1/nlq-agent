#!/usr/bin/env python3
"""Quick test of the intent classifier to verify it's working correctly."""

import sys
sys.path.insert(0, '/Users/madhukanukula/WORKSPACE/github-public/nlq-agent')

from utils.llm_client import LLMClient
from handlers.intent_safety import IntentSafetyClassifier

# Initialize
llm = LLMClient(model_provider="local", model_name="meta-llama/Llama-3.2-3B-Instruct")
classifier = IntentSafetyClassifier(llm)

# Test queries
test_queries = [
    "How many users are there?",
    "List all products under $20",
    "What is Alice's email?",
    "Tell me a joke",
    "What's the weather today?",
]

print("Testing Intent Classifier\n" + "="*50)
for query in test_queries:
    result = classifier.classify(query)
    print(f"\nQuery: {query}")
    print(f"  → Data Query: {result['is_data_query']}")
    print(f"  → Safe: {result['is_safe']}")
    print(f"  → Raw: {result['raw_response'][:100]}")
