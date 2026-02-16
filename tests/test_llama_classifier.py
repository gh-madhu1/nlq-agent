#!/usr/bin/env python3
"""Test intent classifier with Llama model."""

import sys
sys.path.insert(0, '/Users/madhukanukula/WORKSPACE/github-public/nlq-agent')

from utils.llm_client import LLMClient
from handlers.intent_safety import IntentSafetyClassifier

# Initialize with Llama
llm = LLMClient(model_provider="local", model_name="meta-llama/Llama-3.2-3B-Instruct")
classifier = IntentSafetyClassifier(llm)

# Test queries
test_queries = [
    "How many users are there?",
    "List all products under $20",
    "Tell me a joke",
]

print("Testing Intent Classifier with Llama-3.2-3B\n" + "="*60)
for query in test_queries:
    result = classifier.classify(query)
    print(f"\nQuery: {query}")
    print(f"  → Data Query: {result['is_data_query']}")
    print(f"  → Safe: {result['is_safe']}")
    print(f"  → Raw response:")
    print(f"     {result['raw_response']}")
    print()
