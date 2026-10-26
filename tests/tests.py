"""
Evaluation cases for the Phase 1 e-commerce RAG chatbot.

These cases cover:
- product retrieval
- policy retrieval
- missing information / grounding
- conversation routing
- multi-turn memory
- follow-up query rewriting
- exact product retrieval
"""


SINGLE_TURN_TESTS = [

    {
        "name": "product_recommendation",
        "question": "Which laptop should I buy?",
        "expected_route": "rag",
        "expected_source_type": "product",
        "notes": "Should retrieve relevant laptop products."
    },

    {
        "name": "product_battery_query",
        "question": "I want a laptop with good battery life.",
        "expected_route": "rag",
        "expected_source_type": "product",
        "notes": "Should retrieve laptops with relevant battery information."
    },

    {
        "name": "delivery_policy",
        "question": "How long does delivery take?",
        "expected_route": "rag",
        "expected_source_type": "policy",
        "expected_policy_type": "Delivery Policy",
        "notes": "Delivery Policy should be among the retrieved documents."
    },

    {
        "name": "cancellation_policy",
        "question": "Can I cancel an order after it has shipped?",
        "expected_route": "rag",
        "expected_source_type": "policy",
        "expected_policy_type": "Cancellation Policy",
        "notes": "Cancellation Policy should be retrieved."
    },

    {
        "name": "return_policy",
        "question": "Can I return an eligible product after 20 days?",
        "expected_route": "rag",
        "expected_source_type": "policy",
        "expected_policy_type": "Return Policy",
        "notes": "Return Policy should be retrieved."
    },

    {
        "name": "missing_product_information",
        "question": "Does NovaTech Laptop 171 include Microsoft Office?",
        "expected_route": "rag",
        "expected_source_type": "product",
        "expected_behavior": "fallback",
        "notes": (
            "Microsoft Office is not present in the product data. "
            "The chatbot should not substitute another attribute."
        )
    },

    {
        "name": "greeting",
        "question": "Hi",
        "expected_route": "conversation",
        "expected_behavior": "conversation",
        "notes": "Should not use RAG retrieval."
    },

    {
        "name": "mixed_greeting_product_query",
        "question": "Hi, can you recommend a laptop with good battery life?",
        "expected_route": "rag",
        "expected_source_type": "product",
        "notes": (
            "Product intent should take precedence over the greeting."
        )
    }
]


MULTI_TURN_TESTS = [

    {
        "name": "smartwatch_follow_up",
        "turns": [
            {
                "question": "Recommend a smartwatch.",
                "expected_route": "rag",
                "expected_retrieval_mode": "semantic"
            },
            {
                "question": "Tell me more about the smartwatch you recommended.",
                "expected_route": "rag",
                "expected_retrieval_mode": "exact_product",
                "notes": (
                    "Query rewriting should resolve the product "
                    "from conversation history."
                )
            },
            {
                "question": "How much does that smartwatch cost?",
                "expected_route": "rag",
                "expected_retrieval_mode": "exact_product",
                "notes": (
                    "The same smartwatch should be resolved through memory."
                )
            }
        ]
    },

    {
        "name": "laptop_position_follow_up",
        "turns": [
            {
                "question": "Which laptop should I buy?",
                "expected_route": "rag",
                "expected_retrieval_mode": "semantic"
            },
            {
                "question": "How about the third laptop you recommended?",
                "expected_route": "rag",
                "expected_retrieval_mode": "exact_product",
                "notes": (
                    "The rewrite node should resolve 'third laptop' "
                    "to the third product actually mentioned in the "
                    "previous assistant response."
                )
            }
        ]
    },

    {
        "name": "multiple_product_comparison",
        "turns": [
            {
                "question": "Which laptop should I buy?",
                "expected_route": "rag",
                "expected_retrieval_mode": "semantic"
            },
            {
                "question": "Which of those is the cheapest?",
                "expected_route": "rag",
                "notes": (
                    "Comparison follow-up. Current implementation may "
                    "resolve directly to one product. Keep this case as "
                    "an evaluation target for future comparison handling."
                )
            }
        ]
    }
]


if __name__ == "__main__":

    print("SINGLE-TURN TESTS")
    print("=" * 60)

    for test in SINGLE_TURN_TESTS:
        print(f"\n{test['name']}")
        print(f"Question: {test['question']}")
        print(f"Expected route: {test['expected_route']}")

    print("\n\nMULTI-TURN TESTS")
    print("=" * 60)

    for scenario in MULTI_TURN_TESTS:

        print(f"\nScenario: {scenario['name']}")

        for turn_number, turn in enumerate(
            scenario["turns"],
            start=1
        ):
            print(
                f"Turn {turn_number}: "
                f"{turn['question']}"
            )