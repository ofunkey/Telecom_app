"""
Evaluate Retrieval Quality

Metric:
    Recall@3

For each question we check whether the expected
ticket appears in the top-3 retrieved documents.
"""

from retriever import build_retriever

# ---------------------------------------------------------
# Test Set
#
# Replace with your own ticket IDs
# ---------------------------------------------------------

TEST_CASES = [

    (
        "Why is my mobile internet so slow?",
        "TK-008",
    ),

    (
        "My calls keep dropping.",
        "TK-002",
    ),

    (
        "SIM not detected after restart",
        "TK-005",
    ),

    (
        "Duplicate charge on bill?",
        "TK-006",
    ),

    (
        "Why was I charged for roaming?",
        "TK-004",
    ),

    (
        "My call goes to voicemail?",
        "TK-007",
    ),

    (
        "Echo on voice call",
        "TK-017",
    ),

    (
        "Loss of mobile internet",
        "TK-001",
    ),

    (
        "Bill is not downloadable",
        "TK-010",
    ),

    (
        "Port-in takes too long",
        "TK-018",
    ),

]

retriever = build_retriever()

correct = 0

print("=" * 70)
print("Retrieval Evaluation")
print("=" * 70)

for question, expected_ticket in TEST_CASES:

    docs = retriever.invoke(question)

    # Only look at top 3
    top3 = docs[:3]

    retrieved_ids = []

    for doc in top3:

        ticket = doc.metadata.get(
            "ticket_id",
            None,
        )

        if ticket is not None:
            retrieved_ids.append(ticket)

    success = expected_ticket in retrieved_ids

    if success:
        correct += 1

    print(f"\nQuestion : {question}")
    print(f"Expected : {expected_ticket}")
    print(f"Retrieved: {retrieved_ids}")
    print(f"Result   : {'PASS' if success else 'FAIL'}")

recall = correct / len(TEST_CASES)

print("\n" + "=" * 70)
print(f"Recall@3 = {recall:.2%}")
print("=" * 70)