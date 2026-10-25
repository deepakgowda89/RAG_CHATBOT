
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the same embedding model used during ingestion
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to the existing vector database
vectorstores = Chroma(
    persist_directory="rag_db",
    embedding_function=embedding
)

# 3. Create a small evaluation dataset
test_cases = [
    {
        "question": "How many days of annual paid leave do employees receive?",
        "expected": "20 days"
    },
    {
        "question": "What is the annual employee learning allowance?",
        "expected": "Rs. 15000"
    },
    {
        "question": "How many days per week can employees request remote work?",
        "expected": "two days per week"
    },
    {
        "question": "What are the company's password security requirements?",
        "expected": "never share company passwords"
    },
    {
        "question": "How many hours do employees work per day?",
        "expected": "eight hours per day"
    }
]

# 4. Evaluate retrieval
k = 5
hits = 0
reciprocal_ranks = []

print("\n========== RAG RETRIEVAL EVALUATION ==========\n")

for index, test in enumerate(test_cases, start=1):
    question = test["question"]
    expected = test["expected"].lower()

    docs = vectorstores.similarity_search(
        question,
        k=k
    )

    # Find the first retrieved chunk containing
    # the expected information.
    relevant_rank = None

    for rank, doc in enumerate(docs, start=1):
        if expected in doc.page_content.lower():
            relevant_rank = rank
            break

    if relevant_rank is not None:
        hits += 1
        reciprocal_ranks.append(1 / relevant_rank)
        status = "PASS"
    else:
        reciprocal_ranks.append(0)
        status = "FAIL"

    print(f"Test {index}: {status}")
    print("Question:", question)
    print("Expected:", test["expected"])
    print("Relevant rank:", relevant_rank)

    if docs:
        print("Top retrieved chunk:")
        print(docs[0].page_content[:250])

    print("-" * 60)

# 5. Calculate metrics
total = len(test_cases)

hit_rate = hits / total
mrr = sum(reciprocal_ranks) / total

print("\n========== FINAL METRICS ==========")
print(f"Total questions: {total}")
print(f"Successful retrievals: {hits}")
print(f"Hit Rate@{k}: {hit_rate:.2%}")
print(f"MRR@{k}: {mrr:.3f}")

