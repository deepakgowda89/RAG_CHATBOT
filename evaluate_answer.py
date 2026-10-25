
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Load the existing embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to the existing vector database
vectorstores = Chroma(
    persist_directory="rag_db",
    embedding_function=embedding
)

retriever = vectorstores.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# 3. Initialize the LLM
llm = init_chat_model(
    model="openai/gpt-oss-20b",
    model_provider="groq",
    temperature=0
)

# 4. Define the grounded-answer prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a reliable document question-answering assistant.

        Answer using only the supplied context.

        If the context does not contain enough information,
        respond:
        "Sorry, I don't know based on the available documents."

        Do not invent facts.
        Cite supporting chunks using their provided source labels.
        """
    ),
    (
        "human",
        """
        Context:
        {context}

        Question:
        {question}
        """
    )
])

# 5. Prepare test cases
test_cases = [
    {
        "question": "How many days of annual paid leave do employees receive?",
        "expected": "20 days",
        "answerable": True
    },
    {
        "question": "What is the annual employee learning allowance?",
        "expected": "Rs. 15000",
        "answerable": True
    },
    {
        "question": "How many days per week can employees request remote work?",
        "expected": "two days per week",
        "answerable": True
    },
    {
        "question": "What are the company's password security requirements?",
        "expected": "never share company passwords",
        "answerable": True
    },
    {
        "question": "What is the company's cryptocurrency investment policy?",
        "expected": None,
        "answerable": False
    }
]

# 6. Run the evaluation
results = []

for index, test in enumerate(test_cases, start=1):
    question = test["question"]

    docs = retriever.invoke(question)

    context = "\n\n".join(
        f"[Source {i}]\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(final_prompt)
    answer = response.content

    # Simple baseline checks, not semantic evaluation
    if test["answerable"]:
        expected = test["expected"].lower()
        passed = expected in answer.lower()
    else:
        answer_lower = answer.lower()
        passed = (
            "don't know" in answer_lower
            or "do not know" in answer_lower
            or "not contain enough information" in answer_lower
            or "not enough information" in answer_lower
            or "not specified" in answer_lower
            or "cannot determine" in answer_lower
        )

    results.append(passed)

    print(f"\nTest {index}")
    print("Question:", question)
    print("Expected:", test["expected"])
    print("Answer:", answer)
    print("Result:", "PASS" if passed else "CHECK")

# 7. Summarize the evaluation
passed_count = sum(results)
total = len(results)

print("\n========== ANSWER EVALUATION ==========")
print(f"Total tests: {total}")
print(f"Passed baseline checks: {passed_count}")
print(f"Baseline pass rate: {passed_count / total:.2%}")
