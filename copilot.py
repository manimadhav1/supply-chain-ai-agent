from RAG.chroma_db.retriever import retrieve
from llm import llm


def ask_copilot(question):

    # Retrieve relevant knowledge
    docs = retrieve(question)

    context = "\n".join(docs)

    prompt = f"""
You are an AI Supply Chain Copilot.
Answer ONLY using the provided context.
If the answer is not present in the context,
say:
"I do not have enough information in the knowledge base."
Context:
{context}
Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":

    print("\n=== Supply Chain Copilot ===")

    while True:

        question = input("\nAsk a question (type exit to quit): ")

        if question.lower() == "exit":
            break

        answer = ask_copilot(question)

        print("\nAnswer:")
        print(answer)