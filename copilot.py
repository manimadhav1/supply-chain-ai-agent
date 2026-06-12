from RAG.chroma_db.retriever import retrieve
from llm import llm


def ask_copilot(question):
    docs = retrieve(question)

    if not docs:
        return "I do not have enough information in the knowledge base."

    context_items = []
    for item in docs:
        meta = item.get("metadata", {})
        meta_summary = []
        if meta:
            if meta.get("source"):
                meta_summary.append(f"source={meta['source']}")
            if meta.get("record_type"):
                meta_summary.append(f"type={meta['record_type']}")
            if meta.get("sku"):
                meta_summary.append(f"sku={meta['sku']}")
            if meta.get("sheet"):
                meta_summary.append(f"sheet={meta['sheet']}")
        context_items.append(f"[{', '.join(meta_summary)}]\n{item['document']}")

    context = "\n\n".join(context_items)

    prompt = f"""
You are an AI Supply Chain Copilot.
Answer only using the context provided below.
If the answer is not present in the context, say:
"I do not have enough information in the knowledge base."
Provide a concise, user-friendly response.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return response.content.strip()


if __name__ == "__main__":

    print("\n=== Supply Chain Copilot ===")

    while True:

        question = input("\nAsk a question (type exit to quit): ")

        if question.lower() == "exit":
            break

        answer = ask_copilot(question)

        print("\nAnswer:")
        print(answer)