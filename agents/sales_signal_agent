from RAG.chroma_db.retriever import retrieve
from llm import llm
import json


def sales_signal_agent(state):

    sales_note = state["sales_note"]

    try:

        # ==========================
        # RAG Retrieval
        # ==========================

        print("\nSTEP 0 - Retrieving context")

        docs = retrieve(
            sales_note,
            n_results=2
        )

        rag_context = "\n".join(docs)

        print("\n[Sales Signal RAG Context]")
        print(rag_context)
        print()

        # ==========================
        # Prompt
        # ==========================

        prompt = f"""
You are a supply chain intelligence system.

Your task is to estimate the expected demand increase percentage.

Use BOTH:
1. The current sales note
2. Historical business context

Return ONLY valid JSON.

Example:

{{
    "event":"IPL Finals",
    "demand_boost":25
}}

Rules:

- Minor increase → 10-20
- Strong increase → 20-35
- Very high increase → 35-50
- No increase → 0

Historical Context:
{rag_context}

Sales Note:
{sales_note}

Return ONLY JSON.
"""

        # ==========================
        # Gemini Call
        # ==========================

        print("STEP 1 - Calling Gemini")

        response = llm.invoke(prompt)

        print("STEP 2 - Gemini returned")

        print("\n===== GEMINI RAW RESPONSE =====")
        print(response.content)
        print("===============================\n")

        # ==========================
        # Clean Response
        # ==========================

        print("STEP 3 - Cleaning response")

        content = response.content

        content = content.replace(
            "```json",
            ""
        )

        content = content.replace(
            "```",
            ""
        )

        content = content.strip()

        print("STEP 4 - Parsing JSON")

        result = json.loads(content)

        print("STEP 5 - JSON parsed")

        boost = result.get(
            "demand_boost",
            0
        )

        if boost is None:
            boost = 0

        state["demand_boost"] = float(boost)

        print(
            f"[Sales Signal Agent] Boost = {state['demand_boost']}%"
        )

    except Exception as e:

        print("\n===== SALES SIGNAL ERROR =====")
        print(e)
        print("==============================\n")

        note = sales_note.lower()

        # Fallback logic when Gemini fails or quota is exhausted
        if "ipl" in note:
            state["demand_boost"] = 30

        elif "festival" in note:
            state["demand_boost"] = 25

        elif "summer" in note:
            state["demand_boost"] = 20

        else:
            state["demand_boost"] = 10

        print(
            f"[Fallback] Demand Boost = {state['demand_boost']}%"
        )

    print("STEP 6 - Returning state")

    return state