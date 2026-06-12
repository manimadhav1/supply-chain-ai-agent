from llm import llm


def summary_agent(state):

    prompt = f"""
    You are a Coca-Cola supply chain analyst.

    Adjusted Forecast:
    {state['forecast']}

    Inventory:
    {state['inventory']}

    Demand Boost Applied:
    {state['demand_boost']}%

    Risk:
    {state['risk']}

    Reorder Quantity:
    {state['reorder_qty']}

    Generate a concise business summary.

    Mention:

    1. Demand outlook
    2. Inventory risk
    3. Recommended action

    Keep under 100 words.
    """

    response = llm.invoke(prompt)

    state["summary"] = response.content

    return state