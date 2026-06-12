from tools.inventory_tool import inventory_tool


def inventory_agent(state):

    inventory = inventory_tool(
        state["sku"]
    )

    state["inventory"] = inventory

    print(
        f"[Inventory Agent] Stock = {inventory}"
    )

    return state