def procurement_agent(state):

    forecast = state["forecast"]

    inventory = state["inventory"]

    safety_stock = (
        forecast * 0.2
    )

    reorder_qty = max(
        0,
        int(
            forecast
            + safety_stock
            - inventory
        )
    )

    state["reorder_qty"] = reorder_qty

    return state