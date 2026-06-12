def risk_agent(state):

    forecast = state["forecast"]

    inventory = state["inventory"]

    shortage = max(
        0,
        forecast - inventory
    )

    daily_rate = forecast / 7

    days_cover = (
        inventory / daily_rate
    )

    if days_cover < 3:

        risk = "CRITICAL"

    elif days_cover < 7:

        risk = "HIGH"

    elif days_cover < 14:

        risk = "MEDIUM"

    else:

        risk = "LOW"

    state["shortage"] = float(
        round(shortage, 2)
    )

    state["days_cover"] = float(
        round(days_cover, 2)
    )

    state["risk"] = risk

    return state