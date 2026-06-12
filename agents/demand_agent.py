from tools.forecast_tool import forecast_tool


def demand_agent(state):

    forecast = forecast_tool(
        sku=state["sku"],
        region=state["region"],
        temp=state.get("temp", 0),
        promo=state.get("promo", 0),
        event=state.get("event", 0),
        month=state["month"],
        dayofweek=state["dayofweek"],
        channel=state.get("channel", "Unknown")
    )

    # Keep optional demand boost as a separate adjustment, not the core model output.
    boost = float(state.get("demand_boost", 0) or 0)
    adjusted_forecast = forecast * (1 + boost / 100)

    state["forecast"] = float(round(adjusted_forecast, 2))

    print(f"[Demand Agent] Forecast = {state['forecast']}")
    return state