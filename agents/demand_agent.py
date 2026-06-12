from tools.forecast_tool import forecast_tool


def demand_agent(state):

    forecast = forecast_tool(
        sku=state["sku"],
        region=state["region"],
        temp=state["temp"],
        promo=state["promo"],
        event=state["event"],
        month=state["month"],
        dayofweek=state["dayofweek"]
    )

    boost = float(
        state.get(
            "demand_boost",
            0
        ) or 0
    )
    
    

#till here ^
    adjusted_forecast = (
        forecast *
        (1 + boost/100)
    )

    state["forecast"] = float(
        round(
            adjusted_forecast,
            2
        )
    )

    print(
        f"[Demand Agent] Forecast = {state['forecast']}"
    )

    return state