from tools.forecast_tool import forecast_tool


def demand_agent(state):
    forecast = forecast_tool(
        sku_id=state["sku"],
        region=state["region"],
        month=state["month"],
        dayofweek=state["dayofweek"],
        is_weekend=state["is_weekend"],
        temperature=state["temperature"],
        promo_type=state["promo_type"],
        promo_discount_pct=state["promo_discount_pct"],
        event_type=state["event_type"],
        is_competitor_stockout=state["is_competitor_stockout"],
    )

    boost = float(state.get("demand_boost", 0) or 0)
    state["forecast"] = round(forecast * (1 + boost / 100), 2)

    print(f"[Demand Agent] Base forecast={forecast}, boost={boost}%, adjusted={state['forecast']}")
    return state
