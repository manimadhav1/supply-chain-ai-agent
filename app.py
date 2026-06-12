import streamlit as st
import pandas as pd

from workflow.graph import graph


st.set_page_config(
    page_title="Supply Chain Copilot",
    page_icon="📦",
    layout="wide"
)

# ==================================================
# HEADER
# ==================================================

st.title("📦 Supply Chain Copilot")

st.markdown("""
### AI-Powered Demand Forecasting & Inventory Intelligence

This system analyzes:

Historical demand patterns
Inventory availability
Sales signals
Market events

and recommends replenishment actions.
""")

st.divider()

# ==================================================
# INPUTS
# ==================================================

col1, col2 = st.columns(2)

with col1:

    sku = st.selectbox(
        "SKU",
        [
            "Coke500",
            "Sprite500",
            "Fanta500",
            "ThumsUp500"
        ]
    )

    region = st.selectbox(
        "Region",
        [
            "Bangalore",
            "Chennai",
            "Mumbai",
            "Hyderabad"
        ]
    )

    temp = st.slider(
        "Temperature (°C)",
        min_value=20,
        max_value=45,
        value=35
    )

with col2:

    promo = st.selectbox(
        "Promotion Running?",
        [0, 1]
    )

    event = st.selectbox(
        "External Event?",
        [0, 1]
    )

sales_note = st.text_area(
    "Sales Notes",
    height=150,
    placeholder="Example: IPL finals expected to increase beverage sales. Retailers expecting strong demand."
)

# ==================================================
# ANALYZE BUTTON
# ==================================================

if st.button("🚀 Analyze Demand"):

    state = {

        "sku": sku,
        "region": region,

        "temp": temp,
        "promo": promo,
        "event": event,

        "month": 6,
        "dayofweek": 5,

        "sales_note": sales_note,

        "forecast": 0,
        "inventory": 0,
        "shortage": 0,
        "days_cover": 0,
        "risk": "",
        "reorder_qty": 0,
        "demand_boost": 0,
        "summary": ""
    }

    with st.spinner("Running Supply Chain Analysis..."):

        result = graph.invoke(state)

    st.divider()

    # ==================================================
    # KPI CARDS
    # ==================================================

    st.subheader("Key Metrics")

    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        st.metric(
            "Forecast Units",
            f"{round(result['forecast']):,}"
        )

    with row1_col2:
        st.metric(
            "Current Inventory",
            f"{result['inventory']:,}"
        )

    with row1_col3:
        st.metric(
            "Demand Increase",
            f"{result['demand_boost']}%"
        )

    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        st.metric(
            "Coverage Days",
            result["days_cover"]
        )

    with row2_col2:
        st.metric(
            "Shortage",
            f"{round(result['shortage']):,}"
        )

    with row2_col3:
        st.metric(
            "Reorder Quantity",
            f"{result['reorder_qty']:,}"
        )

    st.divider()

    # ==================================================
    # RISK SECTION
    # ==================================================

    st.subheader("Risk Assessment")

    risk = result["risk"]

    if risk == "CRITICAL":
        st.error(f"🔴 {risk}")

    elif risk == "HIGH":
        st.warning(f"🟠 {risk}")

    elif risk == "MEDIUM":
        st.warning(f"🟡 {risk}")

    else:
        st.success(f"🟢 {risk}")

    st.divider()

    # ==================================================
    # INVENTORY GAP CHART
    # ==================================================

    st.subheader("Inventory Gap Analysis")

    chart_df = pd.DataFrame(
        {
            "Units": [
                result["forecast"],
                result["inventory"]
            ]
        },
        index=[
            "Forecast",
            "Inventory"
        ]
    )

    st.bar_chart(chart_df)

    st.divider()

    # ==================================================
    # BUSINESS IMPACT
    # ==================================================

    st.subheader("Business Impact")

    coverage = result["days_cover"]

    if coverage < 3:

        st.error(
            f"Current inventory will only last approximately {coverage} days."
        )

    elif coverage < 7:

        st.warning(
            f"Inventory coverage is only {coverage} days."
        )

    else:

        st.success(
            f"Inventory coverage is {coverage} days."
        )

    st.divider()

    # ==================================================
    # AGENT DECISION TRAIL
    # ==================================================

    st.subheader("Agent Decision Trail")

    st.markdown(f"""
**Sales Signal Agent**
Detected demand increase of **{result['demand_boost']}%**

⬇️

**Demand Agent**
Forecast generated: **{round(result['forecast']):,} units**

⬇️

**Inventory Agent**
Current stock: **{result['inventory']:,} units**

⬇️

**Risk Agent**
Coverage Days: **{result['days_cover']}**
Risk Level: **{result['risk']}**

⬇️

**Procurement Agent**
Recommended reorder quantity: **{result['reorder_qty']:,} units**
""")

    st.divider()

    # ==================================================
    # AI SUMMARY
    # ==================================================

    st.subheader("Executive Recommendation")

    with st.container(border=True):
        st.write(result["summary"])