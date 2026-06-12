from copilot import ask_copilot
import streamlit as st
import pandas as pd
import altair as alt
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from workflow.graph import graph
from tools.forecast_tool import forecast_tool


# ==================================================
# CACHED FUNCTIONS
# ==================================================

@st.cache_data
def compute_mape():
    df = pd.read_csv("data/sales_v2.csv")
    sku_enc    = joblib.load("models/sku_encoder.pkl")
    cat_enc    = joblib.load("models/category_encoder.pkl")
    region_enc = joblib.load("models/region_encoder.pkl")
    model      = joblib.load("models/forecast_model.pkl")
    features   = joblib.load("models/feature_list.pkl")
    df["sku_encoded"]      = sku_enc.transform(df["sku_id"])
    df["category_encoded"] = cat_enc.transform(df["category"])
    df["region_encoded"]   = region_enc.transform(df["region"])
    X, y = df[features], df["sales"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preds = model.predict(X_test)
    return round(float(np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100), 1)


@st.cache_data
def compute_transfer_recommendations():
    inv_df = pd.read_csv("data/inventory_by_region.csv")
    recommendations = []
    for sku_id in inv_df["sku_id"].unique():
        sku_df = inv_df[inv_df["sku_id"] == sku_id].copy()
        product_name = sku_df.iloc[0]["product_name"]
        category = sku_df.iloc[0]["category"]
        critical = sku_df[sku_df["days_cover"] < 5].sort_values("days_cover")
        overstock = sku_df[sku_df["days_cover"] > 10].sort_values("days_cover", ascending=False)
        for _, crit_row in critical.iterrows():
            for _, over_row in overstock.iterrows():
                transfer_qty = min(int(over_row["daily_demand"] * 7), int(over_row["stock"] * 0.4))
                if transfer_qty > 0:
                    recommendations.append({
                        "SKU ID": sku_id, "Product": product_name, "Category": category,
                        "From Region": over_row["region"], "From Stock": over_row["stock"],
                        "From Coverage": f"{over_row['days_cover']}d",
                        "To Region": crit_row["region"], "To Stock": crit_row["stock"],
                        "To Coverage": f"{crit_row['days_cover']}d",
                        "Transfer Qty": transfer_qty,
                        "Priority": (
                            "🔴 P1 · CRITICAL" if crit_row["days_cover"] < 2 else
                            "🟠 P2 · URGENT"   if crit_row["days_cover"] < 3 else
                            "🟡 P3 · HIGH"     if crit_row["days_cover"] < 4 else
                            "🟢 P4 · MEDIUM"
                        ),
                        "Days Left at Dest": crit_row["days_cover"]
                    })
    return inv_df, pd.DataFrame(recommendations)


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(page_title="Supply Chain Copilot", page_icon="📦", layout="wide")

# ==================================================
# SESSION STATE
# ==================================================

if "page" not in st.session_state:
    st.session_state.page = "forecast"

# ==================================================
# THEME VARIABLES — dark mode only
# ==================================================

bg          = "#0e1117"
sidebar_bg  = "#161b22"
card_bg     = "#1c2128"
border      = "#30363d"
text_pri    = "#e6edf3"
text_sec    = "#8b949e"
accent      = "#58a6ff"
hover       = "#21262d"
input_bg    = "#161b22"

# ==================================================
# CSS INJECTION
# ==================================================

st.markdown(f"""
<style>
.stApp {{ background: {bg}; }}
.stApp > header {{ background: {sidebar_bg}; border-bottom: 1px solid {border}; }}
section[data-testid="stSidebar"] {{ background: {sidebar_bg}; border-right: 1px solid {border}; min-width: 230px; }}
section[data-testid="stSidebar"] > div:first-child {{ background: {sidebar_bg}; padding-top: 0; }}
.main .block-container {{ padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }}
h1, h2, h3, h4 {{ color: {text_pri}; font-weight: 600; }}
p, li, label {{ color: {text_sec}; }}
hr {{ border-color: {border}; opacity: 0.5; }}
[data-testid="metric-container"] {{ background: {card_bg}; border: 1px solid {border}; border-radius: 10px; padding: 16px 20px; }}
[data-testid="stMetricValue"] {{ color: {text_pri}; font-size: 26px; font-weight: 600; }}
[data-testid="stMetricLabel"] {{ color: {text_sec}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
[data-testid="stMetricDelta"] {{ font-size: 12px; }}
.stButton > button {{ background: {card_bg}; color: {text_pri}; border: 1px solid {border}; border-radius: 8px; font-size: 14px; font-weight: 500; padding: 8px 18px; transition: all 0.15s; }}
.stButton > button:hover {{ background: {hover}; border-color: {accent}; color: {accent}; }}
.nav-btn button {{ width:100% !important; text-align:left !important; background:transparent !important; border:none !important; border-radius:8px !important; color:{text_sec} !important; font-size:15px !important; font-weight:500 !important; padding:11px 14px !important; margin-bottom:2px !important; transition:background 0.15s, color 0.15s !important; letter-spacing:0.1px !important; }}
.nav-btn button:hover {{ background:{hover} !important; color:{text_pri} !important; }}
.nav-btn-active button {{ background:{hover} !important; color:{accent} !important; border-left:3px solid {accent} !important; font-weight:600 !important; padding-left:11px !important; }}
.stSelectbox > div > div {{ background: {input_bg}; border-color: {border}; color: {text_pri}; border-radius: 8px; }}
.stTextInput > div > div > input {{ background: {input_bg}; border-color: {border}; color: {text_pri}; border-radius: 8px; }}
.stTextArea > div > div > textarea {{ background: {input_bg}; border-color: {border}; color: {text_pri}; border-radius: 8px; }}
.stSlider > div {{ color: {text_pri}; }}
.stTabs [data-baseweb="tab-list"] {{ background: {card_bg}; border-bottom: 1px solid {border}; border-radius: 8px 8px 0 0; gap: 0; padding: 0 8px; }}
.stTabs [data-baseweb="tab"] {{ color: {text_sec}; font-size: 13px; font-weight: 500; padding: 10px 16px; border-radius: 0; border-bottom: 2px solid transparent; }}
.stTabs [aria-selected="true"] {{ color: {accent}; border-bottom: 2px solid {accent}; }}
.stTabs [data-baseweb="tab-panel"] {{ background: {card_bg}; border: 1px solid {border}; border-top: none; border-radius: 0 0 8px 8px; padding: 16px; }}
.stDataFrame {{ background: {card_bg}; border-radius: 8px; border: 1px solid {border}; }}
.stDataFrame th {{ background: {hover}; color: {text_sec}; font-size: 12px; }}
.stDataFrame td {{ color: {text_pri}; font-size: 13px; }}
.stCaption, .caption {{ color: {text_sec}; font-size: 12px; }}
.stAlert {{ border-radius: 8px; }}
div[data-testid="stSidebarNav"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# ==================================================
# SIDEBAR
# ==================================================

if "active_page" not in st.session_state:
    st.session_state.active_page = "Forecast"

with st.sidebar:
    st.markdown(
        f'<div style="padding:16px 16px 18px 16px; border-bottom:1px solid {border}; margin-bottom:12px;">'
        f'<div style="font-size:32px; font-weight:900; color:{text_pri}; letter-spacing:-1px; line-height:1.1;">📦 SupplyAI</div>'
        f'<div style="font-size:12px; color:{text_sec}; margin-top:5px; letter-spacing:0.5px;">Intelligent Supply Chain Copilot</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="font-size:10px; color:{text_sec}; letter-spacing:2px; padding:4px 4px 8px 4px; text-transform:uppercase; font-weight:700; opacity:0.6;">Menu</div>',
        unsafe_allow_html=True
    )

    NAV_ITEMS = [
        ("Forecast",   "📈", "Forecast & Analysis"),
        ("Copilot",    "🤖", "AI Copilot"),
        ("Transfer",   "🔁", "Stock Transfers"),
        ("Knowledge",  "🧠", "Knowledge Base"),
    ]

    for key, icon, label in NAV_ITEMS:
        is_active = st.session_state.active_page == key
        css_class = "nav-btn-active" if is_active else "nav-btn"
        with st.container():
            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.active_page = key
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    page = st.session_state.active_page

    mape = compute_mape()
    if mape <= 10:
        mape_color, mape_label = "#25d366", "High Confidence"
    elif mape <= 20:
        mape_color, mape_label = "#f5a623", "Moderate Confidence"
    else:
        mape_color, mape_label = "#e94560", "Low Confidence"

    st.markdown(
        f'<div style="border-top:1px solid {border}; margin:16px 0 14px 0;"></div>'
        f'<div style="background:{card_bg}; border:1px solid {border}; border-left:4px solid {mape_color}; border-radius:10px; padding:14px 16px; margin-bottom:16px;">'
        f'<div style="font-size:10px; color:{text_sec}; letter-spacing:1px; text-transform:uppercase; font-weight:700; margin-bottom:6px; opacity:0.7;">AI Model Accuracy</div>'
        f'<div style="font-size:26px; font-weight:800; color:{mape_color}; letter-spacing:-0.5px;">MAPE {mape}%</div>'
        f'<div style="font-size:12px; color:{text_sec}; margin-top:4px;">{mape_label} · XGBoost v2</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)

    st.markdown(
        f'<div style="position:absolute; bottom:20px; left:16px; right:16px; border-top:1px solid {border}; padding-top:14px;">'
        f'<div style="font-size:11px; color:{text_sec}; line-height:2; opacity:0.65;">'
        f'Powered by<br>'
        f'<span style="color:{accent}; font-weight:600;">Gemini 2.5 Flash</span> · XGBoost · LangGraph'
        f'</div></div>',
        unsafe_allow_html=True
    )


# ==================================================
# PAGE: FORECAST & ANALYSIS
# ==================================================

if page == "Forecast":

    st.markdown(f"<h1 style='margin-bottom:4px;'>📈 Demand Forecast & Analysis</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{text_sec}; margin-bottom:20px;'>Select a product and city, set today's conditions, then run the AI analysis to get reorder recommendations.</p>", unsafe_allow_html=True)

    st.divider()

    # --- INPUTS ---
    _master_df = pd.read_csv("data/product_master.csv")
    sku_options = {
        f"{r['sku_id']} — {r['product_name']} ({r['category']})": r['sku_id']
        for _, r in _master_df.iterrows()
    }

    st.markdown(f"<h3 style='margin-bottom:12px;'>🔧 Set Today's Conditions</h3>", unsafe_allow_html=True)

    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        _label_help = f"<div style='font-size:12px; color:{text_sec}; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.5px;'>"
        st.markdown(_label_help + "Product (SKU)</div>", unsafe_allow_html=True)
        sku_label = st.selectbox("SKU", list(sku_options.keys()), label_visibility="collapsed")
        sku = sku_options[sku_label]

        st.markdown(_label_help + "City / Zone</div>", unsafe_allow_html=True)
        _REGION_OPTIONS = {
            "🏙 Bangalore (South)":            "Bangalore",
            "🏙 Chennai (South)":              "Chennai",
            "🏙 Mumbai (West)":                "Mumbai",
            "🏙 Hyderabad (Central-South)":    "Hyderabad",
            "🗺 North India — Delhi/NCR":       "Mumbai",
            "🗺 East India — Kolkata/Odisha":   "Chennai",
            "🗺 West India — Pune/Ahmedabad":   "Mumbai",
            "🗺 Central India — Nagpur/Indore": "Hyderabad",
        }
        region_label = st.selectbox("Region", list(_REGION_OPTIONS.keys()), label_visibility="collapsed")
        region = _REGION_OPTIONS[region_label]
        if "🗺" in region_label:
            st.markdown(f"<div style='font-size:10px; color:{text_sec}; margin-top:-8px; margin-bottom:8px; opacity:0.7;'>↳ using {region} demand model</div>", unsafe_allow_html=True)

    with row1_col2:
        import datetime
        st.markdown(_label_help + "Forecast Date</div>", unsafe_allow_html=True)
        forecast_date = st.date_input("Date", value=datetime.date.today(), label_visibility="collapsed")
        _month = forecast_date.month
        _dow   = forecast_date.weekday()
        _is_weekend = int(_dow >= 5)
        st.markdown(f"<div style='font-size:11px; color:{text_sec}; margin-top:-8px; margin-bottom:10px;'>↳ {forecast_date.strftime('%A, %d %b %Y')}</div>", unsafe_allow_html=True)

        st.markdown(_label_help + "Temperature (°C)</div>", unsafe_allow_html=True)
        temperature = st.slider("Temp", min_value=18, max_value=46, value=32, label_visibility="collapsed",
                                help="Average temperature for the day — affects demand for beverages, food, dairy")

        st.markdown(_label_help + "Competitor Out of Stock?</div>", unsafe_allow_html=True)
        is_competitor_stockout = int(st.checkbox("Yes — competitor shelves are empty", value=False,
                                                  help="Check this if key competitors have run out of this product in the market"))

    with row1_col3:
        st.markdown(_label_help + "Promotion Type</div>", unsafe_allow_html=True)
        promo_label_map = {
            "No Promotion": 0,
            "Price Discount (5–15%)": 1,
            "Price Discount (>15%)": 2,
            "Buy One Get One (BOGO)": 3,
            "Bundle Deal": 4,
        }
        promo_label = st.selectbox("Promo Type", list(promo_label_map.keys()), label_visibility="collapsed")
        promo_type = promo_label_map[promo_label]

        if promo_type in [1, 2]:
            st.markdown(_label_help + "Discount Percentage (%)</div>", unsafe_allow_html=True)
            promo_discount_pct = st.slider("Discount %", min_value=5, max_value=50, value=10 if promo_type == 1 else 20, label_visibility="collapsed")
        else:
            promo_discount_pct = 0

        st.markdown(_label_help + "Event / Occasion</div>", unsafe_allow_html=True)
        event_label_map = {
            "No Special Event": 0,
            "Local Festival (Pongal, Navratri, Onam…)": 1,
            "Sports Event (IPL, Cricket, FIFA…)": 2,
            "National Holiday (Republic Day, Diwali…)": 3,
            "Wedding Season": 4,
        }
        event_label = st.selectbox("Event Type", list(event_label_map.keys()), label_visibility="collapsed")
        event_type = event_label_map[event_label]

        st.markdown(_label_help + "Sales Team Notes</div>", unsafe_allow_html=True)
        sales_note = st.text_area("Notes", height=80,
                                  placeholder="E.g. Big retail push this week, IPL finals on Saturday, competitor out of stock near Koramangala...",
                                  label_visibility="collapsed")

    # Backward compat aliases used by the state dict and 7-day chart
    promo = 1 if promo_type > 0 else 0
    event = 1 if event_type > 0 else 0
    temp  = temperature

    st.divider()

    # --- SCENARIO SIMULATOR ---
    st.markdown(f"<h3 style='margin-bottom:4px;'>🎛️ What-If Simulator</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{text_sec}; font-size:13px; margin-bottom:16px;'>Instantly compare your scenario against a normal baseline day — no AI wait time required.</p>", unsafe_allow_html=True)

    sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
    with sim_col1:
        sim_temp = st.slider("Sim Temperature (°C)", min_value=18, max_value=46, value=temperature, key="sim_temp")
    with sim_col2:
        sim_promo_label = st.selectbox("Sim Promotion", list(promo_label_map.keys()), index=promo_type, key="sim_promo")
        sim_promo_type = promo_label_map[sim_promo_label]
    with sim_col3:
        sim_event_label = st.selectbox("Sim Event", list(event_label_map.keys()), index=event_type, key="sim_event")
        sim_event_type = event_label_map[sim_event_label]
    with sim_col4:
        sim_competitor_out = int(st.checkbox("Competitor OOS (sim)", value=bool(is_competitor_stockout), key="sim_comp"))

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sim_daily = [
        round(forecast_tool(
            sku_id=sku, region=region, month=_month, dayofweek=d,
            is_weekend=int(d >= 5), temperature=sim_temp,
            promo_type=sim_promo_type, promo_discount_pct=promo_discount_pct,
            event_type=sim_event_type, is_competitor_stockout=sim_competitor_out,
        )) for d in range(7)
    ]
    baseline_daily = [
        round(forecast_tool(
            sku_id=sku, region=region, month=_month, dayofweek=d,
            is_weekend=int(d >= 5), temperature=32,
            promo_type=0, promo_discount_pct=0,
            event_type=0, is_competitor_stockout=0,
        )) for d in range(7)
    ]

    sim_df = pd.DataFrame({
        "Day": day_labels * 2,
        "Units": sim_daily + baseline_daily,
        "Scenario": ["Your Scenario"] * 7 + ["Normal Day (baseline)"] * 7
    })

    sim_chart = alt.Chart(sim_df).mark_line(point=True).encode(
        x=alt.X("Day:N", sort=day_labels, title="Day of Week"),
        y=alt.Y("Units:Q", title="Units Expected to Sell"),
        color=alt.Color("Scenario:N", scale=alt.Scale(range=[accent, text_sec])),
        tooltip=["Day", "Scenario", "Units"]
    ).properties(height=260).configure_view(strokeWidth=0).configure_axis(
        gridColor=border, labelColor=text_sec, titleColor=text_sec
    ).configure_legend(labelColor=text_sec, titleColor=text_sec)

    st.altair_chart(sim_chart, use_container_width=True)

    sim_total = sum(sim_daily)
    baseline_total = sum(baseline_daily)
    delta = sim_total - baseline_total
    delta_pct = round((delta / baseline_total) * 100, 1) if baseline_total else 0

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Weekly Demand — Your Scenario", f"{sim_total:,} units")
    sc2.metric("Weekly Demand — Normal Day", f"{baseline_total:,} units")
    sc3.metric("Extra Units Needed", f"{delta:+,}", delta=f"{delta_pct:+.1f}%")

    st.divider()

    # --- ANALYZE BUTTON ---
    st.markdown(f"<h3 style='margin-bottom:8px;'>🚀 Run Full AI Analysis</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{text_sec}; font-size:13px; margin-bottom:16px;'>This runs 5 AI agents in sequence — reads your sales notes, forecasts demand, checks stock, assesses risk, and recommends a reorder quantity.</p>", unsafe_allow_html=True)

    if st.button("🚀 Analyze Demand Now", use_container_width=False):

        state = {
            "sku": sku, "region": region,
            "month": _month, "dayofweek": _dow, "is_weekend": _is_weekend,
            "temperature": temperature,
            "promo_type": promo_type, "promo_discount_pct": promo_discount_pct,
            "event_type": event_type, "is_competitor_stockout": is_competitor_stockout,
            "sales_note": sales_note,
            "forecast": 0, "inventory": 0, "shortage": 0, "days_cover": 0,
            "risk": "", "reorder_qty": 0, "demand_boost": 0, "summary": "",
            # legacy fields kept for summary agent compatibility
            "temp": temperature, "promo": promo, "event": event,
        }

        with st.spinner("AI agents are working... this takes about 10 seconds."):
            result = graph.invoke(state)

        st.divider()

        # --- KPI CARDS ---
        st.markdown(f"<h3>📊 Results at a Glance</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{text_sec}; font-size:13px; margin-bottom:16px;'>Here's what the AI found for <b>{sku}</b> in <b>{region}</b>.</p>", unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        k1.metric("🔮 Predicted Demand Today", f"{round(result['forecast']):,} units", help="How many units the AI expects to sell today")
        k2.metric("🏭 Units Currently in Stock", f"{result['inventory']:,} units", help="What's sitting in the warehouse right now")
        k3.metric("📣 Demand Boost from Signals", f"+{result['demand_boost']}%", help="Extra demand detected from sales notes, events, or promotions")

        k4, k5, k6 = st.columns(3)
        k4.metric("📅 Days of Stock Left", f"{result['days_cover']} days", help="How many days current inventory will last at predicted demand")
        k5.metric("⚠️ Units Short", f"{round(result['shortage']):,}", help="Units you'll be missing if you don't reorder")
        k6.metric("🛒 Recommended Reorder", f"{result['reorder_qty']:,} units", help="How many units to order from the supplier")

        st.divider()

        # --- PROCUREMENT CARD ---
        if result["reorder_qty"] > 0:
            st.markdown(f"<h3>📋 Purchase Order</h3>", unsafe_allow_html=True)
            _sku_row = _master_df[_master_df["sku_id"] == sku].iloc[0]
            supplier_lead_days = int(_sku_row["lead_time"])
            order_date = pd.Timestamp.today().strftime("%d %b %Y")
            delivery_date = (pd.Timestamp.today() + pd.Timedelta(days=supplier_lead_days)).strftime("%d %b %Y")
            unit_cost = float(_sku_row["unit_cost"])
            total_cost = result["reorder_qty"] * unit_cost

            _po_html = (
                '<div style="border-radius:16px;padding:28px 32px;background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);max-width:540px;position:relative;overflow:hidden;">'
                '<div style="position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,#e94560,#f5a623,#00d2d3);"></div>'
                '<div style="font-size:11px;color:#a0aec0;margin-bottom:4px;letter-spacing:2px;font-weight:600;">PURCHASE ORDER</div>'
                f'<div style="font-size:24px;font-weight:800;color:#fff;margin-bottom:24px;">{sku} &nbsp;&middot;&nbsp; <span style="color:#00d2d3;">{region_label}</span></div>'
                f'<div style="display:inline-block;background:linear-gradient(90deg,#e94560,#c0392b);color:white;font-size:20px;font-weight:800;padding:10px 22px;border-radius:50px;margin-bottom:24px;">&#8377;{int(total_cost):,}</div>'
                '<div style="font-size:12px;color:#718096;margin-top:-18px;margin-bottom:20px;">Total estimated cost</div>'
                '<div style="border-top:1px solid #2d3748;margin-bottom:18px;"></div>'
                '<table style="width:100%;border-collapse:collapse;font-size:14px;">'
                f'<tr><td style="color:#a0aec0;padding:7px 0;">&#128230; Units to Order</td><td style="font-weight:700;color:#f5a623;text-align:right;">{result["reorder_qty"]:,} units</td></tr>'
                f'<tr><td style="color:#a0aec0;padding:7px 0;">&#128176; Cost per Unit</td><td style="font-weight:600;color:#e2e8f0;text-align:right;">&#8377;{unit_cost}</td></tr>'
                f'<tr><td style="color:#a0aec0;padding:7px 0;">&#128197; Order Date</td><td style="font-weight:600;color:#e2e8f0;text-align:right;">{order_date}</td></tr>'
                f'<tr><td style="color:#a0aec0;padding:7px 0;">&#128666; Arrives By</td><td style="font-weight:600;color:#00d2d3;text-align:right;">{delivery_date}</td></tr>'
                f'<tr><td style="color:#a0aec0;padding:7px 0;">&#9201; Supplier Lead Time</td><td style="font-weight:600;color:#e2e8f0;text-align:right;">{supplier_lead_days} days</td></tr>'
                '</table></div>'
            )
            st.markdown(_po_html, unsafe_allow_html=True)
        else:
            st.success("✅ Stock is sufficient — no purchase order needed right now.")

        st.divider()

        # --- RISK ---
        risk = result["risk"]
        st.markdown(f"<h3>🚨 Risk Level</h3>", unsafe_allow_html=True)

        risk_configs = {
            "CRITICAL": ("#e94560", "🔴", "CRITICAL — Act immediately. Stock will run out very soon."),
            "HIGH":     ("#f5a623", "🟠", "HIGH — Stock is running low. Order within 24–48 hours."),
            "MEDIUM":   ("#f0c040", "🟡", "MEDIUM — Stock is below comfortable levels. Plan a reorder soon."),
            "LOW":      ("#25d366", "🟢", "LOW — Stock levels are healthy. No action needed right now."),
        }
        rc = risk_configs.get(risk, risk_configs["LOW"])
        st.markdown(f"""
        <div style="background:{card_bg}; border:1px solid {border}; border-left:4px solid {rc[0]};
             border-radius:8px; padding:16px 20px; display:flex; align-items:center; gap:14px; max-width:600px;">
            <span style="font-size:28px;">{rc[1]}</span>
            <div>
                <div style="font-size:16px; font-weight:700; color:{rc[0]};">{rc[2].split('—')[0].strip()}</div>
                <div style="font-size:13px; color:{text_sec}; margin-top:2px;">{rc[2].split('—')[1].strip()}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if risk in ("CRITICAL", "HIGH"):
            alert_color = "#e94560" if risk == "CRITICAL" else "#f5a623"
            alert_label = "🚨 CRITICAL STOCKOUT ALERT" if risk == "CRITICAL" else "⚠️ HIGH RISK ALERT"
            alert_time = pd.Timestamp.now().strftime("%I:%M %p")
            st.markdown(f"""
            <div style="max-width:360px; margin-top:20px;">
                <div style="background:#1e1e1e; border-radius:16px; overflow:hidden;">
                    <div style="background:#075e54; padding:12px 18px; display:flex; align-items:center; gap:10px;">
                        <div style="background:#25d366; border-radius:50%; width:36px; height:36px; display:flex; align-items:center; justify-content:center; font-size:18px;">📦</div>
                        <div>
                            <div style="color:#fff; font-weight:700; font-size:14px;">Supply Chain Bot</div>
                            <div style="color:#a8d5a2; font-size:11px;">Automated Alert System</div>
                        </div>
                    </div>
                    <div style="padding:16px 18px; background:#111b21;">
                        <div style="background:#202c33; border-radius:12px; border-left:4px solid {alert_color}; padding:14px 16px;">
                            <div style="color:{alert_color}; font-weight:800; font-size:13px; margin-bottom:10px;">{alert_label}</div>
                            <div style="color:#e9edef; font-size:13px; line-height:1.7;">
                                <b>Product:</b> {sku}<br>
                                <b>City:</b> {region}<br>
                                <b>Stock lasts:</b> {result['days_cover']} days only<br>
                                <b>Units short:</b> {round(result['shortage']):,}<br>
                                <b>Action:</b> Reorder {result['reorder_qty']:,} units immediately
                            </div>
                            <div style="margin-top:12px; color:#25d366; font-size:12px; font-weight:600;">✅ Sent to Procurement Team</div>
                        </div>
                        <div style="text-align:right; color:#8696a0; font-size:11px; margin-top:8px;">{alert_time} ✓✓</div>
                    </div>
                </div>
                <div style="text-align:center; color:{text_sec}; font-size:11px; margin-top:6px;">WhatsApp Alert Preview</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # --- 7-DAY CHART ---
        st.markdown(f"<h3>📅 7-Day Stock Depletion Forecast</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{text_sec}; font-size:13px;'>The blue line shows how many units you expect to sell each day. The red line shows your remaining stock — when it hits zero, you're out.</p>", unsafe_allow_html=True)

        boost = float(result["demand_boost"])
        daily_forecasts = [
            round(forecast_tool(
                sku_id=sku, region=region, month=_month, dayofweek=d,
                is_weekend=int(d >= 5), temperature=temperature,
                promo_type=promo_type, promo_discount_pct=promo_discount_pct,
                event_type=event_type, is_competitor_stockout=is_competitor_stockout,
            ) * (1 + boost / 100))
            for d in range(7)
        ]

        projected_inventory = []
        running_stock = result["inventory"]
        stockout_day = None
        for i, demand in enumerate(daily_forecasts):
            running_stock = max(0, running_stock - demand)
            projected_inventory.append(running_stock)
            if running_stock == 0 and stockout_day is None:
                stockout_day = day_labels[i]

        forecast_chart_df = pd.DataFrame({"Day": day_labels, "Daily Sales Forecast": daily_forecasts, "Remaining Stock": projected_inventory})
        forecast_melted = forecast_chart_df.melt(id_vars="Day", var_name="Metric", value_name="Units")

        chart = alt.Chart(forecast_melted).mark_line(point=True).encode(
            x=alt.X("Day:N", sort=day_labels, title="Day of the Week"),
            y=alt.Y("Units:Q", title="Units"),
            color=alt.Color("Metric:N", scale=alt.Scale(range=[accent, "#e94560"])),
            tooltip=["Day", "Metric", "Units"]
        ).properties(height=300).configure_view(strokeWidth=0).configure_axis(
            gridColor=border, labelColor=text_sec, titleColor=text_sec
        ).configure_legend(labelColor=text_sec, titleColor=text_sec)

        st.altair_chart(chart, use_container_width=True)

        if stockout_day:
            st.error(f"⚠️ Your stock will run out by **{stockout_day}**. You need to reorder before then.")
        else:
            st.success(f"✅ You have enough stock for the full week. {projected_inventory[-1]:,} units remaining by Sunday.")

        st.divider()

        # --- AGENT TRAIL ---
        st.markdown(f"<h3>🤖 How the AI Reached This Decision</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{text_sec}; font-size:13px; margin-bottom:16px;'>Five AI agents ran in sequence — here's what each one found.</p>", unsafe_allow_html=True)

        trail_steps = [
            ("📣", "Sales Signal Agent", f"Detected a <b>+{result['demand_boost']}%</b> demand boost from your sales notes, promotions, and events."),
            ("🔮", "Demand Forecasting Agent", f"Predicted demand of <b>{round(result['forecast']):,} units</b> using the XGBoost machine learning model."),
            ("🏭", "Inventory Agent", f"Checked the warehouse and found <b>{result['inventory']:,} units</b> currently in stock."),
            ("⚠️",  "Risk Assessment Agent", f"Calculated <b>{result['days_cover']} days</b> of stock coverage — risk level: <b>{result['risk']}</b>."),
            ("🛒", "Procurement Agent", f"Recommended ordering <b>{result['reorder_qty']:,} units</b> to cover demand plus a 20% safety buffer."),
        ]

        for i, (icon, name, detail) in enumerate(trail_steps):
            st.markdown(f"""
            <div style="display:flex; gap:14px; margin-bottom:8px;">
                <div style="display:flex; flex-direction:column; align-items:center; gap:0;">
                    <div style="background:{accent}; color:white; border-radius:50%; width:32px; height:32px;
                         display:flex; align-items:center; justify-content:center; font-size:14px; flex-shrink:0;">{icon}</div>
                    {'<div style="width:2px; height:20px; background:' + border + '; margin: 0 auto;"></div>' if i < 4 else ''}
                </div>
                <div style="background:{card_bg}; border:1px solid {border}; border-radius:8px;
                     padding:12px 16px; flex:1; margin-bottom:{'8px' if i < 4 else '0'};">
                    <div style="font-size:12px; font-weight:600; color:{text_sec}; text-transform:uppercase; letter-spacing:0.5px;">{name}</div>
                    <div style="font-size:14px; color:{text_pri}; margin-top:4px;">{detail}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # --- EXECUTIVE SUMMARY ---
        st.markdown(f"<h3>📝 Executive Summary</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{text_sec}; font-size:13px; margin-bottom:12px;'>Plain-English recommendation written by the AI.</p>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{card_bg}; border:1px solid {border}; border-left:4px solid {accent};
             border-radius:8px; padding:20px 24px;">
            <div style="color:{text_pri}; font-size:15px; line-height:1.7;">{result['summary']}</div>
        </div>
        """, unsafe_allow_html=True)


# ==================================================
# PAGE: AI COPILOT
# ==================================================

elif page == "Copilot":

    st.markdown(f"<h1 style='margin-bottom:4px;'>🤖 AI Supply Chain Copilot</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{text_sec}; margin-bottom:20px;'>Ask any question about your inventory, reorder policies, or supply chain. The AI answers using your company's internal knowledge base.</p>", unsafe_allow_html=True)

    st.divider()

    st.markdown(f"""
    <div style="background:{card_bg}; border:1px solid {border}; border-radius:8px; padding:14px 18px; margin-bottom:20px;">
        <div style="font-size:12px; color:{text_sec}; margin-bottom:8px; font-weight:600;">Try asking:</div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
            {''.join([f'<span style="background:{hover}; color:{text_pri}; font-size:12px; padding:4px 12px; border-radius:50px; border:1px solid {border};">{q}</span>' for q in ["What is the reorder policy for Bangalore?", "Which SKU has the highest shelf life?", "What is the safety stock for Maggi?"]])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    question = st.text_input("Your question", placeholder="E.g. What is the reorder point for Coke500?")

    if st.button("Ask the Copilot →", key="copilot_btn"):
        if question.strip():
            try:
                with st.spinner("Searching knowledge base..."):
                    answer = ask_copilot(question)
                st.markdown(f"""
                <div style="background:{card_bg}; border:1px solid {border}; border-left:4px solid {accent};
                     border-radius:8px; padding:20px 24px; margin-top:16px;">
                    <div style="font-size:11px; color:{text_sec}; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.5px;">Answer</div>
                    <div style="color:{text_pri}; font-size:15px; line-height:1.7;">{answer}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Copilot error: {e}")
        else:
            st.warning("Please type a question first.")


# ==================================================
# PAGE: STOCK TRANSFERS
# ==================================================

elif page == "Transfer":

    st.markdown(f"<h1 style='margin-bottom:4px;'>🔁 Stock Transfer Recommendations</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{text_sec}; margin-bottom:20px;'>The AI scans all 20 products across 4 cities and tells you which warehouses have too much stock and where to move it before a stockout happens.</p>", unsafe_allow_html=True)

    inventory_df, transfer_df = compute_transfer_recommendations()

    total_skus = inventory_df["sku_id"].nunique()
    critical_count = len(inventory_df[inventory_df["days_cover"] < 5])
    transfer_count = len(transfer_df)

    st.divider()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Products Tracked", total_skus)
    k2.metric("Cities", 4)
    k3.metric("Locations Running Low", critical_count, delta="need action", delta_color="inverse")
    k4.metric("Transfers Recommended", transfer_count)

    st.divider()

    # --- HEATMAP ---
    st.markdown(f"<h3>📦 Stock Coverage Map</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{card_bg}; border:1px solid {border}; border-radius:8px; padding:12px 18px; margin-bottom:16px; font-size:13px; color:{text_sec}; line-height:1.8;">
        <b style="color:{text_pri};">How to read this:</b> Each coloured box = one product in one city.
        The number inside = how many days of stock remain.
        <span style="color:#e94560;"> 🔴 Red = under 5 days (urgent)</span> &nbsp;·&nbsp;
        <span style="color:#f5a623;"> 🟠 Orange = 5–10 days (watch)</span> &nbsp;·&nbsp;
        <span style="color:#25d366;"> 🟢 Green = 10+ days (healthy)</span>
    </div>
    """, unsafe_allow_html=True)

    categories = inventory_df["category"].unique().tolist()
    cat_tabs = st.tabs(categories)

    for cat_tab, category in zip(cat_tabs, categories):
        with cat_tab:
            cat_df = inventory_df[inventory_df["category"] == category].copy()
            cat_df["status"] = cat_df["days_cover"].apply(lambda d: "Critical" if d < 5 else ("Low" if d < 10 else "Healthy"))

            heatmap = alt.Chart(cat_df).mark_rect(stroke="#111", strokeWidth=2).encode(
                x=alt.X("region:N", title="City", axis=alt.Axis(labelFontSize=12, labelAngle=0)),
                y=alt.Y("product_name:N", title=None, axis=alt.Axis(labelFontSize=12)),
                color=alt.Color("days_cover:Q",
                    scale=alt.Scale(domain=[0, 5, 10, 20], range=["#e94560", "#f5a623", "#f5a623", "#25d366"]),
                    title="Days of Stock", legend=None),
                tooltip=[
                    alt.Tooltip("product_name:N", title="Product"),
                    alt.Tooltip("region:N", title="City"),
                    alt.Tooltip("stock:Q", title="Units in Stock", format=","),
                    alt.Tooltip("daily_demand:Q", title="Daily Sales Rate", format=".0f"),
                    alt.Tooltip("days_cover:Q", title="Days of Stock Left", format=".1f"),
                    alt.Tooltip("status:N", title="Status")
                ]
            ).properties(height=max(120, len(cat_df["product_name"].unique()) * 50))

            text_layer = alt.Chart(cat_df).mark_text(fontWeight="bold", fontSize=14).encode(
                x=alt.X("region:N"), y=alt.Y("product_name:N"),
                text=alt.Text("days_cover:Q", format=".0f"), color=alt.value("white")
            )

            st.altair_chart(heatmap + text_layer, use_container_width=True)

            critical_in_cat = cat_df[cat_df["days_cover"] < 5]
            if not critical_in_cat.empty:
                alerts = ", ".join([f"{r['product_name']} / {r['region']} ({r['days_cover']}d)" for _, r in critical_in_cat.iterrows()])
                st.error(f"🔴 Low stock: {alerts}")

    st.divider()

    # --- TRANSFERS ---
    st.markdown(f"<h3>🚚 Where to Move Stock</h3>", unsafe_allow_html=True)

    if transfer_df.empty:
        st.success("✅ All locations are well stocked — no transfers needed right now.")
    else:
        st.markdown(f"<p style='color:{text_sec}; font-size:13px; margin-bottom:12px;'>{len(transfer_df)} transfer(s) identified. Sorted by urgency — most critical at the top.</p>", unsafe_allow_html=True)

        transfer_df = transfer_df.sort_values("Days Left at Dest").reset_index(drop=True)

        filter_cats = ["All Categories"] + sorted(transfer_df["Category"].unique().tolist())
        selected_cat = st.selectbox("Filter by product category", filter_cats, key="transfer_filter")
        filtered = transfer_df if selected_cat == "All Categories" else transfer_df[transfer_df["Category"] == selected_cat]

        p_tab_data = [
            ("🔴 P1 · CRITICAL", "#e94560", "Dispatch today — less than 2 days of stock left"),
            ("🟠 P2 · URGENT",   "#f5a623", "Dispatch within 24 hours — 2 to 3 days of stock"),
            ("🟡 P3 · HIGH",     "#f0c040", "Plan for tomorrow — 3 to 4 days of stock"),
            ("🟢 P4 · MEDIUM",   "#25d366", "Schedule this week — 4 to 5 days of stock"),
        ]
        p_tab_labels = [
            f"🔴 Critical ({len(filtered[filtered['Priority'] == '🔴 P1 · CRITICAL'])})",
            f"🟠 Urgent ({len(filtered[filtered['Priority'] == '🟠 P2 · URGENT'])})",
            f"🟡 High ({len(filtered[filtered['Priority'] == '🟡 P3 · HIGH'])})",
            f"🟢 Medium ({len(filtered[filtered['Priority'] == '🟢 P4 · MEDIUM'])})",
        ]

        p_tabs = st.tabs(p_tab_labels)

        for p_tab, (priority_key, accent_c, description) in zip(p_tabs, p_tab_data):
            with p_tab:
                subset = filtered[filtered["Priority"] == priority_key]
                if subset.empty:
                    st.success(f"✅ No transfers at this priority level.")
                else:
                    st.caption(f"{description} · {len(subset)} transfer(s)")
                    for _, row in subset.iterrows():
                        col_a, col_b, col_c = st.columns([3, 1, 3])
                        with col_a:
                            st.markdown(f"""
                            <div style="background:#1a2e1a; border:1px solid #25d366; border-radius:12px; padding:14px 18px; margin-bottom:4px;">
                                <div style="color:#25d366; font-size:10px; font-weight:700; letter-spacing:1px;">TAKE FROM HERE</div>
                                <div style="color:#fff; font-size:17px; font-weight:800; margin:4px 0;">{row['From Region']}</div>
                                <div style="color:#a0aec0; font-size:12px;">{row['Product']}</div>
                                <div style="color:#a0aec0; font-size:12px;">Has {row['From Stock']:,} units · {row['From Coverage']} of stock left</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_b:
                            st.markdown(f"""
                            <div style="text-align:center; padding-top:22px;">
                                <div style="color:{accent_c}; font-size:26px;">→</div>
                                <div style="background:{accent_c}; color:#000; font-size:11px; font-weight:800; padding:4px 8px; border-radius:50px; margin-top:6px;">{row['Transfer Qty']:,} units</div>
                                <div style="color:#718096; font-size:10px; margin-top:4px;">{row['SKU ID']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_c:
                            st.markdown(f"""
                            <div style="background:#2e1a1a; border:1px solid {accent_c}; border-radius:12px; padding:14px 18px; margin-bottom:4px;">
                                <div style="color:{accent_c}; font-size:10px; font-weight:700; letter-spacing:1px;">SEND TO HERE · {row['Days Left at Dest']}d left</div>
                                <div style="color:#fff; font-size:17px; font-weight:800; margin:4px 0;">{row['To Region']}</div>
                                <div style="color:#a0aec0; font-size:12px;">{row['Product']}</div>
                                <div style="color:#a0aec0; font-size:12px;">Only {row['To Stock']:,} units · {row['To Coverage']} of stock left</div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

        st.divider()
        st.markdown(f"<h3>📋 Full Transfer List</h3>", unsafe_allow_html=True)
        st.dataframe(
            filtered[["Priority", "SKU ID", "Product", "Category", "From Region", "To Region", "Transfer Qty", "From Coverage", "To Coverage", "Days Left at Dest"]].rename(columns={"Days Left at Dest": "Days Left"}),
            use_container_width=True, hide_index=True
        )


# ==================================================
# PAGE: KNOWLEDGE BASE
# ==================================================

elif "Knowledge" in page:

    st.markdown(f"<h1 style='margin-bottom:4px;'>🧠 Enterprise Knowledge Base</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{text_sec}; margin-bottom:20px;'>This is the internal memory the AI Copilot uses to answer your questions — policies, notes, and operational rules your team has stored.</p>", unsafe_allow_html=True)

    st.divider()

    try:
        memory_df = pd.read_csv("data/enterprise_memory.csv")
        st.markdown(f"<p style='color:{text_sec}; font-size:13px;'>{len(memory_df)} records in the knowledge base.</p>", unsafe_allow_html=True)
        st.dataframe(memory_df, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load knowledge base: {e}")
