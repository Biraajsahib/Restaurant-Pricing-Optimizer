import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from packaging.version import Version

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Shared pricing rules (same file the notebooks use)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
import pricing_engine as engine  # noqa: E402

try:
    from google import genai
except Exception:  # the dashboard still works without the Gemini package
    genai = None

ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_demand_model.pkl")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
COMPRESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_restaurant_sales_data.csv.gz")
CSV_DATA_PATH = os.path.join(DATA_DIR, "processed_restaurant_sales_data.csv")
DATA_PATH = COMPRESSED_DATA_PATH if os.path.exists(COMPRESSED_DATA_PATH) else CSV_DATA_PATH
SHAP_PATH = os.path.join(DATA_DIR, "shap_business_insights.csv")
PERFORMANCE_PATH = os.path.join(DATA_DIR, "model_performance_summary.csv")

st.set_page_config(
    page_title="Restaurant Pricing Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv(ENV_PATH)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Streamlit Cloud uses Secrets instead of the local .env file.
if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        GEMINI_API_KEY = None

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
AI_AVAILABLE = bool(GEMINI_API_KEY) and genai is not None

# Full-width tables and buttons, written the way the installed Streamlit version expects
if Version(st.__version__) >= Version("1.50.0"):
    STRETCH = {"width": "stretch"}
else:
    STRETCH = {"use_container_width": True}


# ============================================================
# CUSTOM UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(49,51,63,.95), rgba(28,30,38,.98));
        border: 1px solid rgba(255,255,255,.10);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0 0 .35rem 0;
        font-size: 2.35rem;
        line-height: 1.15;
        color: #ffffff;
    }

    .hero p {
        margin: 0;
        color: rgba(255,255,255,.72);
        font-size: 1rem;
    }

    .status-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 0.8rem 0 1.25rem 0;
    }

    .status-card {
        min-height: 72px;
        padding: 13px 16px;
        border-radius: 12px;
        background: rgba(128,128,128,.07);
        border: 1px solid rgba(128,128,128,.20);
    }

    .status-label {
        font-size: .78rem;
        color: rgba(255,255,255,.58);
        text-transform: uppercase;
        letter-spacing: .05em;
    }

    .status-value {
        margin-top: 5px;
        font-size: 1rem;
        font-weight: 650;
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 1rem 0 1.4rem 0;
    }

    .kpi-card {
        min-height: 118px;
        padding: 18px 20px;
        border-radius: 15px;
        background: rgba(128,128,128,.065);
        border: 1px solid rgba(128,128,128,.22);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .kpi-label {
        color: rgba(255,255,255,.62);
        font-size: .86rem;
    }

    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        margin-top: 8px;
    }

    .decision-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin: 1rem 0;
    }

    .decision-card {
        min-height: 205px;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,.22);
        background: rgba(128,128,128,.055);
    }

    .decision-card.recommended {
        border: 1px solid rgba(52,211,153,.35);
        background: rgba(52,211,153,.065);
    }

    .decision-title {
        font-size: .82rem;
        text-transform: uppercase;
        letter-spacing: .06em;
        color: rgba(255,255,255,.58);
        margin-bottom: 9px;
    }

    .decision-main {
        font-size: 1.45rem;
        font-weight: 750;
        margin-bottom: 10px;
    }

    .decision-line {
        color: rgba(255,255,255,.78);
        margin: 5px 0;
    }

    .highlight {
        padding: 16px 18px;
        border-radius: 13px;
        background: rgba(52,211,153,.10);
        border: 1px solid rgba(52,211,153,.22);
        margin: 12px 0 18px 0;
        font-weight: 600;
    }

    @media (max-width: 900px) {
        .kpi-grid, .status-grid, .decision-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL / DATA
# ============================================================

if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
    st.error(
        "Model or processed data not found. Run the notebooks in order "
        "(02_Feature_Engineering, then 03_Modeling) to create them."
    )
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"])
    return data


@st.cache_data
def load_optional_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource
def load_ai_client(api_key):
    return genai.Client(api_key=api_key)


model = load_model()
df = load_data()
shap_insights = load_optional_csv(SHAP_PATH)
model_performance = load_optional_csv(PERFORMANCE_PATH)

promotion_options = engine.PROMOTION_OPTIONS
N_SCENARIOS = len(engine.PRICE_ADJUSTMENTS) * len(promotion_options)


def rupees(value):
    return f"-₹{abs(value):,.0f}" if value < 0 else f"₹{value:,.0f}"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <h1>🍽️ Restaurant Pricing Intelligence & Promotion Optimizer</h1>
        <p>Predict orders for any menu item, test {N_SCENARIOS} price &amp; offer strategies, find the most profitable one, and ask an AI advisor why.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="status-grid">
        <div class="status-card">
            <div class="status-label">Demand Model</div>
            <div class="status-value">✓ XGBoost · {len(model.feature_names_in_)} features</div>
        </div>
        <div class="status-card">
            <div class="status-label">Sales History</div>
            <div class="status-value">✓ {len(df):,} item-days · {df['Item_ID'].nunique()} items · {df['Outlet_ID'].nunique()} outlets</div>
        </div>
        <div class="status-card">
            <div class="status-label">Decision Engine</div>
            <div class="status-value">✓ {N_SCENARIOS} scenarios · {len(promotion_options)} offer types</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("📖 How to Use This Dashboard", expanded=False):
    st.markdown(
        """
        **1. Choose a menu item, outlet and date** from the sidebar. The date can be an upcoming day.

        **2. Set the menu price and the offer** you want to evaluate.

        **3. Review the Current Scenario** to see predicted orders, revenue, cost and profit.

        **4. Open Optimization** to see the highest-profit strategy across all price and offer combinations.

        **5. Use the AI Pricing Advisor** for natural-language what-if and strategy questions.

        **6. Use Business Insights** to understand the strongest demand drivers.

        *Optional:* open **Expected conditions** in the sidebar to test a rainy day, a busy day or a competitor price change.
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎛️ Scenario Controls")
st.sidebar.caption("Change these inputs to update the entire dashboard.")

item_table = df[["Item_ID", "Item_Name", "Category"]].drop_duplicates().sort_values("Item_ID")
item_labels = {
    row.Item_ID: f"{row.Item_Name} · {row.Category}"
    for row in item_table.itertuples()
}
selected_item = st.sidebar.selectbox(
    "Menu item",
    list(item_labels),
    format_func=lambda item_id: item_labels[item_id]
)

outlet_table = (
    df[["Outlet_ID", "City", "Outlet_Type", "Location_Type"]]
    .drop_duplicates()
    .sort_values("Outlet_ID")
)
outlet_labels = {
    row.Outlet_ID: f"{row.Outlet_ID} · {row.City} · {row.Outlet_Type}"
    for row in outlet_table.itertuples()
}
selected_outlet = st.sidebar.selectbox(
    "Outlet",
    list(outlet_labels),
    format_func=lambda outlet_id: outlet_labels[outlet_id]
)

data_start = df["Date"].min().date()
data_end = df["Date"].max().date()
horizon_end = pd.Timestamp(engine.PLANNING_HORIZON_END).date()
first_planning_day = (pd.Timestamp(data_end) + pd.Timedelta(days=1)).date()
default_date = min(max(pd.Timestamp.today().date(), first_planning_day), horizon_end)

selected_date = pd.Timestamp(
    st.sidebar.date_input(
        "Date",
        value=default_date,
        min_value=data_start,
        max_value=horizon_end
    )
)

item_history = df[
    (df["Item_ID"] == selected_item)
    & (df["Outlet_ID"] == selected_outlet)
]

if item_history.empty:
    st.error("No sales history exists for this item-outlet combination.")
    st.stop()

# Situation on the chosen date, with conditions filled in from history
default_context = engine.build_context(item_history, selected_date)
base_price = float(default_context["Base_Price"])
price_step = 1.0 if base_price < 100 else 5.0

st.sidebar.caption(
    f"{default_context['Day_of_Week']} · {default_context['Festival']} · {default_context['Season']}"
)

st.sidebar.subheader("💰 Pricing & Offer")

menu_price = st.sidebar.number_input(
    "Menu price (₹)",
    min_value=float(np.round(base_price * 0.80)),
    max_value=float(np.round(base_price * 1.20)),
    value=base_price,
    step=price_step,
    key=f"menu_price_{selected_item}_{selected_outlet}",
    help=f"Base menu price is ₹{base_price:,.0f}. The offer is applied on top of this price."
)

selected_promotion = st.sidebar.selectbox(
    "Offer",
    promotion_options
)

with st.sidebar.expander("🌦️ Expected conditions (auto-filled)"):
    st.caption(
        "Filled from history: same weekday / festival for footfall, same month for weather, "
        "last four weeks for the competitor."
    )
    condition_key = f"{selected_item}_{selected_outlet}_{selected_date.date()}"
    defaults = {
        "Footfall_Index": round(default_context["Footfall_Index"], 2),
        "Temperature": round(default_context["Temperature"], 1),
        "Rainfall": round(default_context["Rainfall"], 1),
        "Competitor_Price": float(np.round(default_context["Competitor_Price"])),
    }
    footfall_input = st.number_input(
        "Footfall index (1.0 = normal day)", min_value=0.30, max_value=2.50,
        value=defaults["Footfall_Index"], step=0.05, key=f"footfall_{condition_key}"
    )
    temperature_input = st.number_input(
        "Temperature (°C)", min_value=0.0, max_value=48.0,
        value=defaults["Temperature"], step=0.5, key=f"temperature_{condition_key}"
    )
    rainfall_input = st.number_input(
        "Rainfall (mm)", min_value=0.0, max_value=150.0,
        value=defaults["Rainfall"], step=1.0, key=f"rainfall_{condition_key}"
    )
    competitor_input = st.number_input(
        "Competitor price (₹)", min_value=1.0,
        value=defaults["Competitor_Price"], step=price_step, key=f"competitor_{condition_key}"
    )

# Only values the user actually changed replace the history-based defaults
overrides = {}
for name, value in [
    ("Footfall_Index", footfall_input),
    ("Temperature", temperature_input),
    ("Rainfall", rainfall_input),
    ("Competitor_Price", competitor_input),
]:
    if abs(value - defaults[name]) > 1e-9:
        overrides[name] = float(value)

context = engine.build_context(item_history, selected_date, overrides)


# ============================================================
# CURRENT SCENARIO CALCULATION
# ============================================================

current = engine.evaluate_scenario(model, context, menu_price, selected_promotion)

predicted_demand = float(current["Predicted_Units"])
customer_pays = float(current["Selling_Price"])
revenue = float(current["Revenue"])
total_cost = float(current["Total_Cost"])
expected_profit = float(current["Expected_Profit"])

price_in_range = engine.price_in_trained_range(menu_price, base_price)


# ============================================================
# OPTIMIZATION
# ============================================================

optimization_results = engine.optimize(model, context)
valid_results = optimization_results[optimization_results["Valid"]]
best_strategy = optimization_results.iloc[0]
promotion_comparison = engine.best_by_promotion(optimization_results)

recommended_profit = float(best_strategy["Expected_Profit"])
profit_uplift = recommended_profit - expected_profit
profit_uplift_pct = (
    profit_uplift / abs(expected_profit) * 100
    if expected_profit != 0 else np.nan
)
current_margin = expected_profit / revenue * 100 if revenue else 0
recommended_margin = (
    recommended_profit / float(best_strategy["Revenue"]) * 100
    if float(best_strategy["Revenue"]) else 0
)
n_valid = int(optimization_results["Valid"].sum())
n_excluded = N_SCENARIOS - n_valid


def describe_strategy(row):
    change = float(row["Price_Adjustment"])
    price_note = "base menu price" if change == 0 else f"menu price {change:+.0%}"
    return f"{row['Promotion_Type']} · {price_note}"


# ============================================================
# TABBED DASHBOARD
# ============================================================

tab_overview, tab_optimizer, tab_ai, tab_insights = st.tabs([
    "📊 Overview",
    "🎯 Optimization",
    "🤖 AI Pricing Advisor",
    "🔍 Business Insights"
])


# ============================================================
# OVERVIEW TAB
# ============================================================

with tab_overview:
    st.header("Current Scenario")
    st.caption(
        f"{context['Item_Name']} · {outlet_labels[selected_outlet]} · "
        f"{selected_date.strftime('%A, %d %b %Y')} · {context['Festival']} · "
        f"{selected_promotion}"
    )

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Predicted Orders</div><div class="kpi-value">{predicted_demand:,.1f} units</div></div>
            <div class="kpi-card"><div class="kpi-label">Revenue</div><div class="kpi-value">{rupees(revenue)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Total Cost</div><div class="kpi-value">{rupees(total_cost)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Expected Profit</div><div class="kpi-value">{rupees(expected_profit)}</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not price_in_range:
        st.warning(
            f"₹{menu_price:,.0f} is more than 10% away from the base price (₹{base_price:,.0f}). "
            "The model has only seen menu prices within ±10% of base, so this prediction is an extrapolation."
        )

    if overrides:
        st.info("Custom conditions in use: " + ", ".join(f"{k.replace('_', ' ')} = {v:g}" for k, v in overrides.items()))

    with st.expander("How this profit is calculated"):
        breakdown = pd.DataFrame({
            "Line": [
                "Menu price", "Customer pays (after offer)", "Predicted orders",
                "Revenue", "Food cost", "Offer cost", "Overhead (fixed)", "Expected profit"
            ],
            "Value": [
                f"₹{menu_price:,.0f}", f"₹{customer_pays:,.0f}", f"{predicted_demand:,.1f} units",
                rupees(revenue), rupees(float(current["Food_Cost"])), rupees(float(current["Promotion_Cost"])),
                rupees(float(current["Overhead_Cost"])), rupees(expected_profit)
            ]
        })
        st.dataframe(breakdown, **STRETCH, hide_index=True)
        st.caption(
            f"Food cost per plate ₹{context['Unit_Cost']:,.2f}. Expected footfall index {context['Footfall_Index']:.2f}, "
            f"{context['Temperature']:.0f}°C, {context['Rainfall']:.1f} mm rain, competitor price ₹{context['Competitor_Price']:,.0f}."
        )

    st.divider()
    st.subheader("🏆 Recommended Strategy")

    st.markdown(
        f"""
        <div class="decision-grid">
            <div class="decision-card">
                <div class="decision-title">Current Scenario</div>
                <div class="decision-main">₹{menu_price:,.0f} · {selected_promotion}</div>
                <div class="decision-line">Customer pays: <b>₹{customer_pays:,.0f}</b></div>
                <div class="decision-line">Orders: <b>{predicted_demand:,.1f}</b> units</div>
                <div class="decision-line">Revenue: <b>{rupees(revenue)}</b></div>
                <div class="decision-line">Expected profit: <b>{rupees(expected_profit)}</b></div>
            </div>
            <div class="decision-card recommended">
                <div class="decision-title">Optimizer Recommendation</div>
                <div class="decision-main">₹{float(best_strategy['Menu_Price']):,.0f} · {best_strategy['Promotion_Type']}</div>
                <div class="decision-line">Customer pays: <b>₹{float(best_strategy['Selling_Price']):,.0f}</b></div>
                <div class="decision-line">Orders: <b>{float(best_strategy['Predicted_Units']):,.1f}</b> units</div>
                <div class="decision-line">Revenue: <b>{rupees(float(best_strategy['Revenue']))}</b></div>
                <div class="decision-line">Expected profit: <b>{rupees(recommended_profit)}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if profit_uplift > 0.5:
        st.markdown(
            f'<div class="highlight">💡 The optimizer identifies a profit opportunity of <b>{rupees(profit_uplift)}</b> ({profit_uplift_pct:.1f}%) versus the current scenario.</div>',
            unsafe_allow_html=True
        )
    elif profit_uplift < -0.5:
        st.info(
            "The current scenario has higher expected profit than the optimizer recommendation. "
            "That happens when it sits outside the strategies the optimizer is allowed to test "
            "(for example a price beyond ±10%, or an offer on a raised menu price)."
        )
    else:
        st.info("The current scenario is already the optimizer's recommendation.")

    st.subheader("Profitability Snapshot")
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Profit Margin", f"{current_margin:.1f}%")
    m2.metric("Recommended Profit Margin", f"{recommended_margin:.1f}%", f"{recommended_margin - current_margin:+.1f} pp")
    m3.metric(
        "Expected Profit Uplift",
        rupees(profit_uplift),
        f"{profit_uplift_pct:+.1f}%" if np.isfinite(profit_uplift_pct) else None
    )


# ============================================================
# OPTIMIZATION TAB
# ============================================================

with tab_optimizer:
    st.header("Price & Offer Optimization")
    st.caption(
        f"{N_SCENARIOS} candidate strategies = {len(engine.PRICE_ADJUSTMENTS)} menu-price levels × {len(promotion_options)} offers. "
        f"{n_excluded} are excluded by the no-fake-discount rule (an offer is never applied to a raised menu price), "
        f"leaving {n_valid}. The objective is to maximize expected profit."
    )

    st.subheader("Best Strategy by Offer")

    display_comparison = promotion_comparison.round(2).copy()
    display_comparison.index.name = "Offer"
    st.dataframe(display_comparison, **STRETCH, hide_index=False)

    st.subheader("Expected Profit by Offer")
    chart_data = promotion_comparison[["Expected_Profit"]].copy()
    st.bar_chart(chart_data)

    st.subheader("Expected Profit: Menu Price Level × Offer")
    profit_grid = valid_results.pivot(
        index="Promotion_Type",
        columns="Price_Adjustment",
        values="Expected_Profit"
    ).reindex(promotion_options)
    profit_grid.columns = [f"{c:+.0%}" for c in profit_grid.columns]
    profit_grid.index.name = "Offer"
    st.dataframe(profit_grid.round(0), **STRETCH)
    st.caption("Columns are the menu price versus the base price. Empty cells are excluded by the no-fake-discount rule.")

    with st.expander(f"🔎 Explore all {N_SCENARIOS} optimization scenarios"):
        st.dataframe(
            optimization_results.round(2),
            **STRETCH,
            hide_index=True
        )

        csv_data = optimization_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"⬇️ Download all {N_SCENARIOS} scenarios",
            data=csv_data,
            file_name="promotion_optimization_results.csv",
            mime="text/csv"
        )


# ============================================================
# AI FUNCTIONS
# ============================================================

def scenario_facts_from(row, price):
    return {
        "menu_price": float(price),
        "customer_pays": float(row["Selling_Price"]),
        "promotion": row["Promotion_Type"],
        "predicted_units": float(row["Predicted_Units"]),
        "revenue": float(row["Revenue"]),
        "total_cost": float(row["Total_Cost"]),
        "profit": float(row["Expected_Profit"])
    }


def detect_ai_intent(query):
    q = query.lower()

    driver_words = [
        "driver", "drivers", "shap", "demand driver",
        "what drives demand", "why is demand"
    ]
    promotion_words = [
        "promotion", "promotions", "discount", "bogo", "offer", "offers",
        "buy one get one", "combo"
    ]
    comparison_words = [
        "compare", "comparison", "versus", "vs", "better",
        "which", "best", "should i use", "recommend",
        "recommended", "recommendation"
    ]
    why_words = ["why", "reason", "explain"]

    percentage_match = re.search(r"(\d+(?:\.\d+)?)\s*%", q)

    if percentage_match:
        if any(x in q for x in ["increase", "raise", "hike", "higher"]):
            return "price_increase"
        if any(x in q for x in ["decrease", "reduce", "lower", "cut"]):
            return "price_decrease"

    if any(x in q for x in driver_words):
        return "demand_drivers"

    if any(x in q for x in promotion_words):
        return "promotion_comparison"

    if any(x in q for x in comparison_words) or any(x in q for x in why_words):
        return "strategy_comparison"

    return "general_business"


def detect_promotion_names(query):
    q = query.lower()
    aliases = {
        "No Promotion": ["no promotion", "no promo", "no offer"],
        "5% Discount": ["5% discount", "5 percent discount", "5% off", "5 percent off"],
        "10% Discount": ["10% discount", "10 percent discount", "10% off", "10 percent off"],
        "15% Discount": ["15% discount", "15 percent discount", "15% off", "15 percent off"],
        "20% Discount": ["20% discount", "20 percent discount", "20% off", "20 percent off"],
        "Combo Offer": ["combo"],
        "Buy One Get One": ["bogo", "buy one get one", "buy 1 get 1", "1+1"]
    }
    found = [p for p, terms in aliases.items() if any(t in q for t in terms)]
    # "15% discount" also contains "5% discount": keep the longer match only
    if "15% Discount" in found and "5% Discount" in found and not re.search(r"(?<!1)5\s*(%|percent)", q):
        found.remove("5% Discount")
    return found


def promotion_facts(promotions):
    facts = {}
    for promotion in promotions:
        if promotion in promotion_comparison.index:
            r = promotion_comparison.loc[promotion]
            facts[promotion] = {
                "best_menu_price": float(r["Best_Menu_Price"]),
                "customer_pays": float(r["Customer_Pays"]),
                "predicted_units": float(r["Predicted_Units"]),
                "revenue": float(r["Revenue"]),
                "promotion_cost": float(r["Promotion_Cost"]),
                "total_cost": float(r["Total_Cost"]),
                "expected_profit": float(r["Expected_Profit"])
            }
    return facts


def build_ai_context(intent, current_facts, scenario_facts=None, selected_promotions=None):
    ai_context = {
        "intent": intent,
        "situation": {
            "menu_item": context["Item_Name"],
            "category": context["Category"],
            "outlet": outlet_labels[selected_outlet],
            "date": selected_date.strftime("%A %d %b %Y"),
            "festival": context["Festival"],
            "base_menu_price": base_price,
            "food_cost_per_plate": float(context["Unit_Cost"]),
            "competitor_price": float(context["Competitor_Price"]),
            "expected_footfall_index": float(context["Footfall_Index"])
        },
        "current_scenario": current_facts,
        "top_strategy": {
            "promotion": best_strategy["Promotion_Type"],
            "menu_price": float(best_strategy["Menu_Price"]),
            "customer_pays": float(best_strategy["Selling_Price"]),
            "predicted_units": float(best_strategy["Predicted_Units"]),
            "revenue": float(best_strategy["Revenue"]),
            "promotion_cost": float(best_strategy["Promotion_Cost"]),
            "total_cost": float(best_strategy["Total_Cost"]),
            "expected_profit": float(best_strategy["Expected_Profit"])
        },
        "optimizer_rule": "Offers are never applied to a raised menu price (no fake discounts)."
    }

    if intent in ["promotion_comparison", "strategy_comparison"]:
        ai_context["all_promotion_results"] = promotion_facts(promotion_options)

    if selected_promotions:
        ai_context["selected_promotion_results"] = promotion_facts(selected_promotions)

    if scenario_facts is not None:
        ai_context["price_scenario"] = scenario_facts

    if intent == "demand_drivers" and not shap_insights.empty:
        ai_context["top_demand_drivers"] = shap_insights.head(10).to_dict("records")

    return ai_context


def explain_with_gemini(user_query, factual_context):
    prompt = f"""
You are a restaurant pricing, promotion and revenue-management advisor.

Answer the user's business question using ONLY the factual context below.

RULES:
1. Never invent numerical values.
2. Never perform independent demand forecasting.
3. Demand estimates come from the XGBoost model.
4. Profit, revenue and cost figures are calculated by Python from the scenario assumptions.
5. SHAP indicates predictive influence, not causality.
6. Do not claim that price increases or discounts always increase/decrease demand.
7. Do not introduce external market statistics.
8. If the provided facts do not support a conclusion, say so.
9. Use ₹ for monetary values.
10. Answer directly and keep the response executive-friendly.
11. If "outside_trained_price_range" is true, say the prediction is an extrapolation and less reliable.

USER QUESTION:
{user_query}

FACTUAL CONTEXT:
{factual_context}

For price scenarios, explain the change in price, predicted orders, revenue and profit, then give a business interpretation.
For promotion questions, compare the relevant offer results and prioritize expected profit while discussing orders, revenue and promotion cost.
For recommendation questions, explain why the highest-profit strategy ranks first using the supplied evidence.
For demand-driver questions, explain the supplied SHAP drivers without implying causality.
"""

    response = load_ai_client(GEMINI_API_KEY).models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text


def built_in_summary(intent, current_facts, scenario_facts, mentioned_promotions):
    """Plain summary of the numbers, used when Gemini is not available."""
    lines = []

    if scenario_facts is not None:
        direction = "raises" if scenario_facts["profit"] > current_facts["profit"] else "lowers"
        lines.append(
            f"Moving the menu price to **₹{scenario_facts['menu_price']:,.0f}** "
            f"({scenario_facts['price_change_percent']:+.1f}%) changes predicted orders from "
            f"{current_facts['predicted_units']:,.1f} to **{scenario_facts['predicted_units']:,.1f}** "
            f"({scenario_facts['demand_change_percent']:+.1f}%)."
        )
        lines.append(
            f"Revenue moves {scenario_facts['revenue_change_percent']:+.1f}% and expected profit "
            f"{scenario_facts['profit_change_percent']:+.1f}%, to **{rupees(scenario_facts['profit'])}**. "
            f"The change **{direction} profit**."
        )
        if scenario_facts.get("outside_trained_price_range"):
            lines.append("This price is outside the ±10% range the model was trained on, so treat it as a rough estimate.")

    elif intent == "demand_drivers" and not shap_insights.empty:
        lines.append("The strongest demand drivers in the model are:")
        for _, r in shap_insights.head(5).iterrows():
            lines.append(f"- **{r['Feature']}** — {r['Business_Interpretation']}")

    else:
        lines.append(
            f"The most profitable strategy for {context['Item_Name']} on {selected_date.strftime('%A %d %b')} is "
            f"**{best_strategy['Promotion_Type']} at a ₹{float(best_strategy['Menu_Price']):,.0f} menu price** "
            f"(customer pays ₹{float(best_strategy['Selling_Price']):,.0f}): "
            f"{float(best_strategy['Predicted_Units']):,.1f} orders and **{rupees(recommended_profit)}** expected profit."
        )
        ranked = promotion_comparison["Expected_Profit"]
        if len(ranked) > 1:
            lines.append(
                f"Next best is {ranked.index[1]} at {rupees(ranked.iloc[1])}; "
                f"the weakest is {ranked.index[-1]} at {rupees(ranked.iloc[-1])}."
            )
        for promotion in mentioned_promotions or []:
            if promotion in promotion_comparison.index and promotion != best_strategy["Promotion_Type"]:
                r = promotion_comparison.loc[promotion]
                lines.append(
                    f"{promotion}: {float(r['Predicted_Units']):,.1f} orders, {rupees(float(r['Expected_Profit']))} profit "
                    f"({rupees(float(r['Expected_Profit']) - recommended_profit)} versus the best strategy)."
                )
        lines.append(
            f"Your current scenario earns {rupees(current_facts['profit'])}, so the gap to the best strategy is "
            f"**{rupees(recommended_profit - current_facts['profit'])}**."
        )

    return "\n\n".join(lines)


# ============================================================
# AI PRICING ADVISOR TAB
# ============================================================

with tab_ai:
    st.header("🤖 AI Pricing Advisor")
    st.caption(
        "Ask a business question. Python and XGBoost generate the numerical evidence; Gemini turns that evidence into an executive explanation."
    )

    if not AI_AVAILABLE:
        st.info(
            "Gemini is not connected, so answers use the built-in summary of the model results. "
            "Add GEMINI_API_KEY to the .env file (or Streamlit Cloud Secrets) to switch on the AI narrative."
        )

    if "ai_query" not in st.session_state:
        st.session_state.ai_query = ""

    st.markdown("**Quick questions**")
    q1, q2, q3, q4 = st.columns(4)

    if q1.button("📈 +10% price", **STRETCH):
        st.session_state.ai_query = "What will happen if I increase the price by 10%?"

    if q2.button("💸 10% discount", **STRETCH):
        st.session_state.ai_query = "Is a 10% discount profitable?"

    if q3.button("🏆 Best strategy", **STRETCH):
        st.session_state.ai_query = "Which offer is best?"

    if q4.button("💡 Why recommended?", **STRETCH):
        st.session_state.ai_query = "Why is the recommended strategy best?"

    user_query = st.text_input(
        "Ask your question",
        key="ai_query",
        placeholder="e.g. What happens if I increase the price by 10%?"
    )

    ask_ai = st.button("✨ Analyze with AI", type="primary", **STRETCH)

    if ask_ai:
        if not user_query.strip():
            st.warning("Please enter a question first.")
        else:
            intent = detect_ai_intent(user_query)
            mentioned_promotions = detect_promotion_names(user_query)

            current_facts = scenario_facts_from(current, menu_price)

            scenario_facts = None
            percentage_match = re.search(r"(\d+(?:\.\d+)?)\s*%", user_query.lower())

            if percentage_match and intent in ["price_increase", "price_decrease"]:
                percentage = float(percentage_match.group(1))
                if intent == "price_increase":
                    scenario_price = menu_price * (1 + percentage / 100)
                else:
                    scenario_price = menu_price * (1 - percentage / 100)
                scenario_price = float(np.round(scenario_price))

                scenario_row = engine.evaluate_scenario(model, context, scenario_price, selected_promotion)
                scenario_facts = scenario_facts_from(scenario_row, scenario_price)

                scenario_facts["price_change_percent"] = (
                    (scenario_price - menu_price) / menu_price * 100
                )
                scenario_facts["demand_change_percent"] = (
                    (scenario_facts["predicted_units"] - predicted_demand)
                    / predicted_demand * 100 if predicted_demand else 0.0
                )
                scenario_facts["revenue_change_percent"] = (
                    (scenario_facts["revenue"] - revenue) / revenue * 100 if revenue else 0.0
                )
                scenario_facts["profit_change_percent"] = (
                    (scenario_facts["profit"] - expected_profit)
                    / abs(expected_profit) * 100 if expected_profit else 0.0
                )
                scenario_facts["outside_trained_price_range"] = not engine.price_in_trained_range(
                    scenario_price, base_price
                )

                st.subheader("Model-Based Scenario Analysis")
                a, b, c, d = st.columns(4)
                a.metric("Scenario Menu Price", f"₹{scenario_price:,.0f}", f"{scenario_facts['price_change_percent']:+.1f}%")
                b.metric("Predicted Orders", f"{scenario_facts['predicted_units']:,.1f}", f"{scenario_facts['demand_change_percent']:+.1f}%")
                c.metric("Revenue", rupees(scenario_facts["revenue"]), f"{scenario_facts['revenue_change_percent']:+.1f}%")
                d.metric("Expected Profit", rupees(scenario_facts["profit"]), f"{scenario_facts['profit_change_percent']:+.1f}%")

                if scenario_facts["outside_trained_price_range"]:
                    st.warning(
                        "This price is more than 10% away from the base price. The model has not seen prices "
                        "out there, so the result is an extrapolation."
                    )

            elif intent == "promotion_comparison" and mentioned_promotions:
                st.subheader("Relevant Offer Evidence")
                relevant = promotion_comparison.loc[
                    promotion_comparison.index.isin(mentioned_promotions)
                ].round(2)
                st.dataframe(relevant, **STRETCH)

            elif intent in ["strategy_comparison", "promotion_comparison"]:
                st.subheader("Optimization Evidence")
                st.dataframe(promotion_comparison.round(2), **STRETCH)

            elif intent == "demand_drivers" and not shap_insights.empty:
                st.subheader("Top Demand Drivers")
                st.dataframe(shap_insights.head(10), **STRETCH, hide_index=True)

            factual_context = build_ai_context(
                intent,
                current_facts,
                scenario_facts,
                mentioned_promotions
            )
            fallback = built_in_summary(intent, current_facts, scenario_facts, mentioned_promotions)

            st.subheader("AI Pricing Advisor")
            if AI_AVAILABLE:
                with st.spinner("Gemini is analyzing the model evidence..."):
                    try:
                        st.markdown(explain_with_gemini(user_query, factual_context))
                    except Exception as e:
                        st.warning(f"Gemini request failed ({e}). Showing the built-in summary instead.")
                        st.markdown(fallback)
            else:
                st.markdown(fallback)

    st.divider()
    st.markdown(
        "**Examples:** `What happens if I increase the price by 10%?` · `Why is this strategy recommended?` · `Is BOGO profitable?` · `Compare combo and 10% discount.` · `What drives demand?`"
    )


# ============================================================
# BUSINESS INSIGHTS TAB
# ============================================================

with tab_insights:
    st.header("🔍 Business Insights")
    st.caption("Interpretability and model-quality evidence supporting the pricing decisions.")

    if not shap_insights.empty:
        st.subheader("Top Demand Drivers")
        shap_display = shap_insights.head(10).copy()
        st.dataframe(
            shap_display,
            **STRETCH,
            hide_index=True
        )

        if "Feature" in shap_display.columns and "Mean_Abs_SHAP" in shap_display.columns:
            st.subheader("Relative Feature Influence")
            shap_chart = (
                shap_display[["Feature", "Mean_Abs_SHAP"]]
                .set_index("Feature")
                .sort_values("Mean_Abs_SHAP")
            )
            st.bar_chart(shap_chart)

        st.info(
            "SHAP values indicate how strongly features influence model predictions; they should not be interpreted as causal effects."
        )
    else:
        st.warning("SHAP insights file was not found.")

    if not model_performance.empty:
        st.subheader("Model Performance")
        st.dataframe(
            model_performance.round(3),
            **STRETCH,
            hide_index=True
        )

        best_model = model_performance.sort_values("R2", ascending=False).iloc[0]
        st.success(
            f"{best_model['Model']} has the highest R² of {best_model['R2']:.3f} among the evaluated models "
            f"(average error {best_model['MAE']:.1f} units per item per day)."
        )

    st.subheader("Decision Logic")
    st.markdown(
        f"""
        **1. XGBoost** predicts orders for the selected item, outlet, date, price and offer.

        **2. Python economics** converts predicted orders into revenue, food cost, offer cost and expected profit.

        **3. Optimization** evaluates {N_SCENARIOS} price/offer combinations, removes fake discounts, and ranks the rest by expected profit.

        **4. SHAP** explains which features are most influential in demand prediction.

        **5. Gemini** explains the numerical evidence in natural language without generating the underlying numbers.
        """
    )


# ============================================================
# FOOTER / MODEL INFO
# ============================================================

st.divider()

footer_col1, footer_col2 = st.columns(2)

with footer_col1:
    st.caption(
        "Decision-support application · Demand prediction + pricing optimization + GenAI explanation"
    )

with footer_col2:
    st.caption(
        f"Model: XGBoost pipeline · {len(model.feature_names_in_)} input features · {len(df):,} historical rows · "
        f"sales history {data_start} to {data_end}"
    )
