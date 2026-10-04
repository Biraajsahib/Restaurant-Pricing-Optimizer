"""
Pricing engine
==============

One shared place for the rules that both the notebooks and the Streamlit app
use: the offer definitions, the festival calendar, the feature engineering and
the scenario / profit calculations.

Keeping this in a single file means the app builds its model inputs with
exactly the same code that built the training data, so the two can never
drift apart.
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Offers
# --------------------------------------------------------------------------

PROMOTION_OPTIONS = [
    "No Promotion",
    "5% Discount",
    "10% Discount",
    "15% Discount",
    "20% Discount",
    "Combo Offer",
    "Buy One Get One",
]

# discount     : advertised % off the menu price
# price_factor : what the customer pays, as a share of the menu price
# depth        : value handed to the customer, as a share of the menu price
PROMOTION_PARAMETERS = {
    "No Promotion":    {"discount": 0.0,  "price_factor": 1.00, "depth": 0.00, "flag": 0},
    "5% Discount":     {"discount": 5.0,  "price_factor": 0.95, "depth": 0.05, "flag": 1},
    "10% Discount":    {"discount": 10.0, "price_factor": 0.90, "depth": 0.10, "flag": 1},
    "15% Discount":    {"discount": 15.0, "price_factor": 0.85, "depth": 0.15, "flag": 1},
    "20% Discount":    {"discount": 20.0, "price_factor": 0.80, "depth": 0.20, "flag": 1},
    # item + add-on at a bundled price: item is 10% cheaper, restaurant absorbs part of the add-on
    "Combo Offer":     {"discount": 0.0,  "price_factor": 0.90, "depth": 0.10, "flag": 1},
    # customer pays for one plate and gets two
    "Buy One Get One": {"discount": 0.0,  "price_factor": 1.00, "depth": 0.50, "flag": 1},
}

# Combo: cost of the add-on the restaurant absorbs, as a share of the item's base price
COMBO_ADDON_COST_RATE = 0.06

# Menu-price levels the optimiser tests, relative to the item's base price.
# These are the same levels that appear in the sales history, so the model is
# never asked about a price it has not seen.
PRICE_ADJUSTMENTS = [-0.10, -0.05, 0.00, 0.05, 0.10]
TRAINED_PRICE_RANGE = (0.90, 1.10)

# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------

# Festival day plus the surrounding days people actually eat out
FESTIVAL_CALENDAR = {
    "2024-10-11": "Dussehra", "2024-10-12": "Dussehra",
    "2024-10-29": "Diwali", "2024-10-30": "Diwali", "2024-10-31": "Diwali",
    "2024-11-01": "Diwali", "2024-11-02": "Diwali",
    "2024-12-24": "Christmas", "2024-12-25": "Christmas",
    "2024-12-31": "New Year", "2025-01-01": "New Year",
    "2025-01-26": "Republic Day",
    "2025-02-14": "Valentines Day",
    "2025-03-13": "Holi", "2025-03-14": "Holi",
    "2025-03-31": "Eid", "2025-04-01": "Eid",
    "2025-08-09": "Raksha Bandhan",
    "2025-08-15": "Independence Day",
    "2025-10-01": "Dussehra", "2025-10-02": "Dussehra",
    "2025-10-18": "Diwali", "2025-10-19": "Diwali", "2025-10-20": "Diwali",
    "2025-10-21": "Diwali", "2025-10-22": "Diwali",
    "2025-12-24": "Christmas", "2025-12-25": "Christmas",
    "2025-12-31": "New Year", "2026-01-01": "New Year",
    "2026-01-26": "Republic Day",
    "2026-02-14": "Valentines Day",
    "2026-03-03": "Holi", "2026-03-04": "Holi",
    "2026-03-21": "Eid", "2026-03-22": "Eid",
    "2026-08-15": "Independence Day",
    "2026-08-28": "Raksha Bandhan",
    # after the sales history ends - lets the app plan upcoming dates
    "2026-10-19": "Dussehra", "2026-10-20": "Dussehra",
    "2026-11-06": "Diwali", "2026-11-07": "Diwali", "2026-11-08": "Diwali",
    "2026-11-09": "Diwali", "2026-11-10": "Diwali",
    "2026-12-24": "Christmas", "2026-12-25": "Christmas",
    "2026-12-31": "New Year",
}
PLANNING_HORIZON_END = "2026-12-31"      # last date the festival calendar covers

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def get_festival(date):
    """Festival name for a date, or 'No Festival'."""
    return FESTIVAL_CALENDAR.get(pd.Timestamp(date).strftime("%Y-%m-%d"), "No Festival")


def get_season(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5, 6):
        return "Summer"
    if month in (7, 8, 9):
        return "Monsoon"
    return "Post-Monsoon"


# --------------------------------------------------------------------------
# Feature engineering (used on the full dataset and on single scenarios)
# --------------------------------------------------------------------------

def add_time_features(df):
    """Month / quarter / week / day-of-month and their cyclical versions. Needs a datetime 'Date'."""
    df = df.copy()
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day_of_Month"] = df["Date"].dt.day

    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Week_Sin"] = np.sin(2 * np.pi * df["Week"] / 52)
    df["Week_Cos"] = np.cos(2 * np.pi * df["Week"] / 52)
    return df


def add_price_features(df):
    """Relative price versus the item's own base price and versus the competitor."""
    df = df.copy()
    df["Price_Index"] = df["Selling_Price"] / df["Base_Price"]
    df["Competitor_Price_Gap"] = df["Selling_Price"] - df["Competitor_Price"]
    df["Competitor_Price_Ratio"] = df["Selling_Price"] / df["Competitor_Price"]
    return df


def add_history_features(df):
    """Average past demand and past price of each item at each outlet.

    Only days strictly before the current day are used (shift(1)), so the
    feature never contains the value it is helping to predict.
    """
    df = df.sort_values(["Item_ID", "Outlet_ID", "Date"]).reset_index(drop=True)
    grouped = df.groupby(["Item_ID", "Outlet_ID"])
    df["Avg_Item_Demand"] = grouped["Units_Sold"].transform(lambda x: x.shift(1).expanding().mean())
    df["Avg_Item_Price"] = grouped["Selling_Price"].transform(lambda x: x.shift(1).expanding().mean())
    return df.sort_values(["Date", "Outlet_ID", "Item_ID"]).reset_index(drop=True)


# Columns that are outcomes, costs or labels - never model inputs
NON_FEATURE_COLUMNS = [
    "Date", "Item_Name",
    "Units_Sold", "Revenue", "Profit",
    "Unit_Cost", "Overhead_Cost", "Promotion_Cost", "Total_Cost",
]

CATEGORICAL_FEATURES = [
    "Day_of_Week", "Season", "Festival",
    "Item_ID", "Category", "Food_Type",
    "Outlet_ID", "City", "Outlet_Type", "Location_Type",
    "Promotion_Type",
]

NUMERICAL_FEATURES = [
    "Is_Weekend", "Holiday_Flag", "Footfall_Index",
    "Base_Price", "Selling_Price",
    "Discount_Percentage", "Promotion_Flag", "Promotion_Depth",
    "Competitor_Price", "Competitor_Promotion_Flag", "Competitor_Discount",
    "Temperature", "Rainfall",
    "Month", "Quarter", "Week", "Day_of_Month",
    "Month_Sin", "Month_Cos", "Week_Sin", "Week_Cos",
    "Price_Index", "Competitor_Price_Gap", "Competitor_Price_Ratio",
    "Avg_Item_Demand", "Avg_Item_Price",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Every feature that rises when our own price rises. The demand model is told
# that predicted demand may never go UP when one of these goes up.
PRICE_MONOTONE_FEATURES = ["Selling_Price", "Price_Index", "Competitor_Price_Gap", "Competitor_Price_Ratio"]


# --------------------------------------------------------------------------
# Scenario building
# --------------------------------------------------------------------------

def calendar_fields(date):
    """Everything the model needs to know about a date."""
    date = pd.Timestamp(date)
    festival = get_festival(date)
    month = date.month
    week = int(date.isocalendar()[1])
    return {
        "Date": date,
        "Day_of_Week": DAY_NAMES[date.dayofweek],
        "Is_Weekend": int(date.dayofweek >= 5),
        "Season": get_season(month),
        "Festival": festival,
        "Holiday_Flag": int(festival != "No Festival"),
        "Month": month,
        "Quarter": (month - 1) // 3 + 1,
        "Week": week,
        "Day_of_Month": date.day,
        "Month_Sin": np.sin(2 * np.pi * month / 12),
        "Month_Cos": np.cos(2 * np.pi * month / 12),
        "Week_Sin": np.sin(2 * np.pi * week / 52),
        "Week_Cos": np.cos(2 * np.pi * week / 52),
    }


def build_context(item_history, date, overrides=None):
    """Describe the situation an item is being priced in.

    item_history : processed sales rows of ONE menu item at ONE outlet
    date         : the day being planned (may be after the history ends)
    overrides    : optional dict to replace any expected condition, e.g.
                   {"Rainfall": 20, "Competitor_Price": 240}

    Costs and the competitor's regular price come from the last four weeks,
    expected footfall from the same weekday (or the same festival) in the
    past, and expected weather from the same calendar month.
    """
    h = item_history.sort_values("Date")
    last = h.iloc[-1]
    recent = h.tail(28)
    cal = calendar_fields(date)

    if cal["Holiday_Flag"] == 1 and (h["Festival"] == cal["Festival"]).any():
        footfall = h.loc[h["Festival"] == cal["Festival"], "Footfall_Index"].mean()
    else:
        same_day = h[(h["Day_of_Week"] == cal["Day_of_Week"]) & (h["Holiday_Flag"] == 0)]
        footfall = same_day["Footfall_Index"].mean()

    same_month = h[h["Date"].dt.month == cal["Month"]]
    regular_competitor = recent.loc[recent["Competitor_Promotion_Flag"] == 0, "Competitor_Price"]

    context = {
        "Item_ID": last["Item_ID"],
        "Item_Name": last["Item_Name"],
        "Category": last["Category"],
        "Food_Type": last["Food_Type"],
        "Outlet_ID": last["Outlet_ID"],
        "City": last["City"],
        "Outlet_Type": last["Outlet_Type"],
        "Location_Type": last["Location_Type"],
        "Base_Price": float(last["Base_Price"]),
        "Footfall_Index": float(footfall),
        "Temperature": float(same_month["Temperature"].mean()),
        "Rainfall": float(same_month["Rainfall"].mean()),
        "Competitor_Price": float(regular_competitor.median() if len(regular_competitor) else last["Competitor_Price"]),
        "Competitor_Promotion_Flag": 0,
        "Competitor_Discount": 0.0,
        "Avg_Item_Demand": float(h["Units_Sold"].mean()),
        "Avg_Item_Price": float(h["Selling_Price"].mean()),
        "Unit_Cost": float(recent["Unit_Cost"].mean()),
        "Overhead_Cost": float(recent["Overhead_Cost"].mean()),
    }
    context.update(cal)
    if overrides:
        context.update(overrides)
    return context


def promotion_cost_per_unit(promotion_type, base_price, unit_cost):
    """Extra cost of an offer for every unit sold (on top of the lower price)."""
    if promotion_type == "Combo Offer":
        return COMBO_ADDON_COST_RATE * base_price
    if promotion_type == "Buy One Get One":
        return unit_cost                      # the free second plate
    return 0.0


def build_scenarios(context, menu_prices, promotion_types):
    """One row per (menu price, offer) pair, with every model input filled in."""
    rows = []
    for menu_price, promotion_type in zip(menu_prices, promotion_types):
        promo = PROMOTION_PARAMETERS[promotion_type]
        menu_price = float(np.round(menu_price))
        row = dict(context)
        row["Menu_Price"] = menu_price
        row["Selling_Price"] = float(np.round(menu_price * promo["price_factor"]))
        row["Promotion_Type"] = promotion_type
        row["Promotion_Flag"] = promo["flag"]
        row["Promotion_Depth"] = promo["depth"]
        row["Discount_Percentage"] = promo["discount"]
        rows.append(row)
    return add_price_features(pd.DataFrame(rows))


def score_scenarios(model, scenarios):
    """Predict demand for each scenario and turn it into revenue, cost and profit."""
    out = scenarios.copy()
    units = np.clip(model.predict(out[list(model.feature_names_in_)]), 0, None).astype(float)

    promo_cost_unit = np.array([
        promotion_cost_per_unit(p, b, c)
        for p, b, c in zip(out["Promotion_Type"], out["Base_Price"], out["Unit_Cost"])
    ])
    out["Predicted_Units"] = units
    out["Revenue"] = out["Selling_Price"] * units
    out["Food_Cost"] = out["Unit_Cost"] * units
    out["Promotion_Cost"] = promo_cost_unit * units
    out["Total_Cost"] = out["Food_Cost"] + out["Overhead_Cost"] + out["Promotion_Cost"]
    out["Expected_Profit"] = out["Revenue"] - out["Total_Cost"]
    return out


def evaluate_scenario(model, context, menu_price, promotion_type):
    """Demand and profit for a single price / offer choice."""
    return score_scenarios(model, build_scenarios(context, [menu_price], [promotion_type])).iloc[0]


def scenario_grid(base_price):
    """All 35 candidate strategies: 5 menu-price levels x 7 offers.

    Guardrail: an offer is only allowed on a menu price at or below the base
    price. Raising the menu price and then "discounting" it back would be a
    fake discount, so those 12 combinations are marked invalid.
    """
    grid = []
    for adjustment in PRICE_ADJUSTMENTS:
        for promotion_type in PROMOTION_OPTIONS:
            grid.append({
                "Price_Adjustment": adjustment,
                "Menu_Price": float(np.round(base_price * (1 + adjustment))),
                "Promotion_Type": promotion_type,
                "Valid": (promotion_type == "No Promotion") or (adjustment <= 0),
            })
    return pd.DataFrame(grid)


RESULT_COLUMNS = [
    "Price_Adjustment", "Menu_Price", "Promotion_Type", "Selling_Price",
    "Predicted_Units", "Revenue", "Food_Cost", "Promotion_Cost",
    "Overhead_Cost", "Total_Cost", "Expected_Profit", "Valid",
]


def optimize(model, context):
    """Score all 35 candidate strategies for one context, best valid strategy first."""
    grid = scenario_grid(context["Base_Price"])
    scored = score_scenarios(model, build_scenarios(context, grid["Menu_Price"], grid["Promotion_Type"]))
    scored["Price_Adjustment"] = grid["Price_Adjustment"].to_numpy()
    scored["Valid"] = grid["Valid"].to_numpy()
    return (
        scored[RESULT_COLUMNS]
        .sort_values(["Valid", "Expected_Profit"], ascending=[False, False])
        .reset_index(drop=True)
    )


def best_by_promotion(results):
    """Best valid menu price for each offer, ranked by expected profit."""
    valid = results[results["Valid"]].sort_values("Expected_Profit", ascending=False)
    best = valid.groupby("Promotion_Type", sort=False).first()
    best = best.rename(columns={"Menu_Price": "Best_Menu_Price", "Selling_Price": "Customer_Pays"})
    return best[["Best_Menu_Price", "Customer_Pays", "Predicted_Units", "Revenue",
                 "Promotion_Cost", "Total_Cost", "Expected_Profit"]]


def price_in_trained_range(menu_price, base_price, tolerance=0.006):
    """True when the menu price is inside the range the model has seen (+/-10% of base price)."""
    index = menu_price / base_price
    return (TRAINED_PRICE_RANGE[0] - tolerance) <= index <= (TRAINED_PRICE_RANGE[1] + tolerance)
