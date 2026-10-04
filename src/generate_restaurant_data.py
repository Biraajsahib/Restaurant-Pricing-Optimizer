"""
Synthetic restaurant sales data generator
=========================================

Creates ``data/synthetic_restaurant_sales_data.csv``: daily unit sales for every
menu item at every outlet of a fictional multi-outlet Indian restaurant brand.

Why synthetic?  Real restaurant sales data with price changes, offers and costs
is confidential.  A synthetic dataset lets the full pricing workflow
(predict -> optimise -> explain) be demonstrated end to end, and because the
demand rules below are known, the optimiser can be checked against the truth
(see ``src/validate_optimizer.py``).

Run from the project root:

    python src/generate_restaurant_data.py

The generator is seeded, so it always reproduces the same file.

Demand logic (one row = one menu item, at one outlet, on one day)
-----------------------------------------------------------------
expected units =  item popularity at that outlet
                x outlet traffic that day      (weekday, festival, rain, payday)
                x item seasonality             (temperature, festival, Tuesday)
                x price effect                 (cheaper -> more orders)
                x offer effect                 (advertised deals sell extra)
                x competitor effect            (competitor cheaper -> fewer orders)
units sold     =  Poisson noise around the expected value
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pricing_engine import (  # noqa: E402  (shared definitions, so data and app always agree)
    COMBO_ADDON_COST_RATE,
    FESTIVAL_CALENDAR,
    PRICE_ADJUSTMENTS,
    PROMOTION_OPTIONS,
    PROMOTION_PARAMETERS,
    get_season,
)

SEED = 42
START_DATE = "2024-10-01"
END_DATE = "2026-09-30"

# --------------------------------------------------------------------------
# Master data
# --------------------------------------------------------------------------

# Item_ID, Item_Name, Category, Food_Type, list price (Rs), food cost as a
# share of list price, average daily units at a "size 1.0" outlet,
# temperature sensitivity (+ve = sells more on hot days)
MENU = [
    ("M01", "Paneer Tikka",           "Starters",       "Veg",     285, 0.31, 26,  0.000),
    ("M02", "Chicken Tikka",          "Starters",       "Non-Veg", 325, 0.34, 30,  0.000),
    ("M03", "Veg Spring Rolls",       "Starters",       "Veg",     215, 0.27, 18,  0.000),
    ("M04", "Honey Chilli Potato",    "Starters",       "Veg",     225, 0.24, 22,  0.000),
    ("M05", "Tandoori Chicken",       "Starters",       "Non-Veg", 345, 0.37, 20, -0.006),
    ("M06", "Butter Chicken",         "Main Course",    "Non-Veg", 395, 0.35, 38, -0.004),
    ("M07", "Dal Makhani",            "Main Course",    "Veg",     295, 0.24, 34, -0.004),
    ("M08", "Paneer Butter Masala",   "Main Course",    "Veg",     335, 0.32, 30,  0.000),
    ("M09", "Kadhai Chicken",         "Main Course",    "Non-Veg", 385, 0.36, 22,  0.000),
    ("M10", "Mixed Veg Curry",        "Main Course",    "Veg",     255, 0.26, 16,  0.000),
    ("M11", "Chicken Biryani",        "Rice & Biryani", "Non-Veg", 345, 0.38, 40,  0.000),
    ("M12", "Veg Biryani",            "Rice & Biryani", "Veg",     275, 0.30, 22,  0.000),
    ("M13", "Mutton Biryani",         "Rice & Biryani", "Non-Veg", 445, 0.42, 14,  0.000),
    ("M14", "Jeera Rice",             "Rice & Biryani", "Veg",     165, 0.20, 26,  0.000),
    ("M15", "Veg Fried Rice",         "Rice & Biryani", "Veg",     215, 0.25, 18,  0.000),
    ("M16", "Butter Naan",            "Breads",         "Veg",      65, 0.20, 95,  0.000),
    ("M17", "Garlic Naan",            "Breads",         "Veg",      85, 0.20, 60,  0.000),
    ("M18", "Tandoori Roti",          "Breads",         "Veg",      35, 0.22, 85,  0.000),
    ("M19", "Lachha Paratha",         "Breads",         "Veg",      75, 0.21, 40,  0.000),
    ("M20", "Amritsari Kulcha",       "Breads",         "Veg",     125, 0.26, 24,  0.000),
    ("M21", "Sweet Lassi",            "Beverages",      "Veg",     115, 0.26, 28,  0.030),
    ("M22", "Fresh Lime Soda",        "Beverages",      "Veg",      95, 0.15, 34,  0.038),
    ("M23", "Cold Coffee",            "Beverages",      "Veg",     165, 0.24, 30,  0.030),
    ("M24", "Masala Chai",            "Beverages",      "Veg",      55, 0.18, 45, -0.035),
    ("M25", "Virgin Mojito",          "Beverages",      "Veg",     175, 0.20, 20,  0.034),
    ("M26", "Gulab Jamun",            "Desserts",       "Veg",     115, 0.24, 24, -0.012),
    ("M27", "Rasmalai",               "Desserts",       "Veg",     145, 0.30, 16,  0.006),
    ("M28", "Brownie with Ice Cream", "Desserts",       "Veg",     215, 0.32, 18,  0.012),
    ("M29", "Malai Kulfi",            "Desserts",       "Veg",     125, 0.28, 14,  0.036),
    ("M30", "Gajar Halwa",            "Desserts",       "Veg",     155, 0.27, 12, -0.060),
]

# Outlet_ID, City, Outlet_Type, Location_Type, traffic scale, price positioning
OUTLETS = [
    ("R01", "Delhi",      "Casual Dining", "High Street", 1.25, 1.05),
    ("R02", "Delhi",      "Quick Service", "Mall",        1.15, 0.95),
    ("R03", "Chandigarh", "Casual Dining", "Mall",        0.95, 1.00),
    ("R04", "Mumbai",     "Casual Dining", "High Street", 1.30, 1.12),
    ("R05", "Mumbai",     "Cloud Kitchen", "Residential", 0.90, 0.92),
    ("R06", "Bengaluru",  "Cafe",          "Office Hub",  1.00, 1.06),
    ("R07", "Hyderabad",  "Quick Service", "Office Hub",  1.05, 0.94),
    ("R08", "Pune",       "Cafe",          "High Street", 0.85, 1.02),
]

CATEGORIES = ["Starters", "Main Course", "Rice & Biryani", "Breads", "Beverages", "Desserts"]
OUTLET_TYPES = ["Casual Dining", "Quick Service", "Cafe", "Cloud Kitchen"]

# How well each category sells in each outlet format (rows = category)
CATEGORY_AFFINITY = pd.DataFrame(
    [[1.10, 0.85, 0.70, 0.80],
     [1.15, 0.70, 0.45, 1.05],
     [0.95, 1.25, 0.40, 1.40],
     [1.15, 0.70, 0.35, 0.95],
     [0.85, 1.10, 1.80, 0.45],
     [0.95, 0.90, 1.50, 0.60]],
    index=CATEGORIES, columns=OUTLET_TYPES,
)

# Traffic by day of week (Mon..Sun) for each location type; each row averages 1.0
DOW_TRAFFIC = {
    "High Street": [0.82, 0.84, 0.88, 0.93, 1.12, 1.24, 1.17],
    "Mall":        [0.74, 0.76, 0.80, 0.86, 1.12, 1.38, 1.34],
    "Office Hub":  [1.06, 1.08, 1.10, 1.08, 1.04, 0.84, 0.80],
    "Residential": [0.88, 0.88, 0.90, 0.94, 1.10, 1.16, 1.14],
}

# Outlet traffic multiplier on each festival
FESTIVAL_TRAFFIC = {
    "No Festival": 1.00, "New Year": 1.45, "Republic Day": 1.15, "Valentines Day": 1.40,
    "Holi": 0.85, "Eid": 1.25, "Independence Day": 1.15, "Raksha Bandhan": 1.25,
    "Dussehra": 1.20, "Diwali": 1.20, "Christmas": 1.35,
}

# Extra category-level lift on a festival (on top of traffic)
FESTIVAL_CATEGORY_BOOST = {
    "Eid":            {"Rice & Biryani": 1.45, "Main Course": 1.10},
    "Valentines Day": {"Desserts": 1.50, "Beverages": 1.25},
    "Diwali":         {"Desserts": 1.40},
    "Raksha Bandhan": {"Desserts": 1.35},
    "Christmas":      {"Desserts": 1.30, "Beverages": 1.20, "Starters": 1.15},
    "New Year":       {"Starters": 1.25, "Beverages": 1.25, "Desserts": 1.15},
    "Holi":           {"Beverages": 1.20},
}

# Monthly climate per city: mean temperature (C), chance of rain, mean rain on a wet day (mm)
CITY_CLIMATE = {
    "Delhi":      ([14, 17, 23, 29, 33, 34, 31, 30, 29, 26, 20, 15],
                   [.06, .07, .06, .05, .08, .18, .45, .48, .25, .05, .02, .03], 14),
    "Chandigarh": ([12, 15, 21, 27, 31, 32, 29, 28, 27, 23, 18, 13],
                   [.10, .12, .10, .06, .10, .25, .55, .55, .30, .06, .03, .06], 16),
    "Mumbai":     ([24, 25, 27, 29, 30, 29, 28, 27, 28, 29, 28, 26],
                   [.01, .01, .01, .02, .05, .60, .85, .80, .55, .15, .04, .02], 24),
    "Bengaluru":  ([21, 23, 26, 28, 27, 24, 23, 23, 23, 23, 22, 21],
                   [.03, .03, .06, .20, .35, .35, .40, .45, .50, .45, .20, .08], 10),
    "Hyderabad":  ([22, 25, 28, 31, 33, 29, 27, 26, 27, 26, 23, 21],
                   [.02, .03, .04, .08, .12, .35, .50, .50, .45, .25, .08, .03], 12),
    "Pune":       ([21, 23, 27, 30, 30, 27, 25, 24, 25, 26, 23, 21],
                   [.01, .01, .02, .04, .10, .50, .70, .60, .45, .20, .06, .02], 13),
}

# --------------------------------------------------------------------------
# Offers
# --------------------------------------------------------------------------

PROMOTION_TYPES = PROMOTION_OPTIONS
# price the customer pays, as a share of the menu price
PROMO_PRICE_FACTOR = np.array([PROMOTION_PARAMETERS[p]["price_factor"] for p in PROMOTION_TYPES])
PROMO_DISCOUNT_PCT = np.array([PROMOTION_PARAMETERS[p]["discount"] for p in PROMOTION_TYPES])
PROMO_DEPTH = np.array([PROMOTION_PARAMETERS[p]["depth"] for p in PROMOTION_TYPES])
PROMO_MIX = np.array([0.14, 0.22, 0.18, 0.14, 0.16, 0.16])   # mix among the 6 offers when one runs
PROMO_WEEK_PROB = 0.45                                        # chance an item runs an offer in a given week
PRICE_LEVELS = np.array(PRICE_ADJUSTMENTS)                    # menu price tests around the base price
PRICE_LEVEL_PROB = np.array([0.14, 0.20, 0.32, 0.20, 0.14])

# --------------------------------------------------------------------------
# Demand "truth" (what the model has to learn from the data)
# --------------------------------------------------------------------------

# Price sensitivity k: demand changes by about -k% for every +1% in the price paid.
# It differs by menu category, is higher in delivery-led formats (guests compare
# prices on apps) and on slow weekdays, and lower on weekends and festivals.
K_CATEGORY = {"Starters": 1.48, "Main Course": 1.38, "Rice & Biryani": 1.45,
              "Breads": 1.16, "Beverages": 1.36, "Desserts": 1.45}
K_OUTLET_TYPE = {"Casual Dining": 0.98, "Quick Service": 1.03, "Cafe": 0.95, "Cloud Kitchen": 1.08}
K_DAY_TYPE = np.array([1.06, 1.00, 0.91, 0.87])                # Mon-Thu, Fri, Sat-Sun, festival
K_ITEM_SPREAD = 0.03                                           # small item-by-outlet variation (+/-)

# Offers sell extra on top of the pure price effect - but mostly on slow days,
# and more in delivery-led formats where guests compare deals.
OFFER_DAY_WEIGHT = np.array([1.00, 0.70, 0.20, 0.10])          # Mon-Thu, Fri, Sat-Sun, festival
OFFER_TYPE_WEIGHT = {"Casual Dining": 1.00, "Quick Service": 1.05, "Cafe": 0.90, "Cloud Kitchen": 1.15}
# advertised % discount: extra response to the "deal" itself, with diminishing
# returns as the discount gets deeper  (lift = A * (1 - exp(-discount / 0.12)))
DISCOUNT_LIFT = {"Starters": 0.18, "Main Course": 0.07, "Rice & Biryani": 0.07,
                 "Breads": 0.03, "Beverages": 0.20, "Desserts": 0.20}
DISCOUNT_SATURATION = 0.12
# combo (item + add-on at a bundled price): works best for "meal" items
COMBO_LIFT = {"Starters": 0.10, "Main Course": 0.24, "Rice & Biryani": 0.26,
              "Breads": 0.03, "Beverages": 0.05, "Desserts": 0.06}
# buy one get one: lift in paid orders
BOGO_LIFT = {"Starters": 0.50, "Main Course": 0.40, "Rice & Biryani": 0.40,
             "Breads": 0.20, "Beverages": 0.55, "Desserts": 0.70}
COMPETITOR_GAMMA = 0.45          # response to the competitor's price
NOISE_SIGMA = 0.06               # unexplained day-to-day variation
OVERHEAD_SHARE = 0.30            # fixed overhead allocated per item-day (share of normal revenue)


def demand_multiplier(k, price_paid_index, promo_idx, cat_names, offer_weight):
    """Price + offer effect on demand, relative to selling at base price with no offer.

    k                 price sensitivity
    price_paid_index  price the customer pays / base price
    promo_idx         index into PROMOTION_TYPES
    cat_names         menu category
    offer_weight      how well offers work in this context (day type x outlet format)
    (all arguments are arrays of the same shape, or scalars)
    """
    k = np.asarray(k, dtype=float)
    x = np.asarray(price_paid_index, dtype=float)
    promo_idx = np.asarray(promo_idx)
    cat_names = np.asarray(cat_names)
    w = np.asarray(offer_weight, dtype=float)

    disc = np.vectorize(DISCOUNT_LIFT.get)(cat_names).astype(float)
    combo = np.vectorize(COMBO_LIFT.get)(cat_names).astype(float)
    bogo = np.vectorize(BOGO_LIFT.get)(cat_names).astype(float)

    advertised = PROMO_DISCOUNT_PCT[promo_idx] / 100.0
    lift = np.ones(np.broadcast(x, promo_idx).shape)
    lift = np.where((promo_idx >= 1) & (promo_idx <= 4),
                    1.0 + disc * w * (1.0 - np.exp(-advertised / DISCOUNT_SATURATION)), lift)
    lift = np.where(promo_idx == 5, 1.0 + combo * w, lift)
    lift = np.where(promo_idx == 6, 1.0 + bogo * w, lift)
    return np.exp(-k * (x - 1.0)) * lift


def generate(seed=SEED, return_hidden=False):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    T, O, I = len(dates), len(OUTLETS), len(MENU)

    menu = pd.DataFrame(MENU, columns=["Item_ID", "Item_Name", "Category", "Food_Type",
                                       "List_Price", "Cost_Ratio", "Base_Volume", "Temp_Sens"])
    outlets = pd.DataFrame(OUTLETS, columns=["Outlet_ID", "City", "Outlet_Type",
                                             "Location_Type", "Traffic_Scale", "Price_Factor"])
    cat = menu["Category"].to_numpy()
    otype = outlets["Outlet_Type"].to_numpy()

    # ---------------- calendar ----------------
    dow = dates.dayofweek.to_numpy()                       # 0 = Monday
    month = dates.month.to_numpy()
    dom = dates.day.to_numpy()
    festival = np.array([FESTIVAL_CALENDAR.get(d.strftime("%Y-%m-%d"), "No Festival") for d in dates])
    holiday = (festival != "No Festival").astype(int)
    is_weekend = (dow >= 5).astype(int)
    day_type = np.where(holiday == 1, 3, np.where(dow >= 5, 2, np.where(dow == 4, 1, 0)))
    season = np.array([get_season(m) for m in month])

    # ---------------- weather (per city, per day) ----------------
    city_temp, city_rain = {}, {}
    mid_month_day = np.array([15.5 + 30.4375 * m for m in range(-1, 13)])   # Dec(prev) .. Jan(next)
    doy = dates.dayofyear.to_numpy().astype(float)
    for city, (temps, rain_p, rain_mm) in CITY_CLIMATE.items():
        t_ext = np.array([temps[11]] + list(temps) + [temps[0]], dtype=float)
        smooth = np.interp(doy, mid_month_day, t_ext)
        noise = np.zeros(T)
        for t in range(1, T):                                # day-to-day persistence
            noise[t] = 0.6 * noise[t - 1] + rng.normal(0, 1.5)
        city_temp[city] = smooth + noise
        p = np.array(rain_p)[month - 1]
        wet = rng.random(T) < p
        city_rain[city] = np.where(wet, rng.gamma(1.3, rain_mm / 1.3, T), 0.0)
    temp = np.column_stack([city_temp[c] for c in outlets["City"]])          # (T, O)
    rain = np.column_stack([city_rain[c] for c in outlets["City"]])

    # ---------------- outlet traffic (footfall index) ----------------
    dow_eff = np.column_stack([np.array(DOW_TRAFFIC[l])[dow] for l in outlets["Location_Type"]])
    fest_eff = np.array([FESTIVAL_TRAFFIC[f] for f in festival])[:, None]
    is_cloud = (otype == "Cloud Kitchen")[None, :]
    rain_eff = np.where(is_cloud, 1.0 + np.minimum(0.25, 0.010 * rain), 1.0 - np.minimum(0.30, 0.012 * rain))
    payday_eff = np.where(dom <= 7, 1.06, np.where(dom >= 25, 0.96, 1.00))[:, None]
    season_eff = np.array([1.04, 1.04, 1.00, 1.00, 0.96, 0.95, 0.96, 0.97, 1.00, 1.02, 1.04, 1.05])[month - 1][:, None]
    shock = rng.lognormal(0.0, 0.07, size=(T, O))
    footfall = dow_eff * fest_eff * rain_eff * payday_eff * season_eff * shock

    # ---------------- prices ----------------
    base_price = np.round(menu["List_Price"].to_numpy()[None, :] * outlets["Price_Factor"].to_numpy()[:, None] / 5.0) * 5.0   # (O, I)

    price_level = np.zeros((T, O, I))
    for o in range(O):
        for i in range(I):
            t = 0
            while t < T:
                length = int(rng.integers(14, 36))
                price_level[t:t + length, o, i] = rng.choice(PRICE_LEVELS, p=PRICE_LEVEL_PROB)
                t += length

    # ---------------- offers ----------------
    promo_idx = np.zeros((T, O, I), dtype=int)
    week_start = np.arange(0, T, 7)
    for o in range(O):
        for i in range(I):
            runs = rng.random(len(week_start)) < PROMO_WEEK_PROB
            for w in week_start[runs]:
                length = int(rng.integers(3, 8))
                start = w + int(rng.integers(0, 8 - length))
                promo_idx[start:start + length, o, i] = 1 + rng.choice(6, p=PROMO_MIX)

    menu_price = np.round(base_price[None, :, :] * (1.0 + price_level))
    selling_price = np.round(menu_price * PROMO_PRICE_FACTOR[promo_idx])
    price_index = selling_price / base_price[None, :, :]

    # ---------------- competitor ----------------
    comp_position = rng.uniform(0.92, 1.08, size=(O, I))
    comp_drift = np.zeros((T, O, I))
    for w in range(0, T, 7):
        prev = comp_drift[w - 1] if w > 0 else np.zeros((O, I))
        comp_drift[w:w + 7] = 0.85 * prev + rng.normal(0, 0.012, size=(O, I))
    comp_disc = np.zeros((T, O, I))
    for o in range(O):
        for i in range(I):
            runs = rng.random(len(week_start)) < 0.12
            for w in week_start[runs]:
                length = int(rng.integers(3, 8))
                comp_disc[w:w + length, o, i] = rng.choice([0.10, 0.15, 0.20, 0.25])
    comp_regular = base_price[None, :, :] * comp_position[None, :, :] * (1.0 + comp_drift)
    competitor_price = np.round(comp_regular * (1.0 - comp_disc))
    comp_effect = (competitor_price / (base_price[None, :, :] * comp_position[None, :, :])) ** COMPETITOR_GAMMA

    # ---------------- demand ----------------
    affinity = CATEGORY_AFFINITY.loc[cat, otype].to_numpy().T                 # (O, I)
    item_outlet_effect = rng.lognormal(0.0, 0.12, size=(O, I))
    popularity = (menu["Base_Volume"].to_numpy()[None, :] * outlets["Traffic_Scale"].to_numpy()[:, None]
                  * affinity * item_outlet_effect)                            # (O, I)

    temp_eff = np.exp(menu["Temp_Sens"].to_numpy()[None, None, :] * (temp[:, :, None] - 25.0))
    chai_rain = np.ones((T, O, I))
    chai_rain[:, :, menu.index[menu["Item_Name"] == "Masala Chai"][0]] = 1.0 + np.minimum(0.30, 0.010 * rain)
    tuesday_nonveg = np.where((dow[:, None, None] == 1) & (menu["Food_Type"].to_numpy()[None, None, :] == "Non-Veg"), 0.88, 1.0)
    fest_cat = np.ones((T, I))
    for t in range(T):
        boost = FESTIVAL_CATEGORY_BOOST.get(festival[t])
        if boost:
            fest_cat[t] = [boost.get(c, 1.0) for c in cat]

    k_base = (np.array([K_CATEGORY[c] for c in cat])[None, :]
              * np.array([K_OUTLET_TYPE[t_] for t_ in otype])[:, None]
              * rng.uniform(1.0 - K_ITEM_SPREAD, 1.0 + K_ITEM_SPREAD, size=(O, I)))             # (O, I)
    k = k_base[None, :, :] * K_DAY_TYPE[day_type][:, None, None]                               # (T, O, I)
    offer_weight = (OFFER_DAY_WEIGHT[day_type][:, None, None]
                    * np.array([OFFER_TYPE_WEIGHT[t_] for t_ in otype])[None, :, None]
                    * np.ones((T, O, I)))

    cat_grid = np.broadcast_to(cat[None, None, :], (T, O, I))
    price_promo_effect = demand_multiplier(k, price_index, promo_idx, cat_grid, offer_weight)

    baseline = (popularity[None, :, :] * footfall[:, :, None] * temp_eff * chai_rain
                * tuesday_nonveg * fest_cat[:, None, :] * comp_effect)
    expected = baseline * price_promo_effect
    noise = rng.lognormal(-NOISE_SIGMA ** 2 / 2.0, NOISE_SIGMA, size=(T, O, I))
    units = rng.poisson(expected * noise)

    # ---------------- costs ----------------
    cost_index = 1.0 + 0.025 * np.sin(2 * np.pi * (doy - 60) / 365.25) + np.cumsum(rng.normal(0, 0.0012, T))
    cost_index = cost_index / cost_index.mean()
    unit_cost = (menu["Cost_Ratio"].to_numpy()[None, None, :] * base_price[None, :, :]
                 * cost_index[:, None, None] * rng.normal(1.0, 0.012, size=(T, O, I)))
    unit_cost = np.round(unit_cost, 2)
    overhead = np.round(OVERHEAD_SHARE * base_price[None, :, :] * popularity[None, :, :]
                        * rng.normal(1.0, 0.02, size=(T, O, I)), 2)
    promo_cost = np.where(promo_idx == 5, COMBO_ADDON_COST_RATE * base_price[None, :, :] * units,
                 np.where(promo_idx == 6, unit_cost * units, 0.0))
    promo_cost = np.round(promo_cost, 2)
    revenue = np.round(selling_price * units, 2)
    total_cost = np.round(unit_cost * units + overhead + promo_cost, 2)
    profit = np.round(revenue - total_cost, 2)

    # ---------------- assemble (one row per date x outlet x item) ----------------
    def per_day(a):      return np.repeat(a, O * I)
    def per_outlet(a):   return np.tile(np.repeat(a, I), T)
    def per_item(a):     return np.tile(a, T * O)
    def day_outlet(a):   return np.repeat(a.reshape(-1), I)
    def outlet_item(a):  return np.tile(a.reshape(-1), T)

    df = pd.DataFrame({
        "Date": per_day(dates.strftime("%Y-%m-%d").to_numpy()),
        "Day_of_Week": per_day(dates.day_name().to_numpy()),
        "Is_Weekend": per_day(is_weekend),
        "Season": per_day(season),
        "Holiday_Flag": per_day(holiday),
        "Festival": per_day(festival),
        "Item_ID": per_item(menu["Item_ID"].to_numpy()),
        "Item_Name": per_item(menu["Item_Name"].to_numpy()),
        "Category": per_item(cat),
        "Food_Type": per_item(menu["Food_Type"].to_numpy()),
        "Outlet_ID": per_outlet(outlets["Outlet_ID"].to_numpy()),
        "City": per_outlet(outlets["City"].to_numpy()),
        "Outlet_Type": per_outlet(otype),
        "Location_Type": per_outlet(outlets["Location_Type"].to_numpy()),
        "Footfall_Index": np.round(day_outlet(footfall), 3),
        "Base_Price": outlet_item(base_price).astype(int),
        "Selling_Price": selling_price.reshape(-1).astype(int),
        "Discount_Percentage": PROMO_DISCOUNT_PCT[promo_idx].reshape(-1).astype(int),
        "Promotion_Flag": (promo_idx > 0).astype(int).reshape(-1),
        "Promotion_Type": np.array(PROMOTION_TYPES)[promo_idx].reshape(-1),
        "Promotion_Depth": PROMO_DEPTH[promo_idx].reshape(-1),
        "Promotion_Cost": promo_cost.reshape(-1),
        "Competitor_Price": competitor_price.reshape(-1).astype(int),
        "Competitor_Promotion_Flag": (comp_disc > 0).astype(int).reshape(-1),
        "Competitor_Discount": comp_disc.reshape(-1),
        "Temperature": np.round(day_outlet(temp), 1),
        "Rainfall": np.round(day_outlet(rain), 1),
        "Units_Sold": units.reshape(-1),
        "Unit_Cost": unit_cost.reshape(-1),
        "Overhead_Cost": overhead.reshape(-1),
        "Total_Cost": total_cost.reshape(-1),
        "Revenue": revenue.reshape(-1),
        "Profit": profit.reshape(-1),
    })

    if return_hidden:
        hidden = pd.DataFrame({
            "Date": df["Date"], "Item_ID": df["Item_ID"], "Outlet_ID": df["Outlet_ID"],
            "true_k": k.reshape(-1),                           # price sensitivity
            "true_offer_weight": offer_weight.reshape(-1),     # how well offers work that day
            "true_baseline": baseline.reshape(-1),             # expected units at base price, no offer
        })
        return df, hidden
    return df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_restaurant_sales_data.csv")
    data = generate()
    data.to_csv(out_path, index=False)
    print("Saved", os.path.abspath(out_path))
    print("Rows:", len(data), "| Columns:", data.shape[1])
    print("Date range:", data["Date"].min(), "to", data["Date"].max())
