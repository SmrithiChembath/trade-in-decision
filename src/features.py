"""
features.py: shared feature engineering for both datasets.

Both models must describe a car the SAME way (same columns, same category
lists), because in Week 3 we take a trade-in from the auction data and ask
the retail model what it would sell for. Different column names or category
orders between the two files would silently break that.
"""
import pandas as pd

from src import config as C


def in_scope(df):
    """Keep only the 20 make/model pairs, with one combined column."""
    df = df.copy()
    df["make_model"] = df["make"] + " " + df["model"]
    return df[df["make_model"].isin(C.SCOPE)]


def set_categories(df):
    """Fixed category list so codes mean the same thing in every model."""
    df["make_model"] = pd.Categorical(df["make_model"], categories=C.SCOPE)
    return df


def prep_auction(auction):
    """Auction sales: age at sale (years) and a shared mileage column."""
    df = in_scope(auction)
    sale_year = df["saledate"].dt.year + (df["saledate"].dt.dayofyear - 1) / 365.25
    df["age"] = (sale_year - df["year"]).clip(lower=0)
    df["miles"] = df["odometer"]
    # 2015 dollars -> 2020 dollars, so auction and retail use the same money.
    df["mmr_2020"] = df["mmr"] * C.INDEX_RATIO
    df["sellingprice_2020"] = df["sellingprice"] * C.INDEX_RATIO
    return set_categories(df)


def prep_retail(retail):
    """Retail listings: age at the scrape date, shared mileage column."""
    df = in_scope(retail)
    df = df[df["daysonmarket"] <= C.MAX_DOM]
    scrape = pd.Timestamp(C.SCRAPE_DATE)
    scrape_year = scrape.year + (scrape.dayofyear - 1) / 365.25
    df["age"] = (scrape_year - df["year"]).clip(lower=0)
    df["miles"] = df["mileage"]
    df["franchise_dealer"] = df["franchise_dealer"].astype(int)
    return set_categories(df)


# Feature lists used by each model
WHOLESALE_FEATURES = ["make_model", "age", "miles", "condition"]
RETAIL_FEATURES = ["make_model", "age", "miles", "franchise_dealer"]
