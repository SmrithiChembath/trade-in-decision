# %% [markdown]
# # 01 Explore: what do we actually have?
# Open in VS Code or Jupyter (the `# %%` lines are cells).
# Goal of this notebook: answer 4 questions before building any model.
# 1. Which make/model pairs appear often in BOTH files? (our scope)
# 2. When was each dataset collected? (the time gap problem)
# 3. How is days on market distributed, and how many listings are censored?
# 4. How do auction prices compare with MMR by condition?

# %%
import pandas as pd
import matplotlib.pyplot as plt
import sys; sys.path.append("..")
from src import config as C

auction = pd.read_parquet(C.PROCESSED / "auction.parquet")
retail = pd.read_parquet(C.PROCESSED / "retail.parquet")
print(auction.shape, retail.shape)

# %% [markdown]
# ## Q1. Scope: the make/model pairs common to both files
# The decision compares a retail price and an auction price for the SAME car,
# so we can only use models that appear often in both datasets.

# %%
a_counts = auction.groupby(["make", "model"]).size().rename("auction_n")
r_counts = retail.groupby(["make", "model"]).size().rename("retail_n")
both = pd.concat([a_counts, r_counts], axis=1, join="inner")
both["min_n"] = both.min(axis=1)
top = both.sort_values("min_n", ascending=False).head(C.TOP_N_MODELS)
top
# Expect some names to differ between files (e.g. "f-150" vs "f150").
# If a popular model is missing, check spellings and add a mapping in load_data.

# %% [markdown]
# ## Q2. The time gap
# If the auction data is ~2015 and the listings are ~2020, raw dollar prices
# are not comparable (used car prices moved a lot). This is why we model
# price RELATIVE to a market benchmark instead of raw dollars.

# %%
print("auction sales:", auction["saledate"].min(), "to", auction["saledate"].max())
print("retail listings:", retail["listed_date"].min(), "to", retail["listed_date"].max())

# %% [markdown]
# ## Q3. Days on market and censoring
# A listing that was still live when the data was scraped has NOT sold yet.
# Its days on market is a lower bound ("at least 40 days"), not the true time
# to sale. This is called right censoring. A survival model uses these rows
# correctly; simply averaging days on market would understate time to sale.
# Note: this dataset is a snapshot of live listings, so most rows are censored.
# Write down what share you see; it shapes how we interpret the survival model.

# %%
retail["daysonmarket"].clip(upper=365).hist(bins=60)
plt.xlabel("days on market (capped at 365)"); plt.ylabel("listings")
plt.title("Days on market"); plt.savefig(C.FIGURES / "dom_hist.png", dpi=150)
retail["daysonmarket"].describe()

# %% [markdown]
# ## Q4. Auction price vs MMR by condition
# This tells us how much condition matters at auction, which drives the
# "fix it up or sell it as is" side of the decision.

# %%
auction["cond_band"] = pd.cut(auction["condition"], [0, 19, 34, 44, 50],
                              labels=list(C.RECON_COST_BY_CONDITION))
auction.groupby("cond_band", observed=True)["price_to_mmr"].describe()

# %% [markdown]
# ##  Data findings (from 01_explore)
# - **Scope:** 20 make/model pairs with at least 4,400 auction sales and 4,900 retail listings each: Nissan Altima, Maxima, Rogue; Ford Fusion, Escape, F-150, Focus, Explorer, Edge; Toyota Camry, Corolla; Dodge Grand Caravan; Chevrolet Cruze, Malibu, Impala; Hyundai Sonata, Elantra; Honda Accord, Civic; BMW 3 Series.
# - **Auction dates:** Jan 2014 to Jul 2015 (359,688 sales, model year 2010+).
# - **Retail dates:** listings scraped Sep 2020, listed Oct 2013 to Sep 2020 (1,359,958 used listings).
# - **Days on market:** median 28, mean 52, 75th percentile 55. About 22k listings are older than a year; treat these as stale and exclude.
#- **Price vs MMR by condition (median):** poor 0.82, fair 0.96, good 1.00, excellent 1.02. Condition spread is widest for poor cars (std 0.165 vs 0.077 for excellent).
## Key finding that changes the plan
#The retail file is a snapshot of active listings: no row has an observed sale. Every days-on-market value is censored. Instead of a standard survival model, we estimate time to sale with **stock sampling**: if listings arrive at a steady rate, the share of the snapshot at age *a* is proportional to the probability a car is still unsold at age *a*. Limitation: 2020 inflow was not steady (COVID), which we state openly.
# %%
