# %% [markdown]
# # 02 Models: the three inputs to the decision
# For each trade-in we need three uncertain quantities:
# - **A. Wholesale price:** what it fetches at auction (depends on condition)
# - **B. Retail price:** what it lists for at a dealer after reconditioning
# - **C. Days to sell:** how long it sits on the lot (drives holding cost)
#
# Each model outputs a RANGE, not a single number, because Week 3 simulates
# thousands of possible outcomes per car. Run cells one at a time.

# %%
import sys
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src import config as C
from src import features as F
from src import models as M

auction = F.prep_auction(pd.read_parquet(C.PROCESSED / "auction.parquet"))
retail = F.prep_retail(pd.read_parquet(C.PROCESSED / "retail.parquet"))
C.MODELS_DIR.mkdir(exist_ok=True)
print(f"auction in scope: {len(auction):,}   retail in scope: {len(retail):,}")
print(f"index ratio (2015 $ -> 2020 $): {C.INDEX_RATIO:.3f}")

# %% [markdown]
# ## A. Wholesale model
# **Target:** `price_to_mmr` (sale price / market value). We model the RATIO,
# not dollars, because MMR already captures make, model, year and mileage.
# What's left to explain is mostly condition, which is exactly what the
# reconditioning decision is about.
#
# **Train/test split:** we fit on 80% of sales and grade on the other 20%.
# Grading on data the model has seen would flatter it.

# %%
Xw = auction[F.WHOLESALE_FEATURES]
yw = auction["price_to_mmr"]
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(
    Xw, yw, test_size=0.2, random_state=C.RANDOM_STATE)

wholesale_models = M.fit_quantiles(Xw_tr, yw_tr)
wpred = M.predict_quantiles(wholesale_models, Xw_te)
M.evaluate_quantiles(wpred, yw_te)
# What good looks like: coverage_80 close to 0.80. If it is 0.70, the
# ranges are too narrow and the simulation will be overconfident.

# %% [markdown]
# ### Calibrate the range
# If coverage came out below 0.80, stretch the ranges until it hits 0.80.
# We find the stretch factor on half of the test set and confirm it on the
# other half, so the check is honest. A factor of 1.0 means no change needed.

# %%
half = len(Xw_te) // 2
w_scale = M.calibrate_width(wpred.iloc[:half], yw_te.iloc[:half])
print("wholesale stretch factor:", w_scale)
M.evaluate_quantiles(M.widen(wpred.iloc[half:], w_scale), yw_te.iloc[half:])

# %% [markdown]
# ### Check: is the model honest for every condition band?
# Overall coverage can hide problems. If poor-condition cars had 60% coverage
# we'd underestimate their risk, which is the group the decision hinges on.

# %%
bands = pd.cut(Xw_te["condition"], [0, 19, 34, 44, 50],
               labels=list(C.RECON_COST_BY_CONDITION))
check = pd.DataFrame({"band": bands,
                      "inside": (yw_te >= wpred["q10"]) & (yw_te <= wpred["q90"]),
                      "pred_median": wpred["q50"], "actual": yw_te})
check.groupby("band", observed=True).agg(
    coverage=("inside", "mean"), pred_median=("pred_median", "median"),
    actual_median=("actual", "median"), n=("actual", "size"))

# %% [markdown]
# ## B. Retail model
# **Target:** log of the listing price. Logs make a $1,000 miss on a $8,000
# Civic count more than on a $40,000 F-150, which matches how errors hurt.
# **Features:** only ones we also know for an auction car (make/model, age,
# miles) plus dealer type, so we can price a trade-in later.

# %%
Xr = retail[F.RETAIL_FEATURES]
yr = np.log(retail["price"])
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    Xr, yr, test_size=0.2, random_state=C.RANDOM_STATE)

retail_models = M.fit_quantiles(Xr_tr, yr_tr)
rpred_log = M.predict_quantiles(retail_models, Xr_te)
rpred = np.exp(rpred_log)                 # back to dollars for grading
M.evaluate_quantiles(rpred, np.exp(yr_te))

# %%
# Same calibration as the wholesale model (done on the log scale).
half = len(Xr_te) // 2
r_scale = M.calibrate_width(rpred_log.iloc[:half], yr_te.iloc[:half])
print("retail stretch factor:", r_scale)
M.evaluate_quantiles(np.exp(M.widen(rpred_log.iloc[half:], r_scale)),
                     np.exp(yr_te.iloc[half:]))

# %% [markdown]
# ### Price gap: how each listing is priced vs the market
# For each listing, compare its price with the model's median for a car like
# it. `price_gap = +0.05` means listed 5% above market. We use out-of-fold
# predictions so each listing is judged by a model that never saw it.
# This takes a few minutes (3 model fits).

# %%
retail["market_log_price"] = M.out_of_fold_median(Xr, yr)
retail["price_gap"] = yr - retail["market_log_price"]
retail["price_gap"].describe()

# %% [markdown]
# ## C. Time to sale (stock sampling)
# Every listing in this file was still for sale on the scrape date, so we
# never see a sale. Trick: if cars are listed at a steady rate, the number of
# listings that are `a` days old is proportional to the chance a car stays
# unsold for `a` days. The age histogram IS the survival curve, rescaled.
#
# We assume time to sale follows a Weibull distribution (flexible: k > 1
# means the chance of selling rises over time, k < 1 means it falls) and let
# its scale depend on price gap, age, mileage and model.
#
# ### First, prove the method works
# Before trusting it on real data, simulate a market where we KNOW the answer
# (k = 1.3, a 10% overprice multiplies time on lot by exp(0.08) = 1.083).
# The simulation keeps about 34k listings, close to one model's worth of
# real data, so this also tells us how precise our real estimates can be and
# check the fit recovers it. Always validate a custom method this way.

# %%
Xs, ages_s = M.simulate_snapshot()
sim_fit = M.fit_stock_weibull(Xs, ages_s)
print(f"k:        true 1.30   estimated {sim_fit['k']:.2f}")
print(f"effect:   true 0.80   estimated {sim_fit['beta']['price_gap']:.2f}")
print(f"scale:    true 40.0   estimated {np.exp(sim_fit['beta']['intercept']):.1f}")

# %% [markdown]
# ### Fit on the real listings
# Features are standardized (age in years from 5, miles in units of 40k from
# 60k) so coefficients are comparable. Model dummies let each model have its
# own baseline speed. We fit on a 300k sample to keep it fast.

# %%
def survival_features(df):
    X = pd.DataFrame({
        "price_gap": df["price_gap"],
        "age_c": (df["age"] - 5) / 3,
        "miles_c": (df["miles"] - 60_000) / 40_000,
    }, index=df.index)
    dummies = pd.get_dummies(df["make_model"], prefix="m", drop_first=True, dtype=float)
    return pd.concat([X, dummies], axis=1)

sample = retail.sample(min(300_000, len(retail)), random_state=C.RANDOM_STATE)
Xt = survival_features(sample)
ttsale = M.fit_stock_weibull(Xt, sample["daysonmarket"])
print(f"k = {ttsale['k']:.2f}")
ttsale["beta"][["intercept", "price_gap", "age_c", "miles_c"]]

# %% [markdown]
# ### The business answer: days to sell vs pricing
# Set every car's price gap to the same value and average the expected days.
# Caveat to state in the README: this is an association, not a clean causal
# effect. Cars priced high may also be nicer in ways the data doesn't show.

# %%
rows = []
for gap in [-0.10, -0.05, 0.0, 0.05, 0.10]:
    Xg = Xt.copy(); Xg["price_gap"] = gap
    rows.append({"priced_vs_market": f"{gap:+.0%}",
                 "expected_days": M.expected_days(ttsale, Xg).mean()})
days_table = pd.DataFrame(rows)
days_table

# %%
t = np.arange(0, 181)
for gap in [-0.10, 0.0, 0.10]:
    Xg = Xt.iloc[:1].copy(); Xg[:] = 0; Xg["price_gap"] = gap
    lam = M.weibull_scale(ttsale, Xg)[0]
    plt.plot(t, np.exp(-(t / lam) ** ttsale["k"]), label=f"priced {gap:+.0%}")
plt.xlabel("days on lot"); plt.ylabel("share still unsold")
plt.title("Time to sale by pricing (baseline car)"); plt.legend()
plt.savefig(C.FIGURES / "survival_by_price.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Save everything for Week 3

# %%
joblib.dump({"models": wholesale_models, "stretch": w_scale},
            C.MODELS_DIR / "wholesale_quantiles.pkl")
joblib.dump({"models": retail_models, "stretch": r_scale},
            C.MODELS_DIR / "retail_quantiles.pkl")
joblib.dump({"fit": ttsale, "columns": list(Xt.columns)},
            C.MODELS_DIR / "time_to_sale.pkl")
print("saved:", [p.name for p in C.MODELS_DIR.glob("*.pkl")])

# %%
# Do older listings look cheaper? If the median gap falls as days on
# market rise, price cuts on aging cars explain the backwards sign.
bins = pd.cut(retail["daysonmarket"], [-1, 7, 30, 60, 120, 365])
retail.groupby(bins, observed=True)["price_gap"].agg(["median", "size"])

# %%
## Week 2 findings
# - **Wholesale model:** 3.8% median error; 80% ranges hold in every condition band (77 to 78% before a small stretch). Slightly overvalues poor condition cars (predicted 0.83 vs actual 0.81 of MMR).
# - **Retail model:** 7.6% median error; 80% coverage after a 1.02 stretch. Ranges are wide (about 32% of price) because trim, location and condition aren't shared across datasets.
# - **Stock sampling method:** validated on simulated data (true k 1.30, estimated 1.29; true effect 0.80, estimated 0.80).
# - **Time to sale:** baseline car priced at market sells in about 38 days on average. k = 0.85, so the chance of selling falls as a listing ages (aged inventory effect).
# - **Pricing effect is not causal:** the raw model says overpriced cars sell faster. Listings start about 1.6% above market and drift to 2.7% below by 60+ days, consistent with markdowns on aging cars (reverse causality). The decision model therefore uses baseline time to sale only (price gap = 0) and makes no pricing recommendations.
# %%
