# %% [markdown]
# # 03 Decision engine: retail or wholesale?
# For each trade-in we simulate 2,000 possible futures and compare:
#
#     wholesale = auction price - auction fees
#     retail    = retail sale price - recon cost - transport - holding cost x days on lot
#
# We compare **proceeds**, not profit. What the dealer paid for the trade-in is
# the same whichever option it picks, so it cancels out.
#
# Then two questions a decision scientist asks:
# 1. **Sensitivity:** which assumption, if wrong, flips the most decisions?
# 2. **Value of information:** what is an inspection worth before deciding?

# %%
import sys
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src import config as C
from src import features as F
from src import simulate as S

auction = F.prep_auction(pd.read_parquet(C.PROCESSED / "auction.parquet"))

# The trade-ins to decide on: cars from the 20% the wholesale model never
# trained on (same split as 02_models), so we don't grade it on its homework.
_, test_idx = train_test_split(auction.index, test_size=0.2,
                               random_state=C.RANDOM_STATE)
cars = auction.loc[test_idx].sample(min(C.N_CARS, len(test_idx)),
                                    random_state=C.RANDOM_STATE)

w, r, t = S.load_models()
inp = S.prepare(cars, w, r, t)
shocks = S.draw_shocks(len(cars), C.N_SIMS, inp["k"])
base = S.default_assumptions()
print(f"{len(cars):,} trade-ins x {C.N_SIMS:,} futures each")
base

# %% [markdown]
# ## 1. Run the simulation and decide
# For each car: average proceeds of each option, the chance retail beats
# wholesale, the decision, how confident it is, and the downside (5th
# percentile: in 1 future out of 20 the chosen option does worse than this).

# %%
sim = S.simulate(inp, shocks, base)
dec = S.decide(sim)
dec.index = cars.index
dec["band"] = pd.Categorical(inp["band"], categories=list(C.RECON_COST_BY_CONDITION))
print(f"sent to retail: {(dec['decision'] == 'retail').mean():.0%}")
dec.head()

# %% [markdown]
# ### Decisions by condition
# Week 1 hinted that reconditioning poor cars barely pays. Does the full model agree?

# %%
by_band = dec.groupby("band", observed=True).agg(
    cars=("decision", "size"),
    share_retail=("decision", lambda d: (d == "retail").mean()),
    avg_gain_from_retail=("exp_gain_from_retail", "mean"),
    avg_confidence=("confidence", "mean"),
)
by_band.round(2)

# %% [markdown]
# ## 2. Is the model worth using?
# Compare average proceeds per car against two simple rules a dealer might
# follow without a model. The gap is the model's value per car.
# "Perfect hindsight" picks the winner in every future: a ceiling no real
# rule can reach, useful to see how much room is left.

# %%
policies = S.policy_comparison(sim, dec)
policies.round(0)

# %%
best_simple = max(policies["always wholesale"], policies["always retail"])
lift = policies["decision model"] - best_simple
print(f"model beats the best simple rule by ${lift:,.0f} per car")
print(f"on 1,000 trade-ins a month that's ${lift * 12_000:,.0f} a year")

# %% [markdown]
# ### One car up close
# The closest call in the sample: the car whose chance of retail winning is
# nearest 50%. The histogram shows every simulated future for it.

# %%
i = (dec["p_retail_better"] - 0.5).abs().to_numpy().argmin()
row = cars.iloc[i]
gain = sim["retail"][i] - sim["wholesale"][i]
print(f"{row['year']:.0f} {row['make_model']}, {row['miles']:,.0f} miles, condition {row['condition']:.0f}")
print(f"P(retail better) = {(gain > 0).mean():.0%}, average gain from retail = ${gain.mean():,.0f}")
plt.hist(gain, bins=60)
plt.axvline(0, color="black", lw=1)
plt.xlabel("retail minus wholesale proceeds ($)"); plt.ylabel("simulated futures")
plt.title("One trade-in, 2,000 possible futures")
plt.savefig(C.FIGURES / "one_car_futures.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## 3. Sensitivity analysis
# Every cost assumption is a placeholder. Which ones actually matter?
# We push each to a low and high value (keeping the others at base) and watch
# the share of cars sent to retail. Long bars = assumptions worth getting
# right; short bars = don't bother refining them.

# %%
base_share = (dec["decision"] == "retail").mean()
tornado = S.sensitivity(inp, shocks, base)
tornado

# %%
fig, ax = plt.subplots(figsize=(8, 4))
y = np.arange(len(tornado))
lo = tornado["retail_share_low"] - base_share
hi = tornado["retail_share_high"] - base_share
ax.barh(y, lo, color="#4C78A8", label="low value")
ax.barh(y, hi, color="#F58518", label="high value")
ax.axvline(0, color="black", lw=1)
ax.set_yticks(y); ax.set_yticklabels(tornado["assumption"].str.replace("$", r"\$", regex=False))
ax.set_xlabel(f"change in share sent to retail (base = {base_share:.0%})")
ax.set_title("Which assumptions move the decision?")
ax.legend(loc="lower right")
plt.savefig(C.FIGURES / "tornado.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ### Reality check: is the retail vs auction price gap realistic?
# The decision hinges on the gap between what a car lists for at retail and
# what it fetches at auction. Industry benchmark: dealers see about $3,400
# between auction purchase price and retail asking price (vAuto, 178k cars
# tracked by VIN). Our data comes from September 2020, when auction prices
# had just spiked to records and retail prices had not caught up, so our gap
# may be unusually thin. Compare, then rerun the decisions in a "normal
# market" where the gap matches the benchmark.

# %%
listing = sim["retail_price"] / (1 - base["retail_discount"])
spread_now = (listing - sim["auction_price"]).mean()
print(f"our gap (retail listing - auction price): ${spread_now:,.0f}")

BENCHMARK_SPREAD = 3_400
normal = dict(base, wholesale_scale=(listing.mean() - BENCHMARK_SPREAD) / sim["auction_price"].mean())
dec_normal = S.decide(S.simulate(inp, shocks, normal))
dec_normal["band"] = pd.Categorical(inp["band"], categories=list(C.RECON_COST_BY_CONDITION))
print(f"auction price level for a normal market: {normal['wholesale_scale']:.3f}x")
print(f"sent to retail, Sept 2020 market: {(dec['decision'] == 'retail').mean():.0%}")
print(f"sent to retail, normal market:    {(dec_normal['decision'] == 'retail').mean():.0%}")
pd.DataFrame({
    "sept_2020_market": dec.groupby("band", observed=True)["decision"].apply(lambda d: (d == "retail").mean()),
    "normal_market": dec_normal.groupby("band", observed=True)["decision"].apply(lambda d: (d == "retail").mean()),
}).round(2)

# %%
# The headline chart: how the best decision shifts with the market.
regime = pd.DataFrame({
    "Sept 2020 (auction prices at record highs)":
        dec.groupby("band", observed=True)["decision"].apply(lambda d: (d == "retail").mean()),
    "Normal market":
        dec_normal.groupby("band", observed=True)["decision"].apply(lambda d: (d == "retail").mean()),
})
ax = (regime * 100).plot.bar(color=["#F58518", "#4C78A8"], figsize=(8, 4.5), rot=0)
ax.set_ylabel("% of trade-ins sent to retail"); ax.set_xlabel("condition")
ax.set_title("When auction prices spike, more trade-ins should go to auction")
ax.set_ylim(0, min(100, regime.to_numpy().max() * 100) * 1.3)
ax.legend(loc="upper center", ncol=2, fontsize=9, frameon=False)
for c in ax.containers:
    ax.bar_label(c, fmt="%.0f%%", fontsize=9)
plt.savefig(C.FIGURES / "market_regime.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## 4. Value of information: is an inspection worth it?
# Before deciding, the dealer could pay for a detailed inspection that
# reveals the exact recon cost. That only has value if it could CHANGE the
# decision. For a car that's clearly a retail winner, knowing the exact
# recon cost changes nothing, so the inspection is worth $0.
# For each car we compute the most a rational dealer should pay to know.

# %%
dec["value_of_inspection"] = S.value_of_recon_info(sim)
dec["value_of_perfect_info"] = S.value_of_perfect_info(sim)
dec["miles_band"] = pd.cut(cars["miles"], [0, 30_000, 60_000, 100_000, 200_000],
                           labels=["<30k", "30-60k", "60-100k", "100k+"])

voi = dec.groupby("band", observed=True).agg(
    avg_value_of_inspection=("value_of_inspection", "mean"),
    share_worth_inspecting=("value_of_inspection",
                            lambda v: (v > C.INSPECTION_COST).mean()),
    avg_value_of_perfect_info=("value_of_perfect_info", "mean"),
)
voi.round(2)

# %%
dec.groupby("miles_band", observed=True)["value_of_inspection"].agg(
    ["mean", lambda v: (v > C.INSPECTION_COST).mean()]
).rename(columns={"mean": "avg_value", "<lambda_0>": "share_worth_inspecting"}).round(2)

# %%
voi["avg_value_of_inspection"].plot.bar(color="#4C78A8")
plt.axhline(C.INSPECTION_COST, color="black", ls="--", lw=1,
            label=f"inspection cost (${C.INSPECTION_COST})")
plt.ylabel("$ per car"); plt.xlabel("condition band"); plt.xticks(rotation=0)
plt.title("What knowing the exact recon cost is worth")
plt.legend()
plt.savefig(C.FIGURES / "value_of_inspection.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Save the decisions for the app and the write-up

# %%
out = cars[["year", "make_model", "miles", "condition", "mmr_2020"]].join(dec)
out.to_parquet(C.PROCESSED / "decisions.parquet")
tornado.to_csv(C.ROOT / "reports" / "sensitivity.csv", index=False)
print("saved decisions for", len(out), "cars")

# %%
