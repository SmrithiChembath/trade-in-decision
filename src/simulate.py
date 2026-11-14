"""
simulate.py: the decision engine.

For each trade-in we simulate N_SIMS possible futures. In each future we draw
an auction price, a retail price, a reconditioning cost and a number of days
on the lot, then compute what each option would earn:

    wholesale = auction price - auction fees
    retail    = retail sale price - recon cost - transport - holding cost x days

We compare PROCEEDS, not profit. The price the dealer paid for the trade-in
is the same whichever option it picks, so it cancels out of the comparison.

Arrays are shaped (n_cars, n_sims): one row per car, one column per future.
"""
import joblib
import numpy as np
import pandas as pd

from src import config as C
from src import features as F
from src import models as M

Z90 = 1.2816   # a standard normal is below -1.2816 10% of the time


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_models():
    w = joblib.load(C.MODELS_DIR / "wholesale_quantiles.pkl")
    r = joblib.load(C.MODELS_DIR / "retail_quantiles.pkl")
    t = joblib.load(C.MODELS_DIR / "time_to_sale.pkl")
    return w, r, t


def split_normal(q, z):
    """Turn three quantiles into random draws.
    Below the median we use a normal curve that puts 10% below q10; above it,
    one that puts 10% above q90. This keeps skew: if the downside range is
    wider than the upside (common for poor cars), the draws reflect that.
    q: DataFrame with q10, q50, q90 (one row per car). z: standard normal draws."""
    q10, q50, q90 = (q[c].to_numpy()[:, None] for c in ("q10", "q50", "q90"))
    sd_lo = (q50 - q10) / Z90
    sd_hi = (q90 - q50) / Z90
    return np.where(z < 0, q50 + z * sd_lo, q50 + z * sd_hi)


def condition_band(condition):
    return pd.cut(condition, [0, 19, 34, 44, 50],
                  labels=list(C.RECON_COST_BY_CONDITION))


def time_to_sale_scale(tts, cars):
    """Weibull scale for each car, priced AT market (price_gap = 0).
    We don't use the price gap effect: Week 2 showed it reflects markdowns,
    not a causal effect of pricing."""
    X = pd.DataFrame(0.0, index=cars.index, columns=tts["columns"])
    X["age_c"] = (cars["age"] - 5) / 3
    X["miles_c"] = (cars["miles"] - 60_000) / 40_000
    mm = cars["make_model"].astype(str)
    for model in mm.unique():
        col = f"m_{model}"
        if col in X:                      # the baseline model has no column
            X.loc[mm == model, col] = 1.0
    return M.weibull_scale(tts["fit"], X)


def prepare(cars, w, r, t):
    """Everything about each car that does NOT change between futures."""
    wq = M.widen(M.predict_quantiles(w["models"], cars[F.WHOLESALE_FEATURES]),
                 w["stretch"])
    # Price the trade-in as a car on a franchise dealer's lot.
    rX = cars[["make_model", "age", "miles"]].assign(franchise_dealer=1)
    rq = M.widen(M.predict_quantiles(r["models"], rX[F.RETAIL_FEATURES]),
                 r["stretch"])
    band = condition_band(cars["condition"])
    lo = band.map({k: v[0] for k, v in C.RECON_COST_BY_CONDITION.items()})
    hi = band.map({k: v[1] for k, v in C.RECON_COST_BY_CONDITION.items()})
    return {
        "mmr": cars["mmr_2020"].to_numpy()[:, None],
        "wq": wq, "rq": rq,
        "lam": time_to_sale_scale(t, cars)[:, None],
        "k": t["fit"]["k"],
        "recon_lo": lo.astype(float).to_numpy()[:, None],
        "recon_hi": hi.astype(float).to_numpy()[:, None],
        "band": band.to_numpy(),
    }


def draw_shocks(n_cars, n_sims, k, seed=C.RANDOM_STATE):
    """All the randomness, drawn ONCE. Every scenario in the sensitivity
    analysis reuses these same draws ("common random numbers"), so when a
    result changes it's because the assumption changed, not luck."""
    rng = np.random.default_rng(seed)
    shape = (n_cars, n_sims)
    return {
        "z_w": rng.standard_normal(shape),
        "z_r": rng.standard_normal(shape),
        "u_recon": rng.random(shape),
        "e_t": rng.weibull(k, shape),
    }


def default_assumptions():
    return {
        "holding_per_day": C.HOLDING_COST_PER_DAY,
        "transport": C.TRANSPORT_TO_RETAIL,
        "auction_fees": C.AUCTION_FEES,
        "retail_discount": C.RETAIL_SALE_DISCOUNT,
        "recon_scale": 1.0,       # multiply all recon costs (for sensitivity)
        "wholesale_scale": 1.0,   # shift all auction prices (time gap uncertainty)
    }


# ---------------------------------------------------------------------------
# Simulation and decision
# ---------------------------------------------------------------------------
def simulate(inp, shocks, a):
    auction_price = inp["mmr"] * split_normal(inp["wq"], shocks["z_w"]) * a["wholesale_scale"]
    wholesale = auction_price - a["auction_fees"]

    retail_price = np.exp(split_normal(inp["rq"], shocks["z_r"])) * (1 - a["retail_discount"])
    recon = (inp["recon_lo"] + shocks["u_recon"] * (inp["recon_hi"] - inp["recon_lo"])) * a["recon_scale"]
    days = inp["lam"] * shocks["e_t"]
    retail = retail_price - recon - a["transport"] - a["holding_per_day"] * days

    return {"wholesale": wholesale, "retail": retail, "recon": recon,
            "days": days, "retail_price": retail_price, "auction_price": auction_price}


def decide(sim):
    """Pick the option with the higher AVERAGE proceeds, and report how sure
    we are (share of futures where that option really was better) and the
    downside (5th percentile of the chosen option's proceeds)."""
    gain = sim["retail"] - sim["wholesale"]
    out = pd.DataFrame({
        "exp_retail": sim["retail"].mean(axis=1),
        "exp_wholesale": sim["wholesale"].mean(axis=1),
        "exp_gain_from_retail": gain.mean(axis=1),
        "p_retail_better": (gain > 0).mean(axis=1),
    })
    go_retail = out["exp_gain_from_retail"].to_numpy() > 0
    out["decision"] = np.where(go_retail, "retail", "wholesale")
    out["confidence"] = np.where(go_retail, out["p_retail_better"], 1 - out["p_retail_better"])
    chosen = np.where(go_retail[:, None], sim["retail"], sim["wholesale"])
    out["p5_chosen"] = np.percentile(chosen, 5, axis=1)
    return out


def policy_comparison(sim, dec):
    """Average proceeds per car under different rules. 'Perfect hindsight'
    picks the better option in every future: the ceiling no rule can beat."""
    go_retail = (dec["decision"] == "retail").to_numpy()
    model = np.where(go_retail, sim["retail"].mean(axis=1), sim["wholesale"].mean(axis=1))
    return pd.Series({
        "always wholesale": sim["wholesale"].mean(),
        "always retail": sim["retail"].mean(),
        "decision model": model.mean(),
        "perfect hindsight (ceiling)": np.maximum(sim["retail"], sim["wholesale"]).mean(),
    }, name="avg proceeds per car ($)")


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------
SENSITIVITY = {
    # label: (assumption key, low value, high value)
    "Holding cost per day ($20 to $50)": ("holding_per_day", 20, 50),
    "Recon cost (70% to 130%)": ("recon_scale", 0.7, 1.3),
    "Auction fees ($250 to $600)": ("auction_fees", 250, 600),
    "Transport to retail ($150 to $500)": ("transport", 150, 500),
    "Retail sale discount (0% to 6%)": ("retail_discount", 0.0, 0.06),
    "Auction price level (-5% to +5%)": ("wholesale_scale", 0.95, 1.05),
}


def sensitivity(inp, shocks, base):
    """Move one assumption at a time to its low and high value, keep the rest
    at base, and record the share of cars sent to retail."""
    rows = []
    for label, (key, lo, hi) in SENSITIVITY.items():
        share = {}
        for side, val in (("low", lo), ("high", hi)):
            a = dict(base)
            a[key] = val
            share[side] = (decide(simulate(inp, shocks, a))["decision"] == "retail").mean()
        rows.append({"assumption": label, "retail_share_low": share["low"],
                     "retail_share_high": share["high"],
                     "swing": abs(share["high"] - share["low"])})
    return pd.DataFrame(rows).sort_values("swing").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Value of information
# ---------------------------------------------------------------------------
def value_of_recon_info(sim):
    """How much is it worth to know a car's exact recon cost BEFORE deciding?

    gain = retail - wholesale = (everything else) - recon cost.
    Let m = average of (everything else). Recon cost is independent of the
    rest, so if an inspection revealed recon = c, the expected gain would be
    m - c and we'd pick retail only when m - c > 0.

        value WITH info    = average over c of max(m - c, 0)
        value WITHOUT info = max(m - average c, 0)

    (both measured relative to wholesale). The difference is the most a
    rational dealer should pay for the inspection on that car."""
    rest = sim["retail"] + sim["recon"] - sim["wholesale"]
    m = rest.mean(axis=1, keepdims=True)
    c = sim["recon"]
    with_info = np.maximum(m - c, 0).mean(axis=1)
    without_info = np.maximum(m[:, 0] - c.mean(axis=1), 0)
    return np.maximum(with_info - without_info, 0)   # clip float noise below 0


def value_of_perfect_info(sim):
    """Upper bound: the value of knowing EVERYTHING (prices, days, recon)."""
    gain = sim["retail"] - sim["wholesale"]
    return np.maximum(np.maximum(gain, 0).mean(axis=1) - np.maximum(gain.mean(axis=1), 0), 0)
