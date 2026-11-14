"""
Retail or auction? An interactive what-if tool for the trade-in decision model.

Run from the project root:
    streamlit run app/streamlit_app.py

Reads app/app_inputs.npz (made by `python -m src.export_app_inputs`), so it
runs without the trained models or raw data. Move the sliders to change the
market and cost assumptions; every trade-in is re-simulated (2,000 futures
each) and the decisions update.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src import simulate as S

st.set_page_config(page_title="Retail or auction?", layout="wide")

REPO = "https://github.com/SmrithiChembath/trade-in-decision"


def money(x):
    """-117 -> '-$117' (instead of '$-117')."""
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


@st.cache_resource(show_spinner="Setting up 2,000 trade-ins x 2,000 futures...")
def setup():
    """Load the precomputed per-car inputs and draw the random futures once.
    float32 halves memory so the app fits comfortably on a free host."""
    d = np.load(ROOT / "app" / "app_inputs.npz")
    cols = ["q10", "q50", "q90"]
    f32 = lambda a: a.astype(np.float32)
    inp = {
        "mmr": f32(d["mmr"]), "lam": f32(d["lam"]), "k": float(d["k"]),
        "recon_lo": f32(d["recon_lo"]), "recon_hi": f32(d["recon_hi"]),
        "wq": pd.DataFrame(f32(d["wq"]), columns=cols),
        "rq": pd.DataFrame(f32(d["rq"]), columns=cols),
        "band": d["band"],
    }
    shocks = {key: f32(v) for key, v in
              S.draw_shocks(len(d["labels"]), C.N_SIMS, inp["k"]).items()}
    base_dec = S.decide(S.simulate(inp, shocks, S.default_assumptions()))
    closest = int((base_dec["p_retail_better"] - 0.5).abs().to_numpy().argmin())
    return inp, shocks, list(d["labels"]), closest


inp, shocks, labels, closest = setup()
bands = pd.Categorical(inp["band"], categories=list(C.RECON_COST_BY_CONDITION))

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Market and cost assumptions")
market = st.sidebar.slider("Auction price level (1.00 = Sept 2020)", 0.85, 1.05, 1.00, 0.01)
st.sidebar.caption("Sept 2020 auction prices were at record highs. About 0.90 "
                   "matches a normal retail vs auction gap (~$3,400).")
holding = st.sidebar.slider("Holding cost per day ($)", 20, 60, C.HOLDING_COST_PER_DAY, 5)
recon = st.sidebar.slider("Reconditioning cost (x base)", 0.5, 1.5, 1.0, 0.1)
discount = st.sidebar.slider("Retail sale discount vs listing", 0.0, 0.08,
                             C.RETAIL_SALE_DISCOUNT, 0.01, format="%.2f")
st.sidebar.markdown(f"[Code and write-up on GitHub]({REPO})")

a = dict(S.default_assumptions(), wholesale_scale=market, holding_per_day=holding,
         recon_scale=recon, retail_discount=discount)
sim = S.simulate(inp, shocks, a)
dec = S.decide(sim)
policies = S.policy_comparison(sim, dec)

# ---------------------------------------------------------------- portfolio view
st.title("Retail or auction? A decision model for used car trade-ins")
st.write("Each trade-in is simulated 2,000 times (auction price, retail price, "
         "reconditioning cost, days on the lot). The model picks the option with "
         "higher expected proceeds. Try moving the auction price level to 0.90.")

best_simple = max(policies["always wholesale"], policies["always retail"])
c1, c2, c3 = st.columns(3)
c1.metric("Trade-ins sent to retail", f"{(dec['decision'] == 'retail').mean():.0%}")
c2.metric("Gain vs best simple rule", money(policies["decision model"] - best_simple) + " per car")
c3.metric("Avg proceeds per car", money(policies["decision model"]))

by_band = (pd.Series(dec["decision"].eq("retail").to_numpy(), index=bands)
           .groupby(level=0, observed=True).mean() * 100).rename("% sent to retail")
st.subheader("Share sent to retail, by condition")
fig0, ax0 = plt.subplots(figsize=(8, 3))
ax0.bar(by_band.index.astype(str), by_band.to_numpy(), color="#4C78A8")
ax0.set_ylabel("% sent to retail"); ax0.set_ylim(0, 100)
for k, v in enumerate(by_band.to_numpy()):
    ax0.text(k, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
st.pyplot(fig0)
plt.close(fig0)

# ---------------------------------------------------------------- one car
st.subheader("One trade-in up close")
i = st.selectbox("Pick a car", range(len(labels)), index=closest,
                 format_func=lambda k: labels[k])
gain = sim["retail"][i] - sim["wholesale"][i]
row = dec.iloc[i]
d1, d2, d3, d4 = st.columns(4)
d1.metric("Decision", "Retail" if row["decision"] == "retail" else "Auction")
d2.metric("Chance retail wins", f"{row['p_retail_better']:.0%}")
d3.metric("Expected gain from retail", money(row["exp_gain_from_retail"]))
d4.metric("Bad case (5th pct) proceeds", money(row["p5_chosen"]))

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.hist(gain, bins=60, color="#4C78A8")
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("retail minus auction proceeds ($)")
ax.set_ylabel("simulated futures")
st.pyplot(fig)
plt.close(fig)
st.caption("Right of the line: futures where retailing wins. The long left tail "
           "is cars that sit on the lot for months.")
