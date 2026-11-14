"""
export_app_inputs.py: save what the deployed app needs in one small file.

The app only needs, for each of the 2,000 trade-ins, the numbers that don't
change between scenarios: its price ranges, time-to-sale scale, recon cost
range, condition band and a label. Saving those (a few hundred KB) means the
deployed app doesn't need the trained models or the multi-GB raw data.

Run from the project root after notebook 02 has saved the models:
    python -m src.export_app_inputs
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as C
from src import features as F
from src import simulate as S

OUT = C.ROOT / "app" / "app_inputs.npz"


def main():
    # Same trade-ins as notebook 03 (same split, same sample, same seed).
    auction = F.prep_auction(pd.read_parquet(C.PROCESSED / "auction.parquet"))
    _, test_idx = train_test_split(auction.index, test_size=0.2,
                                   random_state=C.RANDOM_STATE)
    cars = auction.loc[test_idx].sample(min(C.N_CARS, len(test_idx)),
                                        random_state=C.RANDOM_STATE)
    w, r, t = S.load_models()
    inp = S.prepare(cars, w, r, t)

    labels = np.array([f"{y:.0f} {mm.title()}, {mi:,.0f} mi, condition {c:.0f}"
                       for y, mm, mi, c in zip(cars["year"], cars["make_model"].astype(str),
                                               cars["miles"], cars["condition"])])
    np.savez_compressed(
        OUT,
        mmr=inp["mmr"], lam=inp["lam"], k=np.array(inp["k"]),
        recon_lo=inp["recon_lo"], recon_hi=inp["recon_hi"],
        wq=inp["wq"][["q10", "q50", "q90"]].to_numpy(),
        rq=inp["rq"][["q10", "q50", "q90"]].to_numpy(),
        band=np.asarray(inp["band"]).astype(str), labels=labels,
    )
    print(f"saved {OUT.relative_to(C.ROOT)} ({OUT.stat().st_size / 1024:.0f} KB) for {len(cars)} cars")


if __name__ == "__main__":
    main()
