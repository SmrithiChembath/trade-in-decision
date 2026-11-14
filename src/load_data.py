"""
load_data.py: turn the two raw CSVs into small, clean parquet files.

Run from the project root:
    python -m src.load_data

What it does and why:
1. The CarGurus file is about 10 GB, too big to load at once on a laptop.
   We read it in chunks of 250k rows, keep only the columns we need, filter
   each chunk, and stack the survivors. Memory stays low the whole time.
2. The auction file is small, so we load it in one go and clean it.
3. Both are saved as parquet: a compressed, typed format that loads in
   seconds instead of minutes, so later notebooks start fast.
"""
import pandas as pd

from src import config as C

# Columns we keep from the CarGurus file. Run peek_columns() first: if a name
# differs in your download, fix it here.
RETAIL_COLS = [
    "vin", "make_name", "model_name", "trim_name", "year", "mileage",
    "price", "daysonmarket", "listed_date", "is_new", "body_type",
    "franchise_dealer", "city", "dealer_zip", "latitude", "longitude",
]


def peek_columns(path=C.RETAIL_CSV, n=5):
    """Read only 5 rows so you can see the real column names cheaply."""
    df = pd.read_csv(path, nrows=n, low_memory=False)
    print(f"{len(df.columns)} columns:\n", list(df.columns))
    return df


def load_retail(chunksize=250_000):
    """Chunked read + filter of the CarGurus listings."""
    kept = []
    # Read ID-like and text columns as strings so every chunk has the same
    # type. Otherwise pandas guesses per chunk (number in one, text in the
    # next) and parquet refuses mixed types.
    text_cols = {c: "string" for c in ["vin", "dealer_zip", "city", "trim_name",
                                       "body_type", "is_new", "franchise_dealer"]}
    reader = pd.read_csv(C.RETAIL_CSV, usecols=RETAIL_COLS, dtype=text_cols,
                         chunksize=chunksize, low_memory=False)
    for i, chunk in enumerate(reader):
        # Used cars only: new cars are priced by the manufacturer, not the market.
        chunk = chunk[chunk["is_new"].astype(str).str.lower() != "true"]
        # Mileage sometimes arrives as text; coerce bad values to NaN.
        chunk["mileage"] = pd.to_numeric(chunk["mileage"], errors="coerce")
        chunk = chunk[
            (chunk["year"] >= C.MIN_MODEL_YEAR)
            & chunk["price"].between(*C.PRICE_RANGE)
            & chunk["mileage"].between(0, C.MAX_MILEAGE)
            & chunk["daysonmarket"].ge(0)
        ]
        kept.append(chunk)
        print(f"chunk {i}: kept {len(chunk):,} rows")
    df = pd.concat(kept, ignore_index=True)
    df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce")
    df["franchise_dealer"] = df["franchise_dealer"].str.lower().eq("true")
    df = df.rename(columns={"make_name": "make", "model_name": "model",
                            "trim_name": "trim"})
    return clean_text(df)


def load_auction():
    """Load and clean the auction sales file."""
    df = pd.read_csv(C.AUCTION_CSV)
    df = df.dropna(subset=["make", "model", "year", "odometer",
                           "condition", "mmr", "sellingprice", "saledate"])

    # Condition comes on two scales: most rows are 1 to 49, a few are 1 to 5.
    # We assume the small ones are the 1-5 scale and multiply by 10 so every
    # row is on the same 1-50 scale. Note this assumption in the framing doc.
    df["condition"] = df["condition"].where(df["condition"] > 5,
                                            df["condition"] * 10)

    # saledate looks like "Tue Dec 16 2014 12:30:00 GMT-0800 (PST)".
    # The first 15 characters are the date part, which parses cleanly.
    df["saledate"] = pd.to_datetime(df["saledate"].str[:15],
                                    format="%a %b %d %Y", errors="coerce")

    df = df[
        (df["year"] >= C.MIN_MODEL_YEAR)
        & df["sellingprice"].between(*C.PRICE_RANGE)
        & df["odometer"].between(0, C.MAX_MILEAGE)
    ]
    # Price relative to the auction's own market value (MMR). This ratio is
    # how we compare across years later: 1.05 means it sold 5% above MMR.
    df["price_to_mmr"] = df["sellingprice"] / df["mmr"]
    df = df[df["price_to_mmr"].between(0.3, 2.0)]  # drop obvious data errors
    return clean_text(df)


def clean_text(df):
    """Same casing everywhere so 'Ford' in one file matches 'ford' in the other."""
    for col in ["make", "model", "trim"]:
        if col in df:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df


if __name__ == "__main__":
    C.PROCESSED.mkdir(parents=True, exist_ok=True)

    auction = load_auction()
    auction.to_parquet(C.PROCESSED / "auction.parquet", index=False)
    print(f"auction: {len(auction):,} rows saved")

    retail = load_retail()
    retail.to_parquet(C.PROCESSED / "retail.parquet", index=False)
    print(f"retail: {len(retail):,} rows saved")
