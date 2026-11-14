"""
config.py: every assumption in the project lives here, in one place.

Why: the decision model depends on numbers we do not observe in the data
(reconditioning cost, holding cost per day). Keeping them in one file means
the sensitivity analysis in Week 3 can change them easily, and a reviewer
can see every assumption at a glance.

TODO before publishing: replace each PLACEHOLDER with a value you can cite
(industry article, dealer survey, NADA/auction reports) and add the source.
"""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

RETAIL_CSV = RAW / "used_cars_data.csv"   # CarGurus file (~10 GB)
AUCTION_CSV = RAW / "car_prices.csv"      # auction file (~85 MB)

# ---------- Scope ----------
# Keep the project manageable: recent model years, used cars only,
# and the most common make/model pairs (chosen in 01_explore).
MIN_MODEL_YEAR = 2010
TOP_N_MODELS = 20
MAX_MILEAGE = 200_000
PRICE_RANGE = (2_000, 80_000)

MODELS_DIR = ROOT / "models"

# The 20 make/model pairs chosen in 01_explore (enough data in both files).
SCOPE = [
    "nissan altima", "ford fusion", "ford escape", "ford f-150", "toyota camry",
    "ford focus", "dodge grand caravan", "chevrolet cruze", "hyundai sonata",
    "hyundai elantra", "toyota corolla", "chevrolet malibu", "nissan rogue",
    "ford explorer", "ford edge", "nissan maxima", "honda accord",
    "chevrolet impala", "bmw 3 series", "honda civic",
]

# ---------- Bridging the time gap ----------
# Manheim Used Vehicle Value Index (wholesale prices, mix and mileage adjusted).
# Auction data: Jan 2014 to Jul 2015, index ~124 (123.9 at Dec 2014).
# Retail data scraped Sep 2020, index 161.2.
# Source: Cox Automotive Manheim index releases (Jan 2015, Oct 2020).
INDEX_AUCTION = 124.0
INDEX_RETAIL = 161.2
INDEX_RATIO = INDEX_RETAIL / INDEX_AUCTION   # ~1.30: multiply 2015 $ to get 2020 $
SCRAPE_DATE = "2020-09-13"                   # last listing date in the retail file

# ---------- Modeling ----------
QUANTILES = [0.1, 0.5, 0.9]   # 80% interval plus the median
MAX_DOM = 365                 # listings older than a year are stale; excluded
RANDOM_STATE = 42

# ---------- Decision assumptions (PLACEHOLDERS) ----------
# Reconditioning cost by condition band (auction condition on a 1 to 50 scale).
# Each entry is (low, high) in dollars; Week 3 samples uniformly between them.
RECON_COST_BY_CONDITION = {
    "poor (1-19)":      (2_500, 5_000),   # PLACEHOLDER
    "fair (20-34)":     (1_200, 2_500),   # PLACEHOLDER
    "good (35-44)":     (500, 1_200),     # PLACEHOLDER
    "excellent (45+)":  (200, 600),       # PLACEHOLDER
}

HOLDING_COST_PER_DAY = 35       # PLACEHOLDER: floorplan interest + lot + depreciation
TRANSPORT_TO_RETAIL = 300       # PLACEHOLDER: moving the car to a retail location
AUCTION_FEES = 400              # PLACEHOLDER: seller fee + transport to auction
# Listing price is not sale price. Week 2 showed listings drift from about
# +1.6% to -2.7% vs market as they age (markdowns), so assume the car finally
# sells ~3% below its first listing price.
RETAIL_SALE_DISCOUNT = 0.03     # PLACEHOLDER, informed by the Week 2 markdown check
INSPECTION_COST = 150           # PLACEHOLDER: cost of a detailed pre-decision inspection

# ---------- Simulation ----------
N_CARS = 2_000     # trade-ins to decide on (sampled from held-out auction cars)
N_SIMS = 2_000     # possible futures simulated per car
