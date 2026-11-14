# Retail or wholesale? Problem framing and findings

## The decision
A dealer takes in a used car. It can either:
- **Retail it:** recondition the car, list it, wait for a buyer, collect the retail price.
- **Wholesale it:** send it to auction as is and take the auction price.

For each car, recommend the option with the higher expected proceeds, and say how confident we are.

## The formula
**Retail proceeds** = retail sale price - reconditioning cost - transport to retail - (holding cost per day x days to sell)

**Wholesale proceeds** = auction price - auction fees

**Recommend retail when** expected retail proceeds > expected wholesale proceeds.

We compare proceeds, not profit: the price paid for the trade-in is the same for both options, so it cancels out. Prices, reconditioning cost and days to sell are all uncertain, so we simulate thousands of possible futures per car rather than plugging in one number each.

## What comes from data vs. assumptions
| Input | Source |
|---|---|
| Auction price range | Quantile model on auction sales (price as a share of MMR) |
| Retail price range | Quantile model on CarGurus listings |
| Days to sell distribution | Weibull model fitted to a snapshot of listings (stock sampling) |
| 2015 to 2020 price level | Manheim Used Vehicle Value Index (124 to 161.2) |
| Reconditioning cost | **Assumption** by condition band (`src/config.py`) |
| Holding cost per day | **Assumption** (`src/config.py`) |
| Transport, auction fees, retail discount | **Assumptions** (`src/config.py`) |

## Week 1 findings (data exploration)
- **Scope:** 20 make/model pairs with at least 4,400 auction sales and 4,900 retail listings each: Nissan Altima, Maxima, Rogue; Ford Fusion, Escape, F-150, Focus, Explorer, Edge; Toyota Camry, Corolla; Dodge Grand Caravan; Chevrolet Cruze, Malibu, Impala; Hyundai Sonata, Elantra; Honda Accord, Civic; BMW 3 Series.
- **Auction data:** Jan 2014 to Jul 2015 (359,688 sales, model year 2010+).
- **Retail data:** listings scraped Sep 2020, listed Oct 2013 to Sep 2020 (1,359,958 used listings).
- **Time gap:** about five years between datasets, with used car prices up about 30%. Prices are compared as ratios to a market benchmark and converted with the Manheim index.
- **Days on market:** median 28, mean 52, 75th percentile 55. About 22k listings are older than a year; treated as stale and excluded.
- **Condition matters at auction (median price vs MMR):** poor 0.82, fair 0.96, good 1.00, excellent 1.02. Spread is widest for poor cars (std 0.165 vs 0.077 for excellent).
- **Key data problem:** the retail file is a snapshot of active listings, so no row has an observed sale. Time to sale is estimated with **stock sampling**: if listings arrive at a steady rate, the share of the snapshot at age *a* is proportional to the probability a car is still unsold at age *a*.

## Week 2 findings (models)
- **Wholesale model:** 3.8% median error. 80% ranges hold in every condition band (77 to 78% before a small stretch). Slightly overvalues poor condition cars (predicted 0.83 vs actual 0.81 of MMR).
- **Retail model:** 7.6% median error; 80% coverage after a 1.02 stretch. Ranges are wide (about 32% of price) because trim, location and condition are not shared across datasets.
- **Stock sampling method validated** on simulated data with known answers (true k 1.30, estimated 1.29; true effect 0.80, estimated 0.80).
- **Time to sale:** a car priced at market sells in about 38 days on average. k = 0.85, so the chance of selling falls as a listing ages (aged inventory effect).
- **Pricing effect is not causal:** the raw model says overpriced cars sell faster. Listings start about 1.6% above market and drift to 2.7% below by 60+ days, consistent with markdowns on aging cars (reverse causality). The decision model uses baseline time to sale only (price gap = 0) and makes no pricing recommendations.

## Week 3 findings (decision engine)
- **Setup:** 2,000 held-out trade-ins x 2,000 simulated futures each; decision = option with higher average proceeds.
- **Sept 2020 market:** 25% of trade-ins sent to retail. The model beats the best simple rule (always auction) by $662 per car, capturing about 58% of the value a perfect-hindsight rule could capture ($1,134).
- **Reality check:** our retail listing minus auction price gap is $1,428 vs an industry benchmark of about $3,400 (vAuto, 178k cars tracked by VIN). This is consistent with the 2020 squeeze, when auction prices spiked and retail prices lagged. In a normal market (auction prices about 10% lower), the retail share rises to 51% (poor 62%, fair 60%, good 51%, excellent 39%).
- **Sensitivity:** auction price level matters most (retail share 36% to 17% across +/-5%), then holding cost per day and retail sale discount. Auction fees and transport barely matter.
- **Value of information:** knowing the exact recon cost is worth more than a $150 inspection for only 18% of poor condition cars and almost no others. Perfect information on everything is worth $440 to $600 per car, so price and time-on-lot uncertainty matters far more than recon uncertainty.

## Known limitations (state these openly in the README)
1. **Time gap:** the datasets come from different years; prices are bridged with ratios and the Manheim index.
2. **Unusual market:** the retail data is from September 2020, when auction prices were at record highs and retail lagged. Results are shown for both that market and a normal-market scenario.
3. **Listing price is not sale price:** CarGurus shows asking prices; we apply a 3% discount informed by the markdown check.
4. **No observed sales:** time to sale comes from stock sampling, which assumes a steady flow of new listings (2020 inflow was disrupted by COVID).
5. **Retail model does not see condition:** it treats every reconditioned car alike while auctions pay a premium for clean cars, so "better cars go to auction more" is an artifact, not a finding.
6. **Cost assumptions are placeholders:** reconditioning, holding, transport and fees are tested in the sensitivity analysis.
