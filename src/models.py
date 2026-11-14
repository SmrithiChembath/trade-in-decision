"""
models.py: the three models that feed the decision.

1. Wholesale: what share of market value (MMR) a car fetches at auction,
   given its condition. Quantile gradient boosting -> a range, not one number.
2. Retail: what the car lists for at a dealer. Quantile gradient boosting on log price.
3. Time to sale: Weibull model fitted to a SNAPSHOT of live listings
   (stock sampling), because no listing in the data has an observed sale.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import minimize
from scipy.special import gammaln, gamma
from sklearn.model_selection import KFold

from src import config as C

# scikit-learn's histogram gradient boosting: same family as LightGBM (bins
# each feature, grows trees on the bins), but needs no system libraries.
# categorical_features="from_dtype" treats pandas category columns
# (make_model) as categories rather than numbers.
GB_PARAMS = dict(
    max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
    min_samples_leaf=50, early_stopping=False,
    categorical_features="from_dtype", random_state=C.RANDOM_STATE,
)


def _quantile_model(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, **GB_PARAMS)


# ---------------------------------------------------------------------------
# Quantile models (wholesale and retail)
# ---------------------------------------------------------------------------
def fit_quantiles(X, y, quantiles=C.QUANTILES):
    """One gradient boosting model per quantile. Quantile loss makes the 0.1 model aim for a
    value that 10% of real outcomes fall below, the 0.9 model 90%, etc."""
    models = {}
    for q in quantiles:
        m = _quantile_model(q)
        m.fit(X, y)
        models[q] = m
        print(f"  fitted q={q}")
    return models


def predict_quantiles(models, X):
    """Predict every quantile, then sort each row. Separate models can
    occasionally 'cross' (q10 above q50); sorting fixes that."""
    preds = np.column_stack([models[q].predict(X) for q in sorted(models)])
    preds = np.sort(preds, axis=1)
    return pd.DataFrame(preds, index=X.index,
                        columns=[f"q{int(q*100)}" for q in sorted(models)])


def evaluate_quantiles(pred, y_true):
    """The two checks that matter for a decision model:
    - coverage: share of real values inside [q10, q90]. Target ~80%.
    - median error: how far the typical prediction is from the truth."""
    inside = (y_true >= pred["q10"]) & (y_true <= pred["q90"])
    mape = (np.abs(pred["q50"] - y_true) / y_true).median()
    return {"coverage_80": inside.mean(), "median_abs_pct_error": mape,
            "avg_interval_width_pct": ((pred["q90"] - pred["q10"]) / pred["q50"]).mean()}


def calibrate_width(pred, y_true, target=0.80):
    """Quantile models often produce ranges that are a bit too narrow.
    Find the factor s that stretches each interval around its median
    (q50 +- s * distance to q10/q90) so that `target` of real values fall
    inside. Fit s on one half of the test set, check it on the other half."""
    lo = pred["q50"] - pred["q10"]
    hi = pred["q90"] - pred["q50"]
    for s in np.arange(1.0, 3.0, 0.01):
        inside = (y_true >= pred["q50"] - s * lo) & (y_true <= pred["q50"] + s * hi)
        if inside.mean() >= target:
            return round(float(s), 2)
    return 3.0


def widen(pred, s):
    out = pred.copy()
    out["q10"] = pred["q50"] - s * (pred["q50"] - pred["q10"])
    out["q90"] = pred["q50"] + s * (pred["q90"] - pred["q50"])
    return out


def out_of_fold_median(X, y, n_splits=3):
    """Median predictions where each row is predicted by a model that never
    saw it. In-sample predictions would look too accurate and shrink the
    'priced above or below market' signal we need for time to sale."""
    oof = pd.Series(np.nan, index=X.index)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=C.RANDOM_STATE)
    for i, (tr, te) in enumerate(kf.split(X)):
        m = _quantile_model(0.5)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof.iloc[te] = m.predict(X.iloc[te])
        print(f"  fold {i + 1}/{n_splits} done")
    return oof


# ---------------------------------------------------------------------------
# Time to sale from a snapshot (stock sampling)
# ---------------------------------------------------------------------------
# If cars are listed at a steady rate and each stays T days, then in a
# snapshot the number of listings aged a days is proportional to P(T > a),
# the survival function. So ages in a snapshot have density
#     g(a) = S(a) / E[T]
# With a Weibull time to sale, S(a) = exp(-(a/lam)^k) and
# E[T] = lam * Gamma(1 + 1/k). We let lam depend on car features:
#     log(lam_i) = X_i @ beta
# and fit k and beta by maximum likelihood on the snapshot ages.

def _neg_loglik(theta, X, a):
    log_k, beta = theta[0], theta[1:]
    k = np.exp(log_k)
    log_lam = X @ beta
    z = np.exp(k * (np.log(a) - log_lam))          # (a / lam)^k, computed stably
    ll = -z - log_lam - gammaln(1 + 1 / k)
    return -ll.mean()


def fit_stock_weibull(X, ages_days):
    """X: DataFrame of numeric features (an intercept is added).
    ages_days: days on market at the snapshot."""
    Xm = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    a = np.asarray(ages_days, dtype=float) + 0.5   # day 0 -> half a day
    start = np.zeros(Xm.shape[1] + 1)
    start[1] = np.log(a.mean())                    # sensible starting scale
    res = minimize(_neg_loglik, start, args=(Xm, a), method="L-BFGS-B")
    if not res.success:
        print("warning:", res.message)
    k = float(np.exp(res.x[0]))
    beta = pd.Series(res.x[1:], index=["intercept"] + list(X.columns))
    return {"k": k, "beta": beta, "nll": res.fun}


def weibull_scale(fit, X):
    """lam for each row of X."""
    Xm = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    return np.exp(Xm @ fit["beta"].to_numpy())


def expected_days(fit, X):
    return weibull_scale(fit, X) * gamma(1 + 1 / fit["k"])


def simulate_snapshot(n_cars=1_000_000, k=1.3, base_scale=40.0, effect=0.8,
                      seed=C.RANDOM_STATE):
    """Fake market with KNOWN answers, to check the method recovers them.
    Cars arrive evenly over 3 years; each has a price gap x and stays
    T ~ Weibull(k, base_scale * exp(effect * x)) days. We keep only cars
    still listed on the last day: exactly what the real data looks like."""
    rng = np.random.default_rng(seed)
    horizon = 3 * 365
    arrive = rng.uniform(0, horizon, n_cars)
    x = rng.normal(0, 0.1, n_cars)                 # price vs market, +-10%
    lam = base_scale * np.exp(effect * x)
    T = lam * rng.weibull(k, n_cars)
    alive = arrive + T > horizon
    ages = np.floor(horizon - arrive[alive])
    return pd.DataFrame({"price_gap": x[alive]}), ages
