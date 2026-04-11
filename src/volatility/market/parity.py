from src.volatility.data.quotes import side_fields 
from scipy.optimize import least_squares
import numpy as np

EPS = 1e-12 

def liquidity_mask(df, side, min_open_interest=50, max_rel_spread=0.25):
    f = side_fields(side)

    bid = df[f["bid"]]
    ask = df[f["ask"]]
    mid = 0.5 * (bid + ask)
    rel_spread = (ask - bid) / mid.replace(0, np.nan)

    ok = (
        bid.notna() & ask.notna() &
        (ask >= bid) &
        mid.notna() & (mid > 0) &
        rel_spread.notna() & (rel_spread <= max_rel_spread)
    )

    if f["oi"] in df.columns:
        ok &= df[f["oi"]].fillna(0) >= min_open_interest

    return ok


def infer_forward_from_parity(slice_df, min_open_interest=50, max_rel_spread=0.25):
    """
    Per maturity: C - P = D(F - K) = alpha + beta*K
    beta = -D,  alpha = D*F  =>  F = -alpha/beta
    """
    df = slice_df.copy()

    required = {"STRIKE", "C_BID", "C_ASK", "P_BID", "P_ASK"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    okc = liquidity_mask(df, "call", min_open_interest=min_open_interest, max_rel_spread=max_rel_spread)
    okp = liquidity_mask(df, "put",  min_open_interest=min_open_interest, max_rel_spread=max_rel_spread)
    sub = df[okc & okp].copy()

    if len(sub) < 5:
        return np.nan, np.nan, 0

    sub["C_MID"] = 0.5 * (sub["C_BID"] + sub["C_ASK"])
    sub["P_MID"] = 0.5 * (sub["P_BID"] + sub["P_ASK"])
    sub["Y"] = sub["C_MID"] - sub["P_MID"]
    sub["K"] = sub["STRIKE"].astype(float)

    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["Y", "K"]).copy()
    if len(sub) < 5:
        return np.nan, np.nan, 0

    c_rs = (sub["C_ASK"] - sub["C_BID"]) / sub["C_MID"]
    p_rs = (sub["P_ASK"] - sub["P_BID"]) / sub["P_MID"]
    spread = c_rs + p_rs

    k0 = np.median(sub["K"].to_numpy())
    w = 1.0 / ((spread + EPS) * (1.0 + 0.5 * np.abs(np.log(sub["K"] / k0))))
    w = np.clip(w.to_numpy(), 1e-8, 1e8)

    X = np.column_stack([np.ones(len(sub)), sub["K"].to_numpy()])
    y = sub["Y"].to_numpy()

    def resid(beta):
        return (X @ beta - y) * np.sqrt(w)

    beta0 = np.array([np.nanmedian(y), -1.0], dtype=float)
    res = least_squares(resid, beta0, loss="soft_l1", max_nfev=20000)
    alpha, beta = res.x

    if not np.isfinite(beta) or beta >= 0:
        return np.nan, np.nan, len(sub)

    D = -beta
    F = -alpha / beta

    if not np.isfinite(F) or not np.isfinite(D) or F <= 0 or D <= 0:
        return np.nan, np.nan, len(sub)

    return float(F), float(D), len(sub)
