from src.volatility.market.parity import infer_forward_from_parity
import numpy as np
import pandas as pd 
from src.volatility.config.config import PARITY_MIN_OPEN_INTEREST, PARITY_MAX_REL_SPREAD


def infer_forward_by_maturity(df_open, min_open_interest=PARITY_MIN_OPEN_INTEREST, max_rel_spread=PARITY_MAX_REL_SPREAD):
    rows = []
    for tdays, sl in df_open.groupby("T_days"):
        F, D, n_pairs = infer_forward_from_parity(
            sl,
            min_open_interest=min_open_interest,
            max_rel_spread=max_rel_spread,
        )
        rows.append({
            "T_days":  int(tdays),
            "T":       float(sl["T"].iloc[0]),
            "F_parity": F,
            "D_parity": D,
            "n_pairs":  int(n_pairs),
        })

    out = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

    # FIX: use ffill/bfill instead of deprecated fillna(method=...)
    if out["F_parity"].notna().sum() >= 2:
        valid = out["F_parity"].notna()
        out["F_parity"] = np.interp(
            out["T"].to_numpy(),
            out.loc[valid, "T"].to_numpy(),
            out.loc[valid, "F_parity"].to_numpy(),
        )
    if out["D_parity"].notna().sum() >= 2:
        valid = out["D_parity"].notna()
        out["D_parity"] = np.interp(
            out["T"].to_numpy(),
            out.loc[valid, "T"].to_numpy(),
            out.loc[valid, "D_parity"].to_numpy(),
        )

    return out