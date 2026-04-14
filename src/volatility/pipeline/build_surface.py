import pandas as pd
from src.volatility.config.config import DATA_PATH, SNAPSHOT_DATE, COL_QUOTE_DATE
from src.volatility.data.ingestion import select_snapshot, build_open_options
from src.volatility.market.forwards import infer_forward_by_maturity
from src.volatility.data.iv import compute_snapshot_iv
from src.volatility.data.datasets import build_ssvi_dataset


def build_surface(data_path=DATA_PATH, snapshot_date=SNAPSHOT_DATE):
    df_raw = pd.read_csv(data_path)
    df_raw[COL_QUOTE_DATE] = pd.to_datetime(df_raw[COL_QUOTE_DATE], dayfirst=True, errors="coerce")

    snap     = select_snapshot(df_raw, snapshot_index=snapshot_date, date_col=COL_QUOTE_DATE)
    open_opt = build_open_options(snap)
    parity   = infer_forward_by_maturity(open_opt)
    open_opt = open_opt.merge(parity[["T_days", "F_parity", "D_parity"]], on="T_days", how="left")
    open_opt = compute_snapshot_iv(open_opt)
    return build_ssvi_dataset(open_opt)


if __name__ == "__main__":
    df_smile = build_surface()