import pandas as pd
from src.volatility.data.ingestion import select_snapshot, build_open_options
from src.volatility.market.forwards import infer_forward_by_maturity
from src.volatility.data.iv import compute_snapshot_iv
from src.volatility.data.datasets import build_ssvi_dataset

df_raw = pd.read_csv("data/combined_options_data.csv")
df_raw["QUOTE_DATE"] = pd.to_datetime(df_raw["QUOTE_DATE"], dayfirst=True, errors="coerce")

snap = select_snapshot(df_raw, snapshot_index="01-01-2011", date_col="QUOTE_DATE")
open_opt = build_open_options(snap)
parity   = infer_forward_by_maturity(open_opt)
open_opt = open_opt.merge(parity[["T_days","F_parity","D_parity"]], on="T_days", how="left")
open_opt = compute_snapshot_iv(open_opt)
df_smile = build_ssvi_dataset(open_opt)