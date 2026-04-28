# ══════════════════════════════════════════════════════════════════════════════
#  volatility/config/config.py
#  Single source of truth for all tuneable parameters in the project.
# ══════════════════════════════════════════════════════════════════════════════

# ── Data ──────────────────────────────────────────────────────────────────────

DATA_PATH      = "data/combined_options_data.csv"
SNAPSHOT_DATE  = "01-01-2012"          # default snapshot; override per run
DATE_FORMAT    = "dayfirst"            # passed to pd.to_datetime(dayfirst=...)

# Raw CSV column names expected by ingestion / quote helpers
COL_QUOTE_DATE      = "QUOTE_DATE"
COL_DTE             = "DTE"
COL_UNDERLYING_LAST = "UNDERLYING_LAST"
COL_STRIKE          = "STRIKE"
COL_C_BID, COL_C_ASK, COL_C_LAST, COL_C_OI = "C_BID", "C_ASK", "C_LAST", "C_OI"
COL_P_BID, COL_P_ASK, COL_P_LAST, COL_P_OI = "P_BID", "P_ASK", "P_LAST", "P_OI"


# ── Liquidity filters ─────────────────────────────────────────────────────────

# Used in market.parity / market.forwards (per-maturity parity regression)
PARITY_MIN_OPEN_INTEREST = 50
PARITY_MAX_REL_SPREAD    = 0.25

# Used in data.datasets.build_ssvi_dataset
DATASET_MIN_OPEN_INTEREST       = 500
DATASET_MAX_REL_SPREAD          = 0.25
DATASET_ATM_BAND_FLOOR          = 0.015   # minimum ATM-averaging half-width in k
DATASET_ATM_BAND_SCALE          = 0.5     # ATM half-width = max(floor, scale * sqrt(T))
DATASET_ATM_REL_DIFF_THRESHOLD  = 0.20    # max relative call/put IV divergence for averaging
DATASET_MIN_UNIQUE_STRIKES      = 5       # minimum distinct strikes to keep a maturity slice

# Used in market.forwards (theta knot estimation)
THETA_KNOT_MAX_NEAREST = 7


# ── Calibration ───────────────────────────────────────────────────────────────

# eSSVI (fit_essiv.py / loss.py)
ESSVI_LAMBDA_BUTTERFLY  = 25.0    # penalty weight on butterfly no-arb constraint
ESSVI_MAX_NFEV          = 80_000  # max function evaluations for least_squares
ESSVI_F_SCALE           = 1.0     # soft-L1 robustness scale
ESSVI_LOSS              = "soft_l1"
ESSVI_METHOD            = "trf"

# eSSVI initial parameter values (in unconstrained space)
ESSVI_INIT_RHO_INF  = -0.4   # long-maturity correlation  (tanh-space init)
ESSVI_INIT_RHO_0    = -0.7   # short-maturity correlation (tanh-space init)
ESSVI_INIT_LOG_C_RHO = 0.0   # log(c_rho) = 0  →  c_rho = 1
ESSVI_INIT_LOG_ETA   = 0.0   # log(eta)   = 0  →  eta   = 1
ESSVI_INIT_LOGIT_GAMMA = 0.0 # logit(gamma) = 0  →  gamma = 0.5

# Localized Quadratic (fit_local.py)
LOCAL_MIN_STRIKES = 5   # minimum strikes per maturity slice to attempt a fit


# ── Heston Monte-Carlo (run_malliavin_heston_iv.py) ───────────────────────────

HESTON_N_PATHS  = 5_000
HESTON_N_STEPS  = 200
HESTON_SEED     = 42
HESTON_R        = 0.0       # risk-free rate (zero-rate assumption)

# Default Heston parameters (can be overridden at call site)
HESTON_KAPPA = 5.0    # mean-reversion speed
HESTON_THETA = 0.04   # long-run variance  → 20 % long-run vol
HESTON_XI    = 0.7    # vol-of-vol
HESTON_RHO   = -0.7   # spot/vol Brownian correlation
HESTON_S0    = 1.0    # normalised spot
HESTON_V0    = 0.04   # initial variance   → 20 % initial vol

# Default simulation grid
import numpy as np
HESTON_MATURITIES = np.arange(0.075, 1.11, 0.1)
HESTON_STRIKES    = np.linspace(0.8, 1.2, 9)


# ── Analysis ──────────────────────────────────────────────────────────────────

# ATM finite-difference skew (analysis.diagnostics)
DIAG_DK_BAND = 0.05   # half-width in log-moneyness around k=0 for local regressions

# BIC piecewise power-law (analysis.term_structure)
BIC_MIN_PTS = 3       # minimum points per segment in the BIC search


# ── Weighting (calibration.weights) ───────────────────────────────────────────

WEIGHT_OI_NORMALIZER  = 100.0   # open-interest is normalised by this before clipping to [0,1]
WEIGHT_TIME_DENOM     = 0.05    # time_term = T / (T + TIME_DENOM); controls short-mat damping
WEIGHT_EPS            = 1e-12   # shared epsilon guard (also used in market.parity)


# ── Visualisation ─────────────────────────────────────────────────────────────

# Smile comparison grid (visualization.smiles)
VIZ_N_SLICES = 6

# Interactive IV surface (visualization.interactive_surface)
VIZ_SURFACE_N_K     = 250
VIZ_SURFACE_N_T     = 250
VIZ_SURFACE_K_CLIP  = 0.45          # maximum |k| shown on the surface
VIZ_SURFACE_IV_CLIP = (0.0, 1.5)   # (floor, ceiling) for displayed IV

# Brand colours (shared across smiles.py and diagnostics.py)
COLOR_ESSVI  = "#e6453c"
COLOR_LOCAL  = "#3c7fe6"
COLOR_MARKET = "#222222"
COLOR_FD_ATM = "#e69c3c"