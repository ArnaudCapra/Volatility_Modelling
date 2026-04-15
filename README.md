# Volatility Modelling — Implied Skew & Multifractional Processes with Correlated Exponent

A Python library for calibrating and analysing implied volatility surfaces, with a theoretical focus on the term structure of the ATM skew and its modelling via a **Multifractional Process with Correlated Exponent (MPCE)**.

---

## Motivation

Empirical studies of the S&P 500 ATM skew (notably *Yet Another Analysis of the S&P 500 At-the-Money Skew: Crossover of Different Power-Law Behaviours*) document a clear bifurcation in the term structure of the skew: it follows one power law for maturities up to roughly 2–3 months, and a different, flatter one thereafter. Rough volatility models (Rough Heston, rBergomi) capture the short-end behaviour well — their Hurst exponent $H < \frac{1}{2}$ produces the correct explosion — but they fail at longer maturities without introducing additional factors.

The authors allude that this could be resolved if the **Hurst exponent were itself time-varying**. This project takes that hypothesis seriously and builds a full pipeline around it.

---

## Theoretical Framework

### 1. The MPCE: Multifractional Process with Correlated Exponent

The volatility process is driven by a **Mandelbrot–Van Ness kernel** applied to a CIR process — the rough Heston construction — but with one critical modification: the Hurst exponent $H_s$ is **adapted to the filtration $\mathcal{F}_s$** of the Brownian motion $B(\omega_2)$ with respect to which the kernel is integrated.

$$\begin{cases}
\frac{dS_t}{S_t} = \mu dt + \sqrt{v_t} dB_t(\omega_1) \\
v_t = v_0 + \int_{-\infty}^{t} K(t-s, H_s) dY_s \\
dY_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t}dB_t(\omega_2) \\
K(t-s, H_s) = \frac{\sqrt{2H_s \sin(\pi H_s) \Gamma(2H_s)}}{\Gamma(H_s + \frac{1}{2})} \left[ (t-s)_{+}^{H_s-\frac{1}{2}} - (-s)_{+}^{H_s-\frac{1}{2}} \right] \\
dH_t = \lambda(\eta - H_t)dt + \zeta\sqrt{H_t(1 - H_t)}dB_t(\omega_3)
\end{cases}$$

The three Brownian drivers $(\omega_1, \omega_2, \omega_3)$ are **correlated white noise**, making the skew, vol-of-vol, and roughness structurally interdependent.

**Why this construction matters — two key properties:**

1. **Orthogonality of heterogeneity layers.** Because $H_s$ is adapted to the filtration of $B(\omega_2)$ (as established in *Regularity of Multifractional Moving Average Processes with Random Hurst Exponent*), the heterogeneity of the volatility process is orthogonal from the heterogeneity of the roughness process. This allows the Hurst exponent to be modelled as a genuine stochastic process without contaminating the variance process.

2. **Correlated roughness.** Since $H_s$ is adapted to the same filtration as the volatility driver, it can be correlated with volatility itself, enabling the model to encode the empirical observation that periods of high volatility tend to coincide with changes in the roughness regime.

### 2. The Jacobi Diffusion for $H_t$

The Hurst exponent is modelled as a **Jacobi diffusion**:

$$dH_t = \lambda(\eta - H_t)\,dt + \zeta\sqrt{H_t(1 - H_t)}\,dB_t(\omega_3)$$

This is a mean-reverting process on $(0, 1)$, analogous to the Feller condition for CIR. Parameter restrictions (analogous to $2\lambda\eta > \zeta^2$) ensure $H_t$ stays strictly inside $(0, 1)$, so that the kernel $K$ remains well-defined at all times. The ergodic distribution is a **Beta distribution**, making the parametrisation highly interpretable: $\eta$ controls the long-run average roughness, $\lambda$ its mean-reversion speed, and $\zeta$ the variability of the Hurst exponent.

### 3. IV Surface and Skew via Malliavin Calculus (Alos Theorem)

The implied volatility and its ATM skew are computed analytically using the **Alos representation theorem**. The effective volatility, the Malliavin derivative of variance, and the correction operators are defined as:

$$v_t = \sqrt{\frac{\int_t^T \sigma_s^2\, ds}{T-t}}, \quad V_t^0 = E_t[BS(t, X_t, k, v_t)], \quad I_t^0(k) = BS^{-1}(k, V_t^0)$$

$$G = (\partial_x^2 - \partial_x)BS, \quad H = \partial_x G, \quad \Phi_s = \sigma_s \int_s^T D_s^W \sigma_u^2\, du$$

The full implied volatility and its skew are then given by:

$$I_t(k) = I_t^0(k) + \frac{\rho}{2} E_t\!\left[\int_t^T \frac{e^{-r(s-t)}}{\text{Vega}} H(s, X_s, k, v_s)\,\Gamma_s\, ds\right]$$

$$\frac{\partial I_t}{\partial k}(k_t^*) = \frac{E_t\!\left[\int_t^T \left(\frac{\partial F}{\partial k} - \frac{1}{2}F\right)ds\right]}{\text{Vega}_t}, \quad F = \frac{\rho}{2} e^{-r(s-t)} H \Phi_s$$

This representation is exact for Heston and serves as the foundation for the MPCE surface approximation once the Malliavin derivative is adapted to the stochastic-$H$ kernel.

---

## Project Roadmap

### Done — Baseline Infrastructure

| Module | Description |
|---|---|
| `data/` | Raw options ingestion, snapshot selection, liquidity filtering |
| `market/` | Put-call parity forward inference (no rate curve needed) |
| `pricing/` | Black-Scholes (Black-76), IV inversion via Brent, Vega, G, H operators |
| `models/essiv.py` | eSSVI surface with theta-dependent rho(theta) |
| `models/local_quad.py` | Per-slice quadratic baseline |
| `models/heston.py` | Heston simulation (Euler-Maruyama, full truncation) + Malliavin weights Phi |
| `calibration/` | Weighted least-squares calibration (spread, OI, maturity), butterfly penalty |
| `analysis/` | ATM spot & forward skew, BIC-selected power-law term structure |
| `visualization/` | Interactive 3D surfaces (Plotly), smile grids, term structure diagnostics |
| `pipeline/` | End-to-end orchestration scripts |

### In Progress / Next Steps

**1. Empirical analysis of the Hurst exponent**
- Estimate the Hurst exponent of realised volatility and spot price returns across multiple historical time series
- Inspect how $H_t$ evolves over time: persistence, clustering, regime shifts
- Study the correlation structure between $H_t$, $v_t$, and $S_t$
- Motivate the Jacobi parametrisation from data

**2. MPCE calibration to historical data**
- Implement the stochastic-$H$ kernel with Jacobi dynamics
- Calibrate MPCE parameters to realised volatility time series
- Verify that statistical properties (roughness, correlation, Beta ergodic distribution) are recovered

**3. IV surface calibration via Deep Learning**
- Following *Deep Learning Volatility* (Horvath, Muguruza, Tomas), train a neural network to price options under the MPCE
- Use the network as a fast surrogate pricer for calibration to the implied volatility surface
- Assess how the MPCE captures the two-regime power-law structure of the ATM skew

**4. Exotic greeks and sensitivity analysis**
- Compute sensitivities of vol derivatives (variance swaps, VIX options, cliquets) to the MPCE parameters: lambda, eta, zeta, rho_12, rho_23
- Interpret the forward-measure risk through the Jacobi parametrisation
- Assess whether the Beta ergodic distribution of H leads to tractable forward measures

---

## Current Pipeline Usage

```bash
# Build the IV dataset from raw options data
python -m src.volatility.pipeline.build_surface

# Calibrate eSSVI and Local Quadratic, plot surfaces and smiles
python -m src.volatility.pipeline.run_calibration

# ATM skew and forward skew power-law analysis
python -m src.volatility.pipeline.run_analysis

# Heston + Malliavin IV surface
python -m src.volatility.pipeline.run_malliavin_heston
```

The interactive notebook `tests/test.ipynb` walks through each step with inline outputs.

---

## Data Format

The pipeline expects a CSV file at `data/combined_options_data.csv` with at least the following columns:

| Column | Description |
|---|---|
| `QUOTE_DATE` | Date of the quote (dd-mm-yyyy) |
| `DTE` | Days to expiry |
| `STRIKE` | Option strike |
| `UNDERLYING_LAST` | Underlying last price |
| `C_BID`, `C_ASK` | Call bid/ask |
| `P_BID`, `P_ASK` | Put bid/ask |
| `C_OI`, `P_OI` | Call/put open interest |

---

## Dependencies

```
numpy
pandas
scipy
matplotlib
plotly
```

---

## References

- Alos, E. (2006). *A generalisation of the Hull and White formula with applications to option pricing approximation.*
- El Euch, O. & Rosenbaum, M. (2019). *The characteristic function of rough Heston models.*
- Gatheral, J. et al. (2018). *Volatility is rough.*
- Gatheral, J. & Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces.*
- Delemotte, J. Ségonne, F., de Marco, S (2024). *Yet Another Analysis of the S&P 500 At-the-Money Skew: Crossover of Different Power-Law Behaviours.*
- Horvath, B., Muguruza, A. & Tomas, M. (2021). *Deep Learning Volatility.*
- Loboda, D. Mies, F. Steland, A. (2021) *Regularity of Multifractional Moving Average Processes with Random Hurst Exponent.*
- Nualart, D. (2006). *The Malliavin Calculus and Related Topics.*
