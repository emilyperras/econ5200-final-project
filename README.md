# ECON 5200: Does Party Affiliation Cause Differences in Senate Stock Trading Returns?

**Emily Perras · Northeastern University · ECON 5200: Causal Machine Learning & Applied Analytics**

---

## Causal Question

> Does party affiliation *cause* higher direction-adjusted excess stock returns among U.S. senators?

Using transaction-level STOCK Act disclosure data, I construct **direction-adjusted excess returns** — the signed abnormal return relative to the S&P 500 for each trade — and apply Double Machine Learning to isolate the causal effect of party affiliation after flexibly controlling for senator characteristics, trade features, and market conditions. A second model tests whether **Senate Banking Committee membership** generates measurable informational trading advantages.

---

## Identification Strategy

| Design choice | Detail |
|---|---|
| **Estimator** | Double Machine Learning — Partially Linear Regression (`DoubleMLPLR`) |
| **Cross-fitting** | 5-fold |
| **Nuisance models** | Random Forest (treatment and outcome equations) |
| **Treatment 1** | Party affiliation (Republican = 1, Democrat = 0) |
| **Treatment 2** | Senate Banking Committee membership (Yes = 1) |
| **Outcome** | Direction-adjusted excess return vs. S&P 500 |
| **Controls** | Senator characteristics, trade features, market conditions |

DML partials out confounders from both the treatment and outcome equations using cross-fitted nuisance models, then estimates the Average Treatment Effect (ATE) on the residualized variation. This removes selection bias from observable characteristics without imposing parametric functional form assumptions.

---

## Key Results

| Model | ATE | Std. Error | 95% CI | p-value | Interpretation |
|---|---|---|---|---|---|
| Party Affiliation | −0.0019 | 0.0209 | [−0.043, 0.039] | 0.93 | Not significant |
| Banking Committee | −0.0283 | 0.0260 | [−0.079, 0.023] | 0.28 | Not significant |

After controlling for observable confounders, **neither party affiliation nor Banking Committee membership shows a statistically significant causal effect** on direction-adjusted excess returns. The naive party gap (Democrats +0.32 pp vs. Republicans −0.47 pp) disappears once DML removes selection-on-observables.

> **Interpretation caveat:** The 45-day STOCK Act disclosure lag introduces attenuation bias that may disproportionately affect high-volume traders (Republicans executed ~7.5× more trades in this sample). The null result should be read as a **lower bound** on any true party effect, not definitive evidence of none.

---

## Data

| Field | Detail |
|---|---|
| **Source** | Kaggle — [`heresjohnnyv/congress-investments`](https://www.kaggle.com/datasets/heresjohnnyv/congress-investments) |
| **License** | CC0 Public Domain |
| **Observations** | N = 6,690 trades |
| **Coverage** | U.S. Senate stock disclosures filed under the STOCK Act (2012–present) |

---

## Setup

**1. Clone the repo:**

```bash
git clone https://github.com/emilyperras/econ5200-final-project.git
cd econ5200-final-project
```

**2. Create and activate a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Usage

**Run the analysis notebook:**

```bash
jupyter notebook notebooks/Final_Project.ipynb
```

**Launch the interactive Streamlit dashboard:**

```bash
streamlit run app.py
```

**Deployed app:** [https://econ5200-final-project-uoyazapm2zxnbpuavndwgl.streamlit.app/](https://econ5200-final-project-uoyazapm2zxnbpuavndwgl.streamlit.app/)

---
## Repository Structure

```
econ5200-final-project/
├── README.md
├── requirements.txt
├── app.py                          ← Streamlit dashboard
├── notebooks/
│   └── Final_Project.ipynb         ← Full DML analysis
├── src/
│   ├── utils.py                    ← CI, scaling, and bias-correction helpers
│   └── decompose.py
└── data/                           ← Raw data (not tracked in git)
    └── congress_trading.csv
```

---

## Dependencies

See `requirements.txt`. Core packages: `streamlit`, `pandas`, `numpy`, `matplotlib`, `plotly`, `scikit-learn`, `doubleml`, `scipy`.
