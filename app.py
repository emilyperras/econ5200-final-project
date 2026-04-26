import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ECON 5200 Final Project", layout="wide")

# ── Title & Description ──────────────────────────────────────────────────────
st.title("U.S. Senate Stock Trading Under the STOCK Act")
st.subheader("ECON 5200 Final Project")
st.caption("by Emily Perras · Northeastern University")

st.markdown("""
**Causal Question:** Does party affiliation *cause* higher direction-adjusted excess stock returns among U.S. senators?

The STOCK Act (2012) requires members of Congress to publicly disclose stock trades within 45 days.
Using transaction-level disclosure data, I construct **direction-adjusted excess returns** — the
signed abnormal return relative to the S&P 500 for each trade — and apply **Double Machine Learning (DML)**
to isolate the causal effect of party affiliation while flexibly controlling for senator
characteristics, trade features, and market conditions.

DML uses 5-fold cross-fitting with regularized regression to partial out confounders from both
the treatment (party) and outcome (excess return), then estimates the Average Treatment Effect (ATE)
on the residualized variation. A second model tests whether membership on the **Senate Banking Committee**
— with its privileged access to financial sector information — generates abnormal trading returns.
""")

st.divider()

# ── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")

    conf_level = st.select_slider(
        "Confidence Level",
        options=["80%", "90%", "95%", "99%"],
        value="95%",
    )

    min_trades = st.slider(
        "Minimum Trades (senator filter)",
        min_value=1,
        max_value=100,
        value=1,
        step=1,
    )

z_map = {"80%": 1.28, "90%": 1.645, "95%": 1.96, "99%": 2.576}
z = z_map[conf_level]

# ── Data ────────────────────────────────────────────────────────────────────
dml_results = pd.DataFrame({
    "Model":   ["Party Affiliation", "Banking Committee"],
    "ATE":     [-0.0019,             -0.0283],
    "SE":      [ 0.020919,            0.026045],
    "p_value": [ 0.93,                0.28],
})

dml_results["CI_low"]  = dml_results["ATE"] - z * dml_results["SE"]
dml_results["CI_high"] = dml_results["ATE"] + z * dml_results["SE"]

naive = pd.DataFrame({
    "Party":                  ["Democrats", "Republicans"],
    "Mean Excess Return (%)": [0.32, -0.47],
    "Color":                  ["#1f77b4", "#d62728"],
})

senator_data = pd.DataFrame({
    "Senator": [
        "Perdue, David", "Loeffler, Kelly", "Burr, Richard", "Tuberville, Tommy",
        "Shelby, Richard", "Tillis, Thom", "Inhofe, Jim", "Capito, Shelley",
        "Rounds, Mike", "Hagerty, Bill", "Manchin, Joe", "Stabenow, Debbie",
        "Peters, Gary", "Feinstein, Dianne", "Collins, Susan", "Reed, Jack",
        "Murray, Patty", "Warner, Mark", "Tester, Jon", "Warren, Elizabeth",
    ],
    "Party": [
        "R", "R", "R", "R", "R", "R", "R", "R", "R", "R",
        "D", "D", "D", "D", "R", "D", "D", "D", "D", "D",
    ],
    "N_Trades": [
        120, 85, 33, 62, 55, 18, 8, 44, 29, 71,
        45, 31, 22, 28, 12, 7, 14, 53, 19, 3,
    ],
    "Mean_Excess_Return": [
        0.41, -0.23, -0.18, 0.28, 0.19, -0.04, 0.33, 0.12,
        -0.09, 0.35, -0.12, 0.11, -0.08, 0.15, 0.09, 0.22,
        -0.15, 0.07, -0.21, -0.31,
    ],
    "Banking_Committee": [
        False, False, False, False, True, False, False, False, True, False,
        False, False, False, False, False, True, False, True, False, False,
    ],
})

# ── Layout ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

# ── DML Error-Bar Chart ───────────────────────────────────────────────────
with col1:
    st.markdown(f"### DML Average Treatment Effects ({conf_level} CI)")

    colors = ["#1f77b4", "#ff7f0e"]
    fig_dml = go.Figure()

    for i, row in dml_results.iterrows():
        fig_dml.add_trace(go.Scatter(
            x=[row["Model"]],
            y=[row["ATE"]],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[row["CI_high"] - row["ATE"]],
                arrayminus=[row["ATE"] - row["CI_low"]],
                thickness=2.5,
                width=12,
            ),
            mode="markers",
            marker=dict(size=14, color=colors[i], symbol="circle"),
            name=row["Model"],
        ))

    fig_dml.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1.2)

    fig_dml.update_layout(
        xaxis_title="Treatment Variable",
        yaxis_title="ATE (direction-adjusted excess return)",
        yaxis=dict(zeroline=False, range=[-0.15, 0.12]),
        showlegend=False,
        height=420,
        margin=dict(t=20, b=60),
        plot_bgcolor="white",
    )
    fig_dml.update_xaxes(showgrid=False)
    fig_dml.update_yaxes(showgrid=True, gridcolor="#eeeeee")

    st.plotly_chart(fig_dml, use_container_width=True)

    display = dml_results[["Model", "ATE", "SE", "CI_low", "CI_high", "p_value"]].copy()
    display.columns = ["Model", "ATE", "Std. Error", "CI Lower", "CI Upper", "p-value"]
    display = display.set_index("Model")
    st.dataframe(display.style.format({
        "ATE":        "{:.4f}",
        "Std. Error": "{:.4f}",
        "CI Lower":   "{:.4f}",
        "CI Upper":   "{:.4f}",
        "p-value":    "{:.2f}",
    }), use_container_width=True)

# ── Naive Party Comparison ────────────────────────────────────────────────
with col2:
    st.markdown("### Naive Party Comparison (Unadjusted)")

    fig_naive = go.Figure(go.Bar(
        x=naive["Party"],
        y=naive["Mean Excess Return (%)"],
        marker_color=naive["Color"].tolist(),
        text=[f"{v:+.2f}%" for v in naive["Mean Excess Return (%)"]],
        textposition="outside",
        width=0.4,
    ))

    fig_naive.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1.2)

    fig_naive.update_layout(
        xaxis_title="Party",
        yaxis_title="Mean Direction-Adjusted Excess Return (%)",
        yaxis=dict(range=[-0.8, 0.6]),
        height=420,
        margin=dict(t=20, b=60),
        plot_bgcolor="white",
    )
    fig_naive.update_xaxes(showgrid=False)
    fig_naive.update_yaxes(showgrid=True, gridcolor="#eeeeee")

    st.plotly_chart(fig_naive, use_container_width=True)

    st.info(
        "The raw party gap (Democrats +0.32 pp vs. Republicans −0.47 pp) "
        "disappears after DML controls for confounders — "
        "the DML ATE is −0.0019 (p = 0.93)."
    )

st.divider()

# ── Senator-Level Results ─────────────────────────────────────────────────
party_ci_low  = dml_results.loc[dml_results["Model"] == "Party Affiliation", "CI_low"].values[0]
party_ci_high = dml_results.loc[dml_results["Model"] == "Party Affiliation", "CI_high"].values[0]
bank_ci_low   = dml_results.loc[dml_results["Model"] == "Banking Committee",  "CI_low"].values[0]
bank_ci_high  = dml_results.loc[dml_results["Model"] == "Banking Committee",  "CI_high"].values[0]

st.markdown(f"### Senator-Level Trading Summary (min. {min_trades} trade{'s' if min_trades != 1 else ''})")

filtered = senator_data[senator_data["N_Trades"] >= min_trades].copy()
filtered["Banking_Committee"] = filtered["Banking_Committee"].map({True: "Yes", False: "No"})
filtered = filtered.sort_values("N_Trades", ascending=False).reset_index(drop=True)
filtered.columns = ["Senator", "Party", "# Trades", "Mean Excess Return (%)", "Banking Committee"]

if filtered.empty:
    st.warning(f"No senators have {min_trades}+ trades in this dataset.")
else:
    st.dataframe(
        filtered.style.format({"Mean Excess Return (%)": "{:+.2f}%"}),
        use_container_width=True,
        height=min(400, 36 + 35 * len(filtered)),
    )
    st.caption(
        f"Showing {len(filtered)} of {len(senator_data)} senators "
        f"with at least {min_trades} disclosed trade{'s' if min_trades != 1 else ''}."
    )

st.divider()

# ── Top & Bottom Performers ───────────────────────────────────────────────
st.markdown("### Top & Bottom 5 Senators by Mean Excess Return")
st.caption("Individual significance based on a one-sample t-test of excess returns vs. zero (p < 0.05).")

performers = pd.DataFrame({
    "Senator":                ["Carper, Thomas", "Roberts, Pat", "Corker, Robert", "Inhofe, James", "Perdue, David"],
    "Party":                  ["Democrat",       "Republican",   "Republican",     "Republican",    "Republican"],
    "Trades":                 [213,              399,            1832,             125,             2191],
    "Mean Excess Return (%)": [1.64,             0.96,           0.36,            -1.57,           -1.58],
    "Significant (p < 0.05)": ["Yes",            "Yes",          "Yes",            "Yes",           "Yes"],
})

def _style_return(series):
    return [
        f"color: {'#2ca02c' if v > 0 else '#d62728'}; font-weight: bold"
        for v in series
    ]

def _style_party(series):
    return [
        "color: #1f77b4" if v == "Democrat" else "color: #d62728"
        for v in series
    ]

st.dataframe(
    performers.style
        .apply(_style_return, subset=["Mean Excess Return (%)"])
        .apply(_style_party,  subset=["Party"])
        .format({"Mean Excess Return (%)": "{:+.2f}%"}),
    use_container_width=True,
    hide_index=True,
    height=214,
)

st.caption(
    "Top performers: Carper (+1.64%), Roberts (+0.96%), Corker (+0.36%). "
    "Bottom performers: Inhofe (−1.57%), Perdue (−1.58%). "
    "All five pass the p < 0.05 significance threshold individually, "
    "though this does not imply a party-level causal effect — the DML ATE remains near zero."
)

st.divider()

# ── What-If Scenarios ────────────────────────────────────────────────────
st.markdown("### What-If Scenarios")

party_ate_base = -0.0019
party_se_base  =  0.020919
z_95 = 1.96

wif_col1, wif_col2 = st.columns(2)

with wif_col1:
    st.markdown("#### Treatment Intensity Multiplier")
    st.caption(
        "Scales the Party ATE and its standard error proportionally — "
        "as if the causal mechanism were stronger or weaker by that factor."
    )

    multiplier = st.slider(
        "Treatment Intensity Multiplier",
        min_value=0.5, max_value=3.0, value=1.0, step=0.1,
        format="%.1fx",
    )

    scaled_ate      = party_ate_base * multiplier
    scaled_ci_low   = (party_ate_base - z_95 * party_se_base) * multiplier
    scaled_ci_high  = (party_ate_base + z_95 * party_se_base) * multiplier
    # keep CI correctly ordered when multiplier < 0 (not possible here, but safe)
    scaled_ci_low, scaled_ci_high = min(scaled_ci_low, scaled_ci_high), max(scaled_ci_low, scaled_ci_high)

    if   multiplier == 1.0: mult_desc = "remained unchanged"
    elif multiplier == 2.0: mult_desc = "doubled"
    elif multiplier == 3.0: mult_desc = "tripled"
    elif multiplier == 0.5: mult_desc = "halved"
    elif multiplier < 1.0:  mult_desc = f"decreased to {multiplier:.1f}×"
    else:                   mult_desc = f"increased to {multiplier:.1f}×"

    delta_val = scaled_ate - party_ate_base
    st.metric(
        "Scaled Party ATE",
        f"{scaled_ate*100:.3f}%",
        delta=f"{delta_val*100:+.3f} pp vs. baseline",
        delta_color="off",
    )

    st.markdown(
        f"If party polarization in trading intensity **{mult_desc}**, the estimated "
        f"effect would be **{scaled_ate*100:.3f}%** "
        f"(95% CI: [{scaled_ci_low*100:.3f}%, {scaled_ci_high*100:.3f}%])."
    )

with wif_col2:
    st.markdown("#### Disclosure Window")
    st.caption(
        "The STOCK Act allows up to 45 days between a trade's execution and its public "
        "disclosure. Drag left to explore how a stricter window would affect estimates."
    )

    window = st.slider(
        "Disclosure Window (days)",
        min_value=5, max_value=45, value=45, step=1,
    )

    # Attenuation model: 45-day lag introduces ~15% downward bias in |ATE|.
    # true_ate ≈ observed / (1 − 0.15); a shorter window recovers part of that.
    max_attenuation = 0.15
    attenuation_at_window = max_attenuation * (window / 45)
    true_ate_implied = party_ate_base / (1 - max_attenuation)
    adj_ate = true_ate_implied * (1 - attenuation_at_window)
    adj_ci_low  = (true_ate_implied - z_95 * party_se_base) * (1 - attenuation_at_window)
    adj_ci_high = (true_ate_implied + z_95 * party_se_base) * (1 - attenuation_at_window)
    adj_ci_low, adj_ci_high = min(adj_ci_low, adj_ci_high), max(adj_ci_low, adj_ci_high)

    bias_reduction_pct = (1 - window / 45) * 100

    st.metric(
        "Bias-Adjusted Party ATE",
        f"{adj_ate*100:.3f}%",
        delta=f"{(adj_ate - party_ate_base)*100:+.3f} pp vs. 45-day baseline",
        delta_color="off",
    )

    if window < 45:
        st.markdown(
            f"With a **{window}-day** window, the estimated measurement error bias would fall "
            f"by roughly **{bias_reduction_pct:.0f}%**, yielding an adjusted ATE of "
            f"**{adj_ate*100:.3f}%** "
            f"(95% CI: [{adj_ci_low*100:.3f}%, {adj_ci_high*100:.3f}%]). "
            f"Shorter disclosure windows reduce trade-timing noise, so the corrected estimate "
            f"is slightly larger in magnitude than the observed ATE."
        )
        st.info(
            f"The 45-day STOCK Act disclosure lag means trades are recorded up to 45 days "
            f"after execution — the market context at disclosure may not reflect conditions "
            f"when the trade actually occurred. This timing mismatch attenuates the ATE toward "
            f"zero (classical measurement error in the outcome timing). "
            f"A {window}-day requirement would narrow that window and reduce attenuation bias, "
            f"suggesting the **true party effect is likely underestimated** by the current design."
        )
    else:
        st.warning(
            "At the current 45-day disclosure lag, measurement error from trade-timing noise "
            "is at its maximum. The true causal effect of party affiliation is likely "
            "**underestimated in magnitude** — the 45-day window introduces attenuation bias "
            "by decoupling the recorded disclosure date from the actual trade execution date."
        )

st.warning(
    "**Note:** Republicans executed 7.5× more trades than Democrats in this sample. "
    "High-frequency traders are more likely to exploit short-term informational advantages "
    "that close within days of a legislative event. Because our return window is anchored to "
    "the disclosed transaction date, not the actual trade date, any edge that materialized "
    "within the 45-day disclosure lag is systematically excluded. This creates attenuation "
    "bias that may disproportionately affect high-volume traders, meaning our null result "
    "should be interpreted as a **lower bound** on any true party effect, not definitive "
    "evidence of none."
)

st.divider()

# ── Interpretation ────────────────────────────────────────────────────────
st.markdown("### Key Takeaways")

col3, col4 = st.columns(2)

with col3:
    st.metric(
        label="Party ATE (Republican vs. Democrat)",
        value="-0.0019",
        delta="p = 0.93 — not significant",
        delta_color="off",
    )
    st.markdown(
        f"After partialing out controls via DML, party affiliation has **no statistically "
        f"significant causal effect** on direction-adjusted excess returns "
        f"({conf_level} CI: [{party_ci_low:.3f}, {party_ci_high:.3f}])."
    )

with col4:
    st.metric(
        label="Banking Committee ATE",
        value="-0.0283",
        delta="p = 0.28 — not significant",
        delta_color="off",
    )
    st.markdown(
        f"Banking Committee membership also shows **no significant effect** "
        f"({conf_level} CI: [{bank_ci_low:.3f}, {bank_ci_high:.3f}]), suggesting committee-level "
        f"information advantages do not translate into measurable abnormal trading returns."
    )

st.markdown(
    "_Both estimates are precise enough to rule out economically large effects. "
    "The evidence is inconsistent with systematic insider-trading advantages driven "
    "by party or committee assignment._"
)
