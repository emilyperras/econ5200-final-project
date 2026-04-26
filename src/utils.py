"""
Utility functions for the ECON 5200 Senate STOCK Act trading analysis.

Provides helpers for confidence interval construction, ATE scaling,
attenuation-bias correction, and percentage formatting used throughout
the DML results and Streamlit dashboard.
"""

from scipy import stats


_Z_MAP = {
    0.80: 1.28,
    0.90: 1.645,
    0.95: 1.96,
    0.99: 2.576,
}


def compute_ci(ate: float, se: float, confidence: float = 0.95) -> tuple[float, float]:
    """Return (lower, upper) CI bounds using the normal approximation.

    Uses a lookup table for common confidence levels (0.80, 0.90, 0.95, 0.99)
    and falls back to scipy's ppf for arbitrary levels.
    """
    z = _Z_MAP.get(confidence, stats.norm.ppf((1 + confidence) / 2))
    return ate - z * se, ate + z * se


def scale_ate(ate: float, se: float, multiplier: float) -> tuple[float, float]:
    """Return (scaled_ate, scaled_se) after applying a treatment intensity multiplier.

    Both ATE and SE scale proportionally so that CI width remains consistent
    with the assumption that the entire data-generating process is amplified.
    """
    return ate * multiplier, se * multiplier


def bias_adjusted_ate(
    ate: float,
    se: float,
    window: int,
    max_window: int = 45,
    max_attenuation: float = 0.15,
) -> tuple[float, float]:
    """Return (adjusted_ate, adjusted_se) corrected for disclosure-lag attenuation.

    The STOCK Act allows up to max_window days between trade execution and
    disclosure. This lag introduces measurement error that attenuates the ATE
    toward zero (classical errors-in-variables). The correction assumes linear
    attenuation: at max_window the bias is max_attenuation; at window=0 it is zero.

    Args:
        ate: Observed average treatment effect.
        se: Standard error of the observed ATE.
        window: Hypothetical disclosure window in days (1 to max_window).
        max_window: Current statutory maximum (45 days under the STOCK Act).
        max_attenuation: Assumed attenuation fraction at max_window (default 15%).
    """
    true_ate = ate / (1 - max_attenuation)
    true_se  = se  / (1 - max_attenuation)
    attenuation_at_window = max_attenuation * (window / max_window)
    factor = 1 - attenuation_at_window
    return true_ate * factor, true_se * factor


def format_pct(val: float) -> str:
    """Format a float as a percentage string with 3 decimal places.

    Example: format_pct(-0.00190) -> '-0.190%'
    """
    return f"{val * 100:+.3f}%"
