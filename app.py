import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf
import streamlit as st


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info_cached(ticker: str) -> dict:
    time.sleep(1)  # small baseline delay to reduce rate-limit hits
    session = _make_session()
    info = yf.Ticker(ticker, session=session).info
    if not info or info.get("trailingPegRatio") is None and info.get("symbol") is None:
        raise RuntimeError(f"Yahoo Finance returned empty data for {ticker}. Try again in a moment.")
    return info

from predict import MODEL_PATH, load_artifacts, predict_ticker_with_info

st.set_page_config(page_title="EV/EBITDA Estimator", layout="wide")

if not MODEL_PATH.exists():
    st.error(
        f"No trained model found at `{MODEL_PATH}`.\n\n"
        "Train one first:\n```\npoetry run python -m predict\n```"
    )
    st.stop()


@st.cache_resource
def get_artifacts():
    return load_artifacts()


artifacts = get_artifacts()

# Sidebar — model metadata
with st.sidebar:
    st.header("Model Info")
    st.metric("Test R²", f"{artifacts['r2']:.3f}")
    st.metric("Test MSE", f"{artifacts['mse']:.3f}")
    st.divider()
    st.caption("Architecture: Bagged XGBoost (5 seeds)")
    st.caption("Embeddings: all-MiniLM-L6-v2 → PCA(20)")
    st.caption("Target: log1p(EV/EBITDA), clipped 0–50×")

# Main UI
st.title("EV/EBITDA Estimator")
st.caption(
    "Predicts EV/EBITDA from quantitative metrics (EBITDA margin, leverage) "
    "and business-description embeddings via sentence transformers."
)

ticker = st.text_input(
    "Ticker Symbol",
    value="AAPL",
    placeholder="e.g. AAPL, MSFT, TSLA",
).strip().upper()

if st.button("Predict", type="primary") and ticker:
    with st.spinner(f"Fetching {ticker} from yfinance and predicting…"):
        try:
            info = fetch_info_cached(ticker)
            r = predict_ticker_with_info(
                ticker, info,
                artifacts["models"], artifacts["imputer"], artifacts["scaler"],
                artifacts["pca"], artifacts["ohe_cols"], artifacts["embed_cols"],
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # Company header
    name = info.get("longName") or ticker
    sector = info.get("sector") or "Unknown"
    industry = info.get("industry") or "Unknown"
    st.subheader(f"{name}  ({ticker})")
    st.caption(f"{sector}  ·  {industry}")
    st.divider()

    # Prediction results
    st.subheader("Prediction")
    predicted = r["predicted"]
    actual = r["actual"]

    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Predicted EV/EBITDA", f"{predicted:.2f}×")

    if actual is not None:
        diff = predicted - actual
        res_col2.metric(
            "Actual EV/EBITDA (yfinance)",
            f"{actual:.2f}×",
            delta=f"{diff:+.2f}×",
            delta_color="inverse",
        )
        pct_err = abs(diff) / actual * 100
        res_col3.metric("Absolute % Error", f"{pct_err:.1f}%")
    else:
        res_col2.metric("Actual EV/EBITDA (yfinance)", "N/A")

    st.divider()

    # Financial inputs used
    st.subheader("Key Inputs Used")
    ebitda_margin = info.get("ebitdaMargins")
    debt_to_equity = info.get("debtToEquity")
    ebitda = info.get("ebitda")
    total_debt = info.get("totalDebt") or 0
    total_cash = info.get("totalCash") or 0

    fin_col1, fin_col2, fin_col3 = st.columns(3)
    fin_col1.metric(
        "EBITDA Margin",
        f"{ebitda_margin * 100:.1f}%" if ebitda_margin is not None else "N/A",
    )
    fin_col2.metric(
        "Debt / Equity",
        f"{debt_to_equity:.2f}" if debt_to_equity is not None else "N/A",
    )
    if ebitda:
        net_debt_ebitda = (total_debt - total_cash) / ebitda
        fin_col3.metric("Net Debt / EBITDA", f"{net_debt_ebitda:.2f}×")
    else:
        fin_col3.metric("Net Debt / EBITDA", "N/A")
