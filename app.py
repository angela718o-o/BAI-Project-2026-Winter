import streamlit as st

from predict import MODEL_PATH, load_artifacts, predict_ticker

st.set_page_config(page_title="Public Comps Estimator", layout="centered")
st.title("EV/EBITDA Estimator")
st.caption("Predicts EV/EBITDA from quantitative metrics + business-description embeddings.")

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
st.caption(f"Test R² = {artifacts['r2']:.3f}  ·  MSE = {artifacts['mse']:.3f}")

ticker = st.text_input("Ticker", value="AAPL").strip().upper()

if st.button("Predict", type="primary") and ticker:
    with st.spinner(f"Fetching {ticker} from yfinance and predicting..."):
        try:
            r = predict_ticker(
                ticker, artifacts["model"], artifacts["scaler"], artifacts["embed_cols"]
            )
        except Exception as e:
            st.error(f"Failed: {e}")
        else:
            col1, col2 = st.columns(2)
            col1.metric("Predicted EV/EBITDA", f"{r['predicted']:.2f}")
            actual = r["actual"]
            if actual is not None:
                col2.metric(
                    "Actual (yfinance)",
                    f"{actual:.2f}",
                    delta=f"{r['predicted'] - actual:+.2f}",
                )
            else:
                col2.metric("Actual (yfinance)", "N/A")
