from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "combined_features.csv"
TARGET = "ev_to_ebitda"
QUANT_COLS = [
    "shares_outstanding", "total_debt", "total_cash", "ebitda",
    "debt_to_equity", "ebitda_margin", "forwardPE",
]
# Heavy right tails (mega-cap vs small-cap); signed log1p keeps sign for negative ebitda.
LOG_COLS = ["shares_outstanding", "total_debt", "total_cash", "ebitda"]
# Must match the model used in data/processed/new_processing.ipynb to embed training data.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

_YF_TO_FEATURE = {
    "shares_outstanding": "sharesOutstanding",
    "total_debt": "totalDebt",
    "total_cash": "totalCash",
    "ebitda": "ebitda",
    "debt_to_equity": "debtToEquity",
    "ebitda_margin": "ebitdaMargins",
    "forwardPE": "forwardPE",
}

_embed_model = None


def load_dataset(path: Path = DATA_PATH):
    df = pd.read_csv(path, index_col="ticker")
    X = df.drop(columns=[TARGET]).replace([np.inf, -np.inf], np.nan)
    y = df[TARGET].replace([np.inf, -np.inf], np.nan)

    mask = X.notna().all(axis=1) & y.notna() & (y > 0) & (y < 50)
    X, y = X[mask].copy(), y[mask].copy()

    for c in LOG_COLS:
        X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    y = np.log1p(y)

    return X, y


def build_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    embed_cols = [c for c in X_train.columns if c.startswith("embed_")]
    scaler = StandardScaler()
    quant_train = scaler.fit_transform(X_train[QUANT_COLS])
    quant_test = scaler.transform(X_test[QUANT_COLS])
    X_all_train = np.column_stack([quant_train, X_train[embed_cols].values])
    X_all_test = np.column_stack([quant_test, X_test[embed_cols].values])
    return X_all_train, X_all_test, scaler, embed_cols


def train(test_size: float = 0.2, random_state: int = 42):
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_all_train, X_all_test, scaler, embed_cols = build_features(X_train, X_test)

    model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, verbosity=0,
    )
    model.fit(X_all_train, y_train)
    y_pred = model.predict(X_all_test)

    return {
        "model": model,
        "scaler": scaler,
        "embed_cols": embed_cols,
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
    }


def predict_ev_to_ebitda(X: pd.DataFrame, model, scaler, embed_cols):
    X = X.copy()
    for c in LOG_COLS:
        X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    quant = scaler.transform(X[QUANT_COLS])
    X_all = np.column_stack([quant, X[embed_cols].values])
    # Model predicts log1p(target); invert for the EV/EBITDA scale.
    return np.expm1(model.predict(X_all))


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _features_from_info(ticker: str, info: dict) -> pd.DataFrame:
    quant = {feat: info.get(yf_key) for feat, yf_key in _YF_TO_FEATURE.items()}
    missing = [k for k, v in quant.items() if v is None]
    if missing:
        raise ValueError(f"yfinance missing fields for {ticker}: {missing}")

    summary = info.get("longBusinessSummary") or ""
    if not summary:
        raise ValueError(f"yfinance returned no business summary for {ticker}")

    embedding = _get_embed_model().encode([summary])[0]
    row = {**quant, **{f"embed_{i}": float(v) for i, v in enumerate(embedding)}}
    return pd.DataFrame([row], index=pd.Index([ticker], name="ticker"))


def fetch_ticker_features(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    return _features_from_info(ticker, yf.Ticker(ticker).info)


def predict_ticker(ticker: str, model, scaler, embed_cols) -> dict:
    """Predict EV/EBITDA for a live ticker. Returns {'predicted', 'actual'}."""
    import yfinance as yf
    info = yf.Ticker(ticker).info
    X = _features_from_info(ticker, info)
    predicted = float(predict_ev_to_ebitda(X, model, scaler, embed_cols)[0])
    actual = info.get("enterpriseToEbitda")
    return {"predicted": predicted, "actual": actual}


if __name__ == "__main__":
    out = train()
    print(f"Test R2:  {out['r2']:.4f}")
    print(f"Test MSE: {out['mse']:.4f}")

    import sys
    if len(sys.argv) > 1:
        for t in sys.argv[1:]:
            try:
                r = predict_ticker(t, out["model"], out["scaler"], out["embed_cols"])
                actual = r["actual"]
                actual_str = f"{actual:.2f}" if actual is not None else "N/A"
                diff_str = f", diff={r['predicted'] - actual:+.2f}" if actual is not None else ""
                print(f"{t}: predicted={r['predicted']:.2f}, actual={actual_str}{diff_str}")
            except Exception as e:
                print(f"{t}: failed — {e}")
