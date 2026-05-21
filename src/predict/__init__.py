from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "combined_features.csv"
MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "model.joblib"
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
# Reduce 384-dim embeddings so they don't drown out 7 quant features.
PCA_COMPONENTS = 20
# Seeds for bagged XGBoost — average predictions across these to reduce variance.
BAG_SEEDS = [42, 7, 13, 99, 2024]

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

    # Only drop rows where y is invalid; NaN in X will be filled by KNNImputer in build_features.
    mask = y.notna() & (y > 0) & (y < 50)
    X, y = X[mask].copy(), y[mask].copy()

    for c in LOG_COLS:
        X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    y = np.log1p(y)

    return X, y


def build_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    embed_cols = [c for c in X_train.columns if c.startswith("embed_")]

    # KNNImputer fills missing quant values; fit on train only to avoid leakage.
    imputer = KNNImputer(n_neighbors=5)
    quant_train_imp = imputer.fit_transform(X_train[QUANT_COLS])
    quant_test_imp = imputer.transform(X_test[QUANT_COLS])

    scaler = StandardScaler()
    quant_train = scaler.fit_transform(quant_train_imp)
    quant_test = scaler.transform(quant_test_imp)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    embed_train = pca.fit_transform(X_train[embed_cols].values)
    embed_test = pca.transform(X_test[embed_cols].values)

    X_all_train = np.column_stack([quant_train, embed_train])
    X_all_test = np.column_stack([quant_test, embed_test])
    return X_all_train, X_all_test, imputer, scaler, pca, embed_cols


def train(test_size: float = 0.2, random_state: int = 42):
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_all_train, X_all_test, imputer, scaler, pca, embed_cols = build_features(X_train, X_test)

    models = []
    preds = []
    for seed in BAG_SEEDS:
        m = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=0,
        )
        m.fit(X_all_train, y_train)
        models.append(m)
        preds.append(m.predict(X_all_test))
    y_pred = np.mean(preds, axis=0)

    return {
        "models": models,
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "embed_cols": embed_cols,
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
    }


def predict_ev_to_ebitda(X: pd.DataFrame, models, imputer, scaler, pca, embed_cols):
    X = X.copy()
    for c in LOG_COLS:
        X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    quant = scaler.transform(imputer.transform(X[QUANT_COLS]))
    embed = pca.transform(X[embed_cols].values)
    X_all = np.column_stack([quant, embed])
    # Each model predicts log1p(target); average then invert for the EV/EBITDA scale.
    preds = np.mean([m.predict(X_all) for m in models], axis=0)
    return np.expm1(preds)


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


def predict_ticker(ticker: str, models, imputer, scaler, pca, embed_cols) -> dict:
    """Predict EV/EBITDA for a live ticker. Returns {'predicted', 'actual'}."""
    import yfinance as yf
    info = yf.Ticker(ticker).info
    X = _features_from_info(ticker, info)
    predicted = float(predict_ev_to_ebitda(X, models, imputer, scaler, pca, embed_cols)[0])
    actual = info.get("enterpriseToEbitda")
    return {"predicted": predicted, "actual": actual}


def save_artifacts(out: dict, path: Path = MODEL_PATH) -> None:
    joblib.dump(
        {k: out[k] for k in ("models", "imputer", "scaler", "pca", "embed_cols", "r2", "mse")},
        path,
    )


def load_artifacts(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)
