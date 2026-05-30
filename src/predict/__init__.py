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

DATA_PATH    = Path(__file__).resolve().parents[2] / "data" / "processed" / "combined_features.csv"
METRICS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "cleaned_metrics.csv"
MODEL_PATH   = Path(__file__).resolve().parents[2] / "data" / "processed" / "model.joblib"
TARGET = "ev_to_ebitda"
# All 7 quant features used in the notebook's best model (Plan D).
QUANT_COLS = [
    "shares_outstanding", "total_debt", "total_cash", "ebitda",
    "debt_to_equity", "ebitda_margin", "forwardPE",
]
CATEGORICAL_COLS = ["sector", "industry"]
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
PCA_COMPONENTS = 20
BAG_SEEDS = [42, 7, 13, 99, 2024]

# Optuna best params from notebook (Plan D: test R² ≈ 0.54, Spearman ρ ≈ 0.75).
_BEST_PARAMS = {
    "n_estimators": 469,
    "max_depth": 8,
    "learning_rate": 0.018526066660175013,
    "subsample": 0.7242793234086645,
    "colsample_bytree": 0.825868423791473,
    "min_child_weight": 2,
    "reg_alpha": 0.24780887061061857,
    "reg_lambda": 3.1678045863649835,
    "verbosity": 0,
}

_YF_TO_FEATURE = {
    "shares_outstanding": "sharesOutstanding",
    "total_debt": "totalDebt",
    "total_cash": "totalCash",
    "ebitda": "ebitda",
    "debt_to_equity": "debtToEquity",
    "ebitda_margin": "ebitdaMargins",
    "forwardPE": "forwardPE",  # optional — imputed by KNNImputer if missing
}

# Fields that cannot be meaningfully imputed and are required for prediction.
_REQUIRED_FIELDS = {"shares_outstanding", "total_debt", "total_cash", "ebitda", "debt_to_equity", "ebitda_margin"}

_embed_model = None


def load_dataset(path: Path = DATA_PATH, metrics_path: Path = METRICS_PATH):
    df = pd.read_csv(path, index_col="ticker")
    meta = pd.read_csv(metrics_path)[["ticker", "sector", "industry"]].set_index("ticker")
    df = df.join(meta, how="left")

    X = df.drop(columns=[TARGET]).replace([np.inf, -np.inf], np.nan)
    y = df[TARGET].replace([np.inf, -np.inf], np.nan)

    mask = y.notna() & (y > 0) & (y < 50)
    X, y = X[mask].copy(), y[mask].copy()

    # Clip debt_to_equity at 99th pct — extreme outliers skew the model.
    dte_cap = X["debt_to_equity"].quantile(0.99)
    X["debt_to_equity"] = X["debt_to_equity"].clip(upper=dte_cap)

    # Derive net_debt_to_ebitda (available in X but not in QUANT_COLS — kept for completeness).
    X["net_debt_to_ebitda"] = (
        (X["total_debt"] - X["total_cash"]) / X["ebitda"].replace(0, np.nan)
    ).clip(-15, 30)

    y = np.log1p(y)
    return X, y


def build_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    embed_cols = [c for c in X_train.columns if c.startswith("embed_")]

    imputer = KNNImputer(n_neighbors=5)
    quant_train_imp = imputer.fit_transform(X_train[QUANT_COLS])
    quant_test_imp  = imputer.transform(X_test[QUANT_COLS])

    scaler = StandardScaler()
    quant_train = scaler.fit_transform(quant_train_imp)
    quant_test  = scaler.transform(quant_test_imp)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    embed_train = pca.fit_transform(X_train[embed_cols].values)
    embed_test  = pca.transform(X_test[embed_cols].values)

    ohe_train = pd.get_dummies(X_train[CATEGORICAL_COLS].fillna("Unknown"), dtype=float)
    ohe_test  = pd.get_dummies(X_test[CATEGORICAL_COLS].fillna("Unknown"),  dtype=float).reindex(
        columns=ohe_train.columns, fill_value=0.0)
    ohe_cols = ohe_train.columns.tolist()

    X_all_train = np.column_stack([quant_train, embed_train, ohe_train.values])
    X_all_test  = np.column_stack([quant_test,  embed_test,  ohe_test.values])
    return X_all_train, X_all_test, imputer, scaler, pca, ohe_cols, embed_cols


def train(test_size: float = 0.2, random_state: int = 42):
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_all_train, X_all_test, imputer, scaler, pca, ohe_cols, embed_cols = build_features(X_train, X_test)

    models = []
    preds = []
    for seed in BAG_SEEDS:
        m = xgb.XGBRegressor(**_BEST_PARAMS, random_state=seed)
        m.fit(X_all_train, y_train)
        models.append(m)
        preds.append(m.predict(X_all_test))
    y_pred = np.mean(preds, axis=0)

    return {
        "models": models,
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "ohe_cols": ohe_cols,
        "embed_cols": embed_cols,
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
    }


def predict_ev_to_ebitda(X: pd.DataFrame, models, imputer, scaler, pca, ohe_cols, embed_cols):
    X = X.copy()
    quant = scaler.transform(imputer.transform(X[QUANT_COLS]))
    embed = pca.transform(X[embed_cols].values)
    ohe   = pd.get_dummies(X[CATEGORICAL_COLS].fillna("Unknown"), dtype=float).reindex(
        columns=ohe_cols, fill_value=0.0).values
    X_all = np.column_stack([quant, embed, ohe])
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

    missing_required = [k for k in _REQUIRED_FIELDS if quant.get(k) is None]
    if missing_required:
        raise ValueError(f"yfinance missing required fields for {ticker}: {missing_required}")

    # forwardPE is optional — KNNImputer fills it using training-set neighbors.
    if quant.get("forwardPE") is None:
        quant["forwardPE"] = np.nan

    summary = info.get("longBusinessSummary") or ""
    if not summary:
        raise ValueError(f"yfinance returned no business summary for {ticker}")

    ebitda = quant["ebitda"] or np.nan
    net_debt_to_ebitda = float(np.clip(
        (quant["total_debt"] - quant["total_cash"]) / ebitda if ebitda else np.nan,
        -15, 30,
    ))

    embedding = _get_embed_model().encode([summary])[0]
    row = {
        **quant,
        "net_debt_to_ebitda": net_debt_to_ebitda,
        "sector":   info.get("sector")   or "Unknown",
        "industry": info.get("industry") or "Unknown",
        **{f"embed_{i}": float(v) for i, v in enumerate(embedding)},
    }
    return pd.DataFrame([row], index=pd.Index([ticker], name="ticker"))


def fetch_ticker_features(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    return _features_from_info(ticker, yf.Ticker(ticker).info)


def predict_ticker(ticker: str, models, imputer, scaler, pca, ohe_cols, embed_cols) -> dict:
    """Predict EV/EBITDA for a live ticker. Returns {'predicted', 'actual', 'info'}."""
    import yfinance as yf
    info = yf.Ticker(ticker).info
    return predict_ticker_with_info(ticker, info, models, imputer, scaler, pca, ohe_cols, embed_cols)


def predict_ticker_with_info(ticker: str, info: dict, models, imputer, scaler, pca, ohe_cols, embed_cols) -> dict:
    """Like predict_ticker but accepts a pre-fetched yfinance info dict. Returns {'predicted', 'actual', 'info'}."""
    X = _features_from_info(ticker, info)
    predicted = float(predict_ev_to_ebitda(X, models, imputer, scaler, pca, ohe_cols, embed_cols)[0])
    actual = info.get("enterpriseToEbitda")
    return {"predicted": predicted, "actual": actual, "info": info}


def save_artifacts(out: dict, path: Path = MODEL_PATH) -> None:
    joblib.dump(
        {k: out[k] for k in ("models", "imputer", "scaler", "pca", "ohe_cols", "embed_cols", "r2", "mse")},
        path,
    )


def load_artifacts(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)
