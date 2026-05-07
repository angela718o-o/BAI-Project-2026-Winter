import sys

from . import MODEL_PATH, predict_ticker, save_artifacts, train

out = train()
save_artifacts(out)
print(f"Test R2:  {out['r2']:.4f}")
print(f"Test MSE: {out['mse']:.4f}")
print(f"Saved -> {MODEL_PATH}")

for t in sys.argv[1:]:
    try:
        r = predict_ticker(t, out["model"], out["scaler"], out["embed_cols"])
        actual = r["actual"]
        actual_str = f"{actual:.2f}" if actual is not None else "N/A"
        diff_str = f", diff={r['predicted'] - actual:+.2f}" if actual is not None else ""
        print(f"{t}: predicted={r['predicted']:.2f}, actual={actual_str}{diff_str}")
    except Exception as e:
        print(f"{t}: failed — {e}")
