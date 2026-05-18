"""
save_model.py
─────────────
【這個檔案在做什麼】
用完整的 features.csv 訓練最終模型（XGBoost），
然後把訓練好的 pipeline 存成 model.joblib，
讓 Streamlit app 啟動時直接載入，不需要每次重新訓練。

【為什麼要存模型】
Streamlit 每次有使用者互動都會重跑整個腳本，
如果每次都重新訓練會很慢（而且需要 features.csv 在雲端）。
把模型存成檔案，app 只需要載入，啟動時間 < 1 秒。
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

BASE      = Path(__file__).parent
FEAT_FILE = BASE / "features.csv"
MODEL_OUT = BASE / "model.joblib"

FEATURES = [
    "age_start", "sex", "laterality",
    "myopia_d", "axl_baseline", "axl_dev_from_norm",
    "mi_rx", "atropine_conc",
    "early_slope_6m", "axl_std_6m",
]
TARGET = "y_annual_mm_yr"

df  = pd.read_csv(FEAT_FILE)
sub = df[FEATURES + [TARGET]].dropna(subset=[TARGET])
X   = sub[FEATURES].values
y   = sub[TARGET].values

model = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("mdl", XGBRegressor(
        n_estimators=200, max_depth=3,
        learning_rate=0.05, subsample=0.8,
        random_state=42, verbosity=0,
    )),
])
model.fit(X, y)

# 同時存 imputer 填補用的中位數（供 app 顯示預設值）
imp_medians = dict(zip(FEATURES, model.named_steps["imp"].statistics_))

joblib.dump({"model": model, "features": FEATURES,
             "medians": imp_medians, "n_train": len(y)}, MODEL_OUT)
print(f"✓ 模型已存至 {MODEL_OUT}  (n={len(y)})")
print("  訓練集 RMSE (in-sample):",
      round(float(np.sqrt(((model.predict(X) - y)**2).mean())), 4))