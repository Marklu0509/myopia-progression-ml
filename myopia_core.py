"""
myopia_core.py — 純預測核心（無 UI 框架依賴）
────────────────────────────────────────────
把原 Streamlit app.py 的預測邏輯抽出來，供 FastAPI (server.py) 呼叫：
載入模型 → 組裝特徵 → 預測年化眼軸進展 → 風險分級 + 文獻對照 + 東亞常模百分位 + SHAP。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import joblib

# ── 載入模型 bundle ───────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model.joblib"
_bundle = joblib.load(MODEL_PATH)
MODEL = _bundle["model"]
FEATURES: list[str] = _bundle["features"]
MEDIANS: dict[str, float] = _bundle["medians"]
N_TRAIN: int = _bundle["n_train"]

# ── 常數對照表 ────────────────────────────────────────────────
ATRO_MAP = {"None (0%)": 0.0, "0.01%": 0.01, "0.02%": 0.02,
            "0.05%": 0.05, "0.125%": 0.125}

# 文獻參考值 (mm/yr, age_range, ethnicity)
RCT_REF = [
    ("Single Vision control · Liu 2021", 0.36, "6–12 yrs", "East Asian · Singapore"),
    ("MiSight 1yr · Chamberlain 2019",   0.13, "8–12 yrs", "Multi-ethnic · EU/US/NZ/SG"),
    ("MiSight 3yr · Chamberlain 2019",   0.15, "8–12 yrs", "Multi-ethnic · EU/US/NZ/SG"),
    ("Atropine 0.01% · ATOM2 2012",      0.28, "6–12 yrs", "East Asian · Singapore"),
    ("Atropine 0.1% · LAMP 2019",        0.19, "6–12 yrs", "East Asian · Hong Kong"),
]

# 東亞兒童眼軸常模中位數（He et al. Br J Ophthalmol 2023, n=14,127）
EA_NORM_MALE = {
    4: 22.39, 5: 22.69, 6: 22.97, 7: 23.25, 8: 23.51, 9: 23.76, 10: 23.99,
    11: 24.22, 12: 24.43, 13: 24.62, 14: 24.81, 15: 24.98, 16: 25.13,
    17: 25.28, 18: 25.41, 19: 25.50, 20: 25.55,
}
EA_NORM_FEMALE = {
    4: 21.78, 5: 22.10, 6: 22.41, 7: 22.70, 8: 22.98, 9: 23.25, 10: 23.51,
    11: 23.75, 12: 23.97, 13: 24.19, 14: 24.39, 15: 24.57, 16: 24.75,
    17: 24.91, 18: 25.05, 19: 25.12, 20: 25.15,
}
EA_PCTILE = {
    7:  dict(p10_m=22.43, p50_m=23.25, p90_m=24.08, p10_f=21.89, p50_f=22.70, p90_f=23.51),
    8:  dict(p10_m=22.67, p50_m=23.51, p90_m=24.35, p10_f=22.08, p50_f=22.98, p90_f=23.88),
    9:  dict(p10_m=22.90, p50_m=23.76, p90_m=24.62, p10_f=22.28, p50_f=23.25, p90_f=24.22),
    10: dict(p10_m=23.11, p50_m=23.99, p90_m=24.87, p10_f=22.47, p50_f=23.51, p90_f=24.55),
    11: dict(p10_m=23.30, p50_m=24.22, p90_m=25.14, p10_f=22.65, p50_f=23.75, p90_f=24.85),
    12: dict(p10_m=23.49, p50_m=24.43, p90_m=25.37, p10_f=22.82, p50_f=23.97, p90_f=25.12),
    13: dict(p10_m=23.66, p50_m=24.62, p90_m=25.58, p10_f=22.98, p50_f=24.19, p90_f=25.40),
    14: dict(p10_m=23.82, p50_m=24.81, p90_m=25.80, p10_f=23.13, p50_f=24.39, p90_f=25.65),
    15: dict(p10_m=23.97, p50_m=24.98, p90_m=25.99, p10_f=23.27, p50_f=24.57, p90_f=25.87),
    16: dict(p10_m=24.10, p50_m=25.13, p90_m=26.16, p10_f=23.40, p50_f=24.75, p90_f=26.10),
    17: dict(p10_m=24.22, p50_m=25.28, p90_m=26.34, p10_f=23.52, p50_f=24.91, p90_f=26.30),
    18: dict(p10_m=24.33, p50_m=25.41, p90_m=26.49, p10_f=23.63, p50_f=25.05, p90_f=26.47),
}

FEAT_LABELS = {
    "age_start": "Age", "sex": "Sex", "laterality": "Eye (OD/OS)",
    "myopia_d": "Baseline myopia (D)", "axl_baseline": "Baseline AXL (mm)",
    "axl_dev_from_norm": "AXL deviation from norm",
    "mi_rx": "MiSight Rx power", "atropine_conc": "Atropine conc.",
    "early_slope_6m": "Early slope 0–6m", "axl_std_6m": "AXL variability",
}

# ── SHAP explainer（啟動時建一次）────────────────────────────
_explainer = None
_shap_bg = None


def _get_explainer():
    """延遲建立 SHAP explainer；優先 TreeExplainer，失敗退回 KernelExplainer。"""
    global _explainer, _shap_bg
    if _explainer is not None:
        return _explainer
    import shap
    imp = MODEL.named_steps["imp"]
    raw_mdl = MODEL.named_steps["mdl"]
    try:
        _explainer = ("tree", shap.TreeExplainer(raw_mdl.get_booster()))
    except Exception:
        _shap_bg = imp.transform(np.array([[MEDIANS.get(f, 0) for f in FEATURES]]))
        _explainer = ("kernel", shap.KernelExplainer(raw_mdl.predict, _shap_bg))
    return _explainer


# ── 純函式 ────────────────────────────────────────────────────
def get_ea_norm(age_val: int, sex_str: str) -> float:
    if sex_str == "Male":
        return EA_NORM_MALE.get(age_val, EA_NORM_MALE[18])
    return EA_NORM_FEMALE.get(age_val, EA_NORM_FEMALE[18])


def axl_percentile_approx(axl: float, age_val: int, sex_str: str) -> float:
    """線性插值估計 AXL 在同齡同性別東亞兒童的百分位（0–100）。"""
    pct = EA_PCTILE.get(age_val)
    if pct is None:
        return 50.0
    if sex_str == "Male":
        p10, p50, p90 = pct["p10_m"], pct["p50_m"], pct["p90_m"]
    else:
        p10, p50, p90 = pct["p10_f"], pct["p50_f"], pct["p90_f"]
    if axl <= p10:
        return float(np.interp(axl, [p10 - 2.0, p10], [2, 10]))
    if axl <= p50:
        return float(np.interp(axl, [p10, p50], [10, 50]))
    if axl <= p90:
        return float(np.interp(axl, [p50, p90], [50, 90]))
    return float(np.interp(axl, [p90, p90 + 2.0], [90, 98]))


def risk_tier(y: float) -> dict[str, str]:
    if y >= 0.30:
        return {"label": "Fast Progressor", "key": "fast", "color": "#B0472E",
                "advice": "This patient may benefit from a treatment review. Escalating adjunct "
                          "therapy (e.g. higher atropine concentration) could be worth discussing "
                          "with the patient and family, based on clinical judgment and individual response."}
    if y >= 0.10:
        return {"label": "Moderate Progressor", "key": "moderate", "color": "#C08A2E",
                "advice": "Current treatment may be providing reasonable control. A closer follow-up "
                          "interval (e.g. every 3 months) might help monitor whether progression is "
                          "stabilizing. Clinical context should guide any treatment decisions."}
    return {"label": "Slow Progressor", "key": "slow", "color": "#3F8F5B",
            "advice": "Progression appears relatively well-controlled under current treatment. "
                      "Continuing with routine follow-up may be appropriate, though individual "
                      "variation should be considered."}


def _parse_visits(visits: list[dict]) -> list[tuple[float, float]]:
    out = []
    for v in visits or []:
        axl = v.get("axl", None)
        if axl in (None, ""):
            continue
        try:
            out.append((float(v.get("month", 0)), float(axl)))
        except (TypeError, ValueError):
            continue
    return sorted(out, key=lambda x: x[0])


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """核心預測：接收前端 payload，回傳完整結果 dict。"""
    # ── 驗證輸入 ──
    try:
        age = int(round(float(payload["age"])))
    except (KeyError, TypeError, ValueError):
        raise ValueError("age is required and must be a number")
    age = max(6, min(20, age))
    sex = "Male" if str(payload.get("sex", "Female")).lower().startswith("m") else "Female"
    eye = str(payload.get("eye", "OD"))
    myopia_d = float(payload.get("myopia_d", -3.0))
    mi_rx = float(payload.get("mi_rx", -2.0))
    atro = payload.get("atropine", "None (0%)")
    if atro not in ATRO_MAP:
        atro = "None (0%)"
    fallback = float(payload.get("axl_fallback", 24.0))

    # ── 從 visits 推 baseline / slope / std ──
    vv = _parse_visits(payload.get("visits", []))
    early_slope = float(MEDIANS.get("early_slope_6m", 0.015))
    axl_std = float(MEDIANS.get("axl_std_6m", 0.05))
    slope_ok = False
    if len(vv) >= 1:
        axl_base = vv[0][1]
        if len(vv) >= 2:
            m_arr = np.array([v[0] for v in vv])
            a_arr = np.array([v[1] for v in vv])
            early_slope = float(np.polyfit(m_arr, a_arr, 1)[0])
            axl_std = float(a_arr.std())
            slope_ok = True
        else:
            axl_std = 0.0
    else:
        axl_base = fallback

    norm_axl = get_ea_norm(age, sex)
    axl_dev = axl_base - norm_axl

    input_vals = {
        "age_start":         float(age),
        "sex":               1.0 if sex == "Male" else 0.0,
        "laterality":        0.0 if "OD" in eye else 1.0,
        "myopia_d":          float(myopia_d),
        "axl_baseline":      float(axl_base),
        "axl_dev_from_norm": float(axl_dev),
        "mi_rx":             float(mi_rx),
        "atropine_conc":     ATRO_MAP[atro],
        "early_slope_6m":    float(early_slope),
        "axl_std_6m":        float(axl_std),
    }
    X = np.array([[input_vals[f] for f in FEATURES]])
    y = float(MODEL.predict(X)[0])

    tier = risk_tier(y)
    pctile = axl_percentile_approx(float(axl_base), age, sex)

    literature = [
        {"name": name, "ref": ref, "meta": f"{age_range} · {eth}",
         "ea": "East Asian" in eth, "delta": y - ref}
        for name, ref, age_range, eth in RCT_REF
    ]

    shap_items, base_val = _compute_shap(X, input_vals)

    # 病人軌跡點（實際年齡 = age + month/12）
    patient_points = ([{"age": age + m / 12.0, "axl": a} for m, a in vv]
                      if vv else [{"age": float(age), "axl": float(axl_base)}])

    return {
        "prediction": y,
        "tier": tier,
        "age": age,
        "sex": sex,
        "axl_base": float(axl_base),
        "norm_axl": float(norm_axl),
        "axl_dev": float(axl_dev),
        "percentile": float(pctile),
        "early_slope": float(early_slope),   # mm/month
        "slope_ok": slope_ok,
        "base_value": float(base_val),
        "literature": literature,
        "shap": shap_items,
        "patient_points": patient_points,
        "n_train": N_TRAIN,
    }


def _compute_shap(X: np.ndarray, input_vals: dict[str, float]):
    """回傳 (list[{name,val,shap}], base_value)。失敗時回傳空 list。"""
    try:
        kind, explainer = _get_explainer()
        imp = MODEL.named_steps["imp"]
        X_imp = imp.transform(X)
        if kind == "tree":
            sv = explainer.shap_values(X_imp)[0]
        else:
            sv = explainer.shap_values(X_imp, nsamples=100)[0]
        base_val = float(np.ravel(explainer.expected_value)[0])
        items = [
            {"name": FEAT_LABELS.get(f, f),
             "val": f"{input_vals[f]:.3g}",
             "shap": float(sv[i])}
            for i, f in enumerate(FEATURES)
        ]
        return items, base_val
    except Exception:
        return [], float(np.mean([0.0]))
