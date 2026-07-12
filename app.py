"""
app.py — Myopia Progression Predictor
──────────────────────────────────────
【這個檔案在做什麼】
Streamlit web app，讓使用者（醫師或面試官）輸入臨床參數，
即時看到模型預測的年化眼軸進展速度 + 風險分級 + SHAP 特徵貢獻。

【部署方式】
本機執行：streamlit run app.py
雲端部署：上傳至 Streamlit Cloud（https://streamlit.io/cloud），
          連結 GitHub repo 即可，免費且無需伺服器。

【頁面結構】
Sidebar  — 臨床參數輸入（sliders + radio buttons）
Main     — 預測結果 + 風險分級 + 文獻對照 + SHAP waterfall
Footer   — 資料聲明與模型資訊
"""

import numpy as np
import joblib
import streamlit as st
from pathlib import Path
from datetime import date, timedelta

# ── 頁面設定（必須是第一個 st 指令）────────────────────────────
st.set_page_config(
    page_title="Myopia Progression Predictor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 載入模型 ──────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    bundle  = load_model()
    model   = bundle["model"]
    FEATURES = bundle["features"]
    medians = bundle["medians"]
    n_train = bundle["n_train"]
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    st.error(f"模型載入失敗：{e}")

# ── 文獻參考值 ────────────────────────────────────────────────
# (mm/yr, age_range, ethnicity_note)
RCT_REF = {
    "Single Vision (control)\n(Liu 2021)"        : (0.36, "6–12 yrs", "East Asian · Singapore"),
    "MiSight 1yr (Chamberlain 2019)"             : (0.13, "8–12 yrs", "Multi-ethnic · EU/US/NZ/SG"),
    "MiSight 3yr (Chamberlain 2019)"             : (0.15, "8–12 yrs", "Multi-ethnic · EU/US/NZ/SG"),
    "Atropine 0.01% · ATOM2 (Chia 2012)"         : (0.28, "6–12 yrs", "East Asian · Singapore"),
    "Atropine 0.1% · LAMP (Yam 2019)"            : (0.19, "6–12 yrs", "East Asian · Hong Kong"),
}

# ── 風險分級 ──────────────────────────────────────────────────
def risk_tier(y: float):
    if y >= 0.30:
        return "🔴 Fast Progressor", "#C00000", \
               "This patient may benefit from a treatment review. " \
               "Escalating adjunct therapy (e.g., higher atropine concentration) could be worth discussing with the patient and family, " \
               "based on clinical judgment and individual response."
    elif y >= 0.10:
        return "🟡 Moderate Progressor", "#ED7D31", \
               "Current treatment may be providing reasonable control. " \
               "A closer follow-up interval (e.g., every 3 months) might help monitor whether progression is stabilizing. " \
               "Clinical context should guide any treatment decisions."
    else:
        return "🟢 Slow Progressor", "#375623", \
               "Progression appears relatively well-controlled under current treatment. " \
               "Continuing with routine follow-up may be appropriate, though individual variation should be considered."

# ══════════════════════════════════════════════════════════════
# Sidebar — 輸入參數
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("👁️ Patient Parameters")
    st.caption("Adjust sliders to match patient profile")

    st.markdown("#### 📋 Baseline Demographics")
    age   = st.slider("Age at treatment start (years)", 6, 20, 11)
    sex   = st.radio("Sex", ["Female", "Male"], horizontal=True)
    lat   = st.radio("Eye", ["OD (Right)", "OS (Left)"], horizontal=True)

    st.markdown("#### 🔬 Baseline Ocular")
    myopia_d = st.slider("Baseline myopia (D)", -12.0, -0.25,
                         float(round(medians.get("myopia_d", -3.0), 2)),
                         step=0.25, format="%.2f")

    st.markdown("#### 💊 Treatment")
    mi_rx = st.slider("MiSight Rx power (D)", -6.0, -0.5, -2.0, step=0.25, format="%.2f")
    atro  = st.selectbox("Adjunct atropine", [
        "None (0%)", "0.01%", "0.02%", "0.05%", "0.125%"
    ])

    st.markdown("#### 📈 Early AXL Measurements")
    st.caption("Visit 1 = treatment start (baseline). Enter up to 4 visits.")

    # 輸入方式選擇
    axl_input_mode = st.radio(
        "Input mode", ["Enter dates + AXL", "Enter months + AXL directly"],
        horizontal=True
    )

    NUM_PTS = 4
    raw_visits = []   # list of (month_float, axl_float)

    if axl_input_mode == "Enter dates + AXL":
        st.markdown("**Date & AXL per visit** *(Visit 1 = treatment start date)*")
        dates_axls = []
        for i in range(NUM_PTS):
            c1, c2 = st.columns(2)
            with c1:
                vd = st.date_input(
                    f"Visit {i+1} date",
                    value=None,
                    key=f"vdate_{i}",
                )
            with c2:
                va = st.number_input(
                    f"AXL {i+1} (mm)",
                    min_value=20.0, max_value=30.0,
                    value=None, step=0.01, format="%.2f",
                    key=f"vaxl_{i}",
                    placeholder="mm"
                )
            if vd is not None and va is not None:
                dates_axls.append((vd, float(va)))

        # Visit 1 = month 0 (baseline)
        if len(dates_axls) >= 1:
            base_date = dates_axls[0][0]
            for vd, va in dates_axls:
                mo = (vd - base_date).days / 30.44
                raw_visits.append((mo, va))

    else:  # Enter months directly
        st.markdown("**Visit 1 = month 0 (baseline). Enter months + AXL:**")
        for i in range(NUM_PTS):
            c1, c2 = st.columns(2)
            label_mo  = "Month (0 = start)" if i == 0 else f"Month {i+1}"
            label_axl = "AXL (mm)"
            with c1:
                mo = st.number_input(
                    label_mo,
                    min_value=0.0, max_value=8.0,
                    value=None, step=0.5, format="%.1f",
                    key=f"vmo_{i}",
                    placeholder="0 = start"
                )
            with c2:
                va = st.number_input(
                    label_axl,
                    min_value=20.0, max_value=30.0,
                    value=None, step=0.01, format="%.2f",
                    key=f"vaxl2_{i}",
                    placeholder="mm"
                )
            if mo is not None and va is not None:
                raw_visits.append((float(mo), float(va)))

    # ── 從 visits 決定 baseline AXL + 計算斜率 ──────────────────
    # baseline = Visit 1 的 AXL（month 0）；若沒有填則 fallback 到 slider
    # 只有在完全沒填 visit 時才顯示 fallback slider；有 visit 就隱藏（改用 Visit 1）
    if len(raw_visits) == 0:
        axl_slider = st.slider(
            "Baseline AXL fallback (mm)",
            21.0, 28.0,
            float(round(medians.get("axl_baseline", 24.0), 1)),
            step=0.05, format="%.2f",
            help="Used only if no visits are entered above",
        )
    else:
        axl_slider = float(round(medians.get("axl_baseline", 24.0), 1))

    early_slope = float(medians.get("early_slope_6m", 0.015))
    axl_std     = float(medians.get("axl_std_6m", 0.05))
    slope_ok    = False

    # 有效的 visits（month 必須遞增、AXL 合理）
    valid_visits = sorted(raw_visits, key=lambda x: x[0])

    if len(valid_visits) >= 1:
        # Visit 1 的 AXL 就是 baseline
        axl_base = valid_visits[0][1]
        if len(valid_visits) >= 2:
            m_arr = np.array([v[0] for v in valid_visits])
            a_arr = np.array([v[1] for v in valid_visits])
            coeffs      = np.polyfit(m_arr, a_arr, 1)
            early_slope = float(coeffs[0])
            axl_std     = float(a_arr.std())
            slope_ok    = True
            st.success(
                f"✓ {len(valid_visits)} visits  |  "
                f"Baseline AXL = **{axl_base:.2f} mm** (Visit 1)  |  "
                f"Slope = **{early_slope:+.4f} mm/month**"
            )
            st.caption(f"≈ {early_slope * 12:+.3f} mm/yr annualized · SD = {axl_std:.4f} mm")
        else:
            axl_std = 0.0
            st.warning(
                f"Baseline AXL = {axl_base:.2f} mm (Visit 1). "
                "Enter at least one more visit to calculate slope."
            )
    else:
        axl_base = axl_slider
        st.info("No visits entered — using fallback slider for baseline AXL and cohort median for slope")

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Progression", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════
# 主畫面
# ══════════════════════════════════════════════════════════════
st.title("👁️ Myopia Progression Predictor")
st.markdown(
    "**ML-powered clinical decision support** for pediatric myopia management. "
    "Enter patient parameters in the sidebar to predict annualized axial length progression."
)
st.caption(
    f"Model: XGBoost · Validated via LOOCV · "
    f"Training cohort: n={n_train if MODEL_OK else '—'} eyes · "
    "⚠️ For research/demonstration only — not for clinical use"
)
st.divider()

# ── 解析輸入 ──────────────────────────────────────────────────
atro_map = {"None (0%)": 0.0, "0.01%": 0.01, "0.02%": 0.02,
            "0.05%": 0.05, "0.125%": 0.125}

# ── 東亞兒童眼軸常模（He et al. Br J Ophthalmol 2023, n=14,127）────
# 50th percentile（中位數），分性別，來源：上海+廣州大型研究
EA_NORM_MALE = {
    4:22.39,5:22.69,6:22.97,7:23.25,8:23.51,9:23.76,10:23.99,
    11:24.22,12:24.43,13:24.62,14:24.81,15:24.98,16:25.13,
    17:25.28,18:25.41,19:25.50,20:25.55,
}
EA_NORM_FEMALE = {
    4:21.78,5:22.10,6:22.41,7:22.70,8:22.98,9:23.25,10:23.51,
    11:23.75,12:23.97,13:24.19,14:24.39,15:24.57,16:24.75,
    17:24.91,18:25.05,19:25.12,20:25.15,
}
# P10 / P50 / P90 對照表（用於百分位計算 panel）
EA_PCTILE = {
    7:  dict(p10_m=22.43,p50_m=23.25,p90_m=24.08, p10_f=21.89,p50_f=22.70,p90_f=23.51),
    8:  dict(p10_m=22.67,p50_m=23.51,p90_m=24.35, p10_f=22.08,p50_f=22.98,p90_f=23.88),
    9:  dict(p10_m=22.90,p50_m=23.76,p90_m=24.62, p10_f=22.28,p50_f=23.25,p90_f=24.22),
    10: dict(p10_m=23.11,p50_m=23.99,p90_m=24.87, p10_f=22.47,p50_f=23.51,p90_f=24.55),
    11: dict(p10_m=23.30,p50_m=24.22,p90_m=25.14, p10_f=22.65,p50_f=23.75,p90_f=24.85),
    12: dict(p10_m=23.49,p50_m=24.43,p90_m=25.37, p10_f=22.82,p50_f=23.97,p90_f=25.12),
    13: dict(p10_m=23.66,p50_m=24.62,p90_m=25.58, p10_f=22.98,p50_f=24.19,p90_f=25.40),
    14: dict(p10_m=23.82,p50_m=24.81,p90_m=25.80, p10_f=23.13,p50_f=24.39,p90_f=25.65),
    15: dict(p10_m=23.97,p50_m=24.98,p90_m=25.99, p10_f=23.27,p50_f=24.57,p90_f=25.87),
    16: dict(p10_m=24.10,p50_m=25.13,p90_m=26.16, p10_f=23.40,p50_f=24.75,p90_f=26.10),
    17: dict(p10_m=24.22,p50_m=25.28,p90_m=26.34, p10_f=23.52,p50_f=24.91,p90_f=26.30),
    18: dict(p10_m=24.33,p50_m=25.41,p90_m=26.49, p10_f=23.63,p50_f=25.05,p90_f=26.47),
}

def get_ea_norm(age_val: int, sex_str: str) -> float:
    if sex_str == "Male":
        return EA_NORM_MALE.get(age_val, EA_NORM_MALE.get(18, 25.41))
    return EA_NORM_FEMALE.get(age_val, EA_NORM_FEMALE.get(18, 25.05))

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
    elif axl <= p50:
        return float(np.interp(axl, [p10, p50], [10, 50]))
    elif axl <= p90:
        return float(np.interp(axl, [p50, p90], [50, 90]))
    else:
        return float(np.interp(axl, [p90, p90 + 2.0], [90, 98]))

norm_axl = get_ea_norm(age, sex)
axl_dev  = axl_base - norm_axl

input_vals = {
    "age_start"         : float(age),
    "sex"               : 1.0 if sex == "Male" else 0.0,
    "laterality"        : 0.0 if "OD" in lat else 1.0,
    "myopia_d"          : float(myopia_d),
    "axl_baseline"      : float(axl_base),
    "axl_dev_from_norm" : axl_dev,
    "mi_rx"             : float(mi_rx),
    "atropine_conc"     : atro_map[atro],
    "early_slope_6m"    : float(early_slope),
    "axl_std_6m"        : float(axl_std),
}

X_input = np.array([[input_vals[f] for f in FEATURES]])

# ── 預測 ──────────────────────────────────────────────────────
if MODEL_OK and predict_btn:
    y_pred = float(model.predict(X_input)[0])
    tier_label, tier_color, tier_advice = risk_tier(y_pred)

    # 結果區塊
    col1, col2, col3 = st.columns([1.2, 1, 1.8])

    with col1:
        st.markdown("### Predicted Progression")
        st.markdown(
            f"<div style='background:{tier_color}22; border-left:5px solid {tier_color}; "
            f"padding:18px 22px; border-radius:8px;'>"
            f"<span style='font-size:2.6rem; font-weight:800; color:{tier_color};'>"
            f"{y_pred:.3f}</span>"
            f"<span style='font-size:1rem; color:#555;'> mm/yr</span><br>"
            f"<span style='font-size:1.1rem; font-weight:600; color:{tier_color};'>"
            f"{tier_label}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(f"> {tier_advice}")
        st.markdown(
            f"**AXL deviation from age norm:** {axl_dev:+.2f} mm "
            f"(norm for age {age}: {norm_axl} mm)"
        )

    with col2:
        st.markdown("### vs. Literature")
        st.caption("⚠️ East Asian children progress faster than multi-ethnic cohorts — compare within same ethnicity for fairness.")
        for study, (ref_val, age_range, ethnicity) in RCT_REF.items():
            delta = y_pred - ref_val
            arrow = "▲" if delta > 0 else "▼"
            color = "#C00000" if delta > 0.05 else ("#375623" if delta < -0.05 else "#555")
            # 標示 East Asian 研究（和你的台灣資料族群相近）
            ea_badge = " 🟡" if "East Asian" in ethnicity else " ⚪"
            st.markdown(
                f"<div style='margin-bottom:10px; padding:6px 8px; "
                f"border-radius:5px; background:{'#FFFBE6' if 'East Asian' in ethnicity else '#F8F8F8'};'>"
                f"<span style='font-size:0.78rem; color:#555; font-weight:600;'>"
                f"{study.replace(chr(10), ' ')}{ea_badge}</span><br>"
                f"<span style='font-size:0.74rem; color:#888;'>"
                f"{age_range} · {ethnicity}</span><br>"
                f"<span style='font-weight:700;'>{ref_val:.2f}</span> mm/yr &nbsp;"
                f"<span style='color:{color}; font-weight:600;'>"
                f"{arrow} {abs(delta):.3f}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.caption("🟡 East Asian cohorts (most comparable to this study's Taiwan population)")

    with col3:
        st.markdown("### Risk Gauge")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(5, 2.8))
        zones = [(0.0, 0.10, "#375623", "Slow\n<0.10"),
                 (0.10, 0.30, "#ED7D31", "Moderate\n0.10–0.30"),
                 (0.30, 0.60, "#C00000", "Fast\n≥0.30")]
        for lo, hi, c, lbl in zones:
            ax.barh(0, hi-lo, left=lo, height=0.5, color=c, alpha=0.82)
            ax.text((lo+hi)/2, 0, lbl, ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

        # 指針
        pred_clamped = min(max(y_pred, 0.0), 0.58)
        ax.annotate("", xy=(pred_clamped, 0.28), xytext=(pred_clamped, 0.7),
                    arrowprops=dict(arrowstyle="-|>", color="#222", lw=2.2))
        ax.text(pred_clamped, 0.78, f"{y_pred:.3f}", ha="center",
                fontsize=10, fontweight="bold", color="#222")

        ax.set_xlim(0, 0.60)
        ax.set_ylim(-0.3, 1.1)
        ax.axis("off")
        ax.set_title("Annualized AXL Progression (mm/yr)", fontsize=9, pad=4)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── 東亞常模對比 Panel ────────────────────────────────────
    st.divider()
    st.markdown("### 📏 AXL vs. East Asian Norms")
    st.caption(
        "Compares this patient's axial length to same-age, same-sex East Asian children "
        "(He et al., *Br J Ophthalmol* 2023 · n = 14,127 · Shanghai + Guangzhou)"
    )

    pctile = axl_percentile_approx(float(axl_base), age, sex)
    pct_data = EA_PCTILE.get(age)

    col_n1, col_n2 = st.columns([1, 1.6])

    with col_n1:
        # 百分位指示
        if pctile >= 90:
            pctile_label, pctile_color, pctile_msg = (
                "⚠️ Very Long (≥ P90)",
                "#C00000",
                "AXL is in the top 10% for this age/sex group — significantly longer than peers."
            )
        elif pctile >= 75:
            pctile_label, pctile_color, pctile_msg = (
                "🔴 Long (P75–P90)",
                "#ED7D31",
                "AXL is above average — longer than ~75–90% of same-age peers."
            )
        elif pctile >= 25:
            pctile_label, pctile_color, pctile_msg = (
                "🟢 Average (P25–P75)",
                "#375623",
                "AXL is within the typical range for this age and sex."
            )
        else:
            pctile_label, pctile_color, pctile_msg = (
                "🔵 Short (< P25)",
                "#2E75B6",
                "AXL is shorter than ~75% of same-age peers — low axial elongation risk."
            )

        st.markdown(
            f"<div style='background:{pctile_color}22; border-left:5px solid {pctile_color}; "
            f"padding:14px 18px; border-radius:8px; margin-bottom:10px;'>"
            f"<div style='font-size:0.82rem; color:#555; font-weight:600;'>Estimated Percentile</div>"
            f"<div style='font-size:2.2rem; font-weight:800; color:{pctile_color};'>P{pctile:.0f}</div>"
            f"<div style='font-size:0.9rem; font-weight:600; color:{pctile_color};'>{pctile_label}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(f"> {pctile_msg}")
        st.markdown(
            f"**Patient AXL:** {axl_base:.2f} mm  \n"
            f"**East Asian P50 (age {age}, {sex}):** {norm_axl:.2f} mm  \n"
            f"**Deviation:** {axl_dev:+.2f} mm"
        )
        st.caption("Source: He X et al. *Br J Ophthalmol* 2023;107:167–175")

    with col_n2:
        # 圖：常模帶（P10/P50/P90） + 病人實際量測軌跡
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ── 常模曲線資料（橫跨 7–18 歲）
        ages_plot = list(range(7, 19))
        key_m = "p10_m" if sex == "Male" else "p10_f"
        key_50 = "p50_m" if sex == "Male" else "p50_f"
        key_90 = "p90_m" if sex == "Male" else "p90_f"
        p10_vals = [EA_PCTILE[a][key_m]  for a in ages_plot if a in EA_PCTILE]
        p50_vals = [EA_PCTILE[a][key_50] for a in ages_plot if a in EA_PCTILE]
        p90_vals = [EA_PCTILE[a][key_90] for a in ages_plot if a in EA_PCTILE]

        # ── 把 valid_visits（月份, AXL）轉成（實際年齡, AXL）
        # 規則：Visit 1 = age（歲），每多 1 個月 = +1/12 歲
        pt_ages = [age + mo / 12.0 for mo, _ in valid_visits]
        pt_axls = [axl for _, axl in valid_visits]

        # 若無 visit 就只畫基線單點（fallback）
        if len(pt_ages) == 0:
            pt_ages = [float(age)]
            pt_axls = [float(axl_base)]

        fig_n, ax_n = plt.subplots(figsize=(6, 3.8))

        # 常模帶
        ax_n.fill_between(ages_plot, p10_vals, p90_vals,
                          alpha=0.13, color="#2E75B6")
        ax_n.plot(ages_plot, p50_vals, color="#2E75B6", lw=2.2,
                  linestyle="--", label="East Asian P50 (median)")
        ax_n.plot(ages_plot, p10_vals, color="#2E75B6", lw=1,
                  linestyle=":", alpha=0.55, label="P10 / P90")
        ax_n.plot(ages_plot, p90_vals, color="#2E75B6", lw=1,
                  linestyle=":", alpha=0.55)

        # 病人軌跡
        pt_color = pctile_color
        if len(pt_ages) >= 2:
            ax_n.plot(pt_ages, pt_axls, color=pt_color, lw=2.2,
                      marker="o", markersize=7, zorder=5,
                      label="This patient (measured)")
        else:
            ax_n.scatter(pt_ages, pt_axls, color=pt_color, s=130,
                         zorder=5, label="This patient (baseline)")

        # 標注每個量測點：白底框 + 引線 + 交錯上下，避免字被線蓋住
        for i, (pa, paxl) in enumerate(zip(pt_ages, pt_axls)):
            visit_pctile = axl_percentile_approx(paxl, int(round(pa)), sex)
            label_str = f"V{i+1}: {paxl:.2f} mm  (P{visit_pctile:.0f})"
            va = "bottom" if i % 2 == 0 else "top"
            y_off = 0.28 if va == "bottom" else -0.28
            ax_n.annotate(
                label_str,
                xy=(pa, paxl),
                xytext=(pa, paxl + y_off),
                ha="center", va=va,
                fontsize=8.5, color="#14320f", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=pt_color, lw=1.0, alpha=0.92),
                arrowprops=dict(arrowstyle="-", color=pt_color,
                                lw=0.8, alpha=0.7),
                zorder=6,
            )

        # 如果有 ≥2 次，畫預測延伸線（用 early_slope 延伸到 +1 年）
        if slope_ok and len(pt_ages) >= 2:
            last_age = pt_ages[-1]
            last_axl = pt_axls[-1]
            proj_age = last_age + 1.0
            proj_axl = last_axl + early_slope * 12  # mm/yr
            ax_n.annotate(
                "",
                xy=(proj_age, proj_axl),
                xytext=(last_age, last_axl),
                arrowprops=dict(
                    arrowstyle="-|>", color=pt_color,
                    lw=1.5, linestyle="dashed"
                ),
            )
            ax_n.text(proj_age + 0.08, proj_axl,
                      f"Projected +1yr\n{proj_axl:.2f} mm",
                      fontsize=7.5, color="#14320f", va="center",
                      bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                ec=pt_color, lw=0.8, alpha=0.88),
                      zorder=6)

        ax_n.set_xlabel("Age (years)", fontsize=10)
        ax_n.set_ylabel("Axial Length (mm)", fontsize=10)
        ax_n.set_title(
            f"AXL Growth Trajectory vs. East Asian Norms · {sex}",
            fontsize=9.5, fontweight="bold"
        )
        ax_n.legend(fontsize=8, loc="upper left")
        ax_n.spines[["top", "right"]].set_visible(False)
        ax_n.grid(axis="y", linestyle="--", alpha=0.3)

        # 自動調整 x 軸讓病人軌跡在中央
        x_min = max(6.5, min(pt_ages) - 1.0)
        x_max = min(19.5, max(pt_ages) + 2.0)
        ax_n.set_xlim(x_min, x_max)
        plt.tight_layout()
        st.pyplot(fig_n, use_container_width=True)
        plt.close()

        st.caption(
            "Dashed arrow = projected trajectory based on current slope (+1 yr).  "
            "Blue band = East Asian P10–P90 (He et al. *Br J Ophthalmol* 2023)."
        )

    # ── SHAP Waterfall ────────────────────────────────────────
    st.divider()
    st.markdown("### Feature Contribution (SHAP Waterfall)")

    with st.expander("💡 How to read this chart", expanded=False):
        st.markdown("""
**What is SHAP?**
SHAP (SHapley Additive exPlanations) answers the question: *"Why did the model predict this specific value for this specific patient?"*

The model starts from a **base value** — the average prediction across all patients in the training data.
Then each feature either **pushes the prediction up** (🔴 red, faster progression) or **pushes it down** (🔵 blue, slower progression).

**How to read the bars:**
- The **length** of each bar = how much that feature changed the prediction
- **Red bars**: this feature is making the model predict faster progression than average
- **Blue bars**: this feature is making the model predict slower progression than average
- The feature value (e.g. `= 0.014`) is shown next to the feature name

**Example:** If "Early slope 0–6m = 0.014" has a long blue bar, it means this patient's early AXL slope is relatively slow compared to the training cohort, pulling the prediction *below* the base value.

**Why does this matter clinically?**
Unlike a black-box prediction, SHAP lets you tell the patient's family *why* the model thinks their child is at moderate risk — e.g. "the early measurement trend is the main factor keeping the risk moderate."
        """)

    st.caption("How each input parameter contributed to this prediction")

    try:
        import shap
        imp     = model.named_steps["imp"]
        raw_mdl = model.named_steps["mdl"]
        X_imp   = imp.transform(X_input)

        try:
            booster   = raw_mdl.get_booster()
            explainer = shap.TreeExplainer(booster)
            sv        = explainer.shap_values(X_imp)[0]
            base_val  = explainer.expected_value
        except Exception:
            bg        = imp.transform(
                np.array([[medians.get(f, 0) for f in FEATURES]])
            )
            explainer = shap.KernelExplainer(raw_mdl.predict, bg)
            sv        = explainer.shap_values(X_imp, nsamples=100)[0]
            base_val  = explainer.expected_value

        feat_labels = {
            "age_start": "Age", "sex": "Sex", "laterality": "Eye (OD/OS)",
            "myopia_d": "Baseline myopia (D)", "axl_baseline": "Baseline AXL (mm)",
            "axl_dev_from_norm": "AXL deviation from norm",
            "mi_rx": "MiSight Rx power", "atropine_conc": "Atropine conc.",
            "early_slope_6m": "Early slope 0–6m", "axl_std_6m": "AXL variability",
        }
        labels = [feat_labels.get(f, f) for f in FEATURES]
        vals   = list(input_vals.values())

        # 手動畫 waterfall（避免 shap plot API 版本差異）
        sorted_idx = np.argsort(np.abs(sv))[::-1][:8]
        fig2, ax2  = plt.subplots(figsize=(8, 4.5))
        colors_shap = ["#C00000" if s > 0 else "#2E75B6" for s in sv[sorted_idx]]
        bars = ax2.barh(
            [f"{labels[i]}\n= {vals[i]:.3g}" for i in sorted_idx],
            sv[sorted_idx],
            color=colors_shap, alpha=0.85, edgecolor="white"
        )
        ax2.axvline(0, color="#333", linewidth=0.8)
        for bar, val in zip(bars, sv[sorted_idx]):
            ax2.text(
                val + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height()/2,
                f"{val:+.4f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8.5, color="#222"
            )
        ax2.set_xlabel("SHAP value  (impact on predicted mm/yr)", fontsize=10)
        ax2.set_title(
            f"Base value: {float(base_val):.3f}  →  Prediction: {y_pred:.3f} mm/yr",
            fontsize=10, fontweight="bold"
        )
        ax2.spines[["top","right"]].set_visible(False)
        ax2.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.caption(
            "🔴 Red bars push prediction **higher** (faster progression)  |  "
            "🔵 Blue bars push prediction **lower** (slower progression)"
        )

    except Exception as e:
        st.info(f"SHAP analysis unavailable: {e}")

elif not predict_btn:
    # 未按按鈕時顯示說明
    st.info("👈 **Adjust parameters in the sidebar, then click 'Predict Progression'**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### How to use")
        st.markdown("""
1. Enter patient's **age, eye measurements, and treatment details** in the sidebar
2. Set the **early AXL slope** — the most predictive feature
3. Click **Predict Progression**
4. Review the predicted mm/yr, risk tier, and SHAP breakdown
        """)
    with col_b:
        st.markdown("#### Risk Tiers")
        st.markdown("""
| Tier | mm/yr | Action |
|---|---|---|
| 🟢 Slow | < 0.10 | Annual review |
| 🟡 Moderate | 0.10–0.30 | 3-month follow-up |
| 🔴 Fast | ≥ 0.30 | Escalate therapy |
        """)

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.8rem;'>"
    "⚠️ <b>Research/Demo Only</b> — Not validated for clinical use. "
    "Model trained on retrospective data (n=171 eyes, 2 Taiwanese clinics). "
    "Always consult clinical guidelines and professional judgment.<br>"
    "Built by Mark Lu · Optometrist × CS · "
    "<a href='mailto:marklu0509@gmail.com'>marklu0509@gmail.com</a>"
    "</div>",
    unsafe_allow_html=True
)
