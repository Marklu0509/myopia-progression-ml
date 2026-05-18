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
    st.image("https://img.icons8.com/color/96/eye.png", width=60)
    st.title("Patient Parameters")
    st.caption("Adjust sliders to match patient profile")

    st.markdown("#### 📋 Baseline Demographics")
    age   = st.slider("Age at treatment start (years)", 6, 20, 11)
    sex   = st.radio("Sex", ["Female", "Male"], horizontal=True)
    lat   = st.radio("Eye", ["OD (Right)", "OS (Left)"], horizontal=True)

    st.markdown("#### 🔬 Baseline Ocular")
    axl_base = st.slider("Baseline axial length (mm)", 21.0, 28.0,
                         float(round(medians.get("axl_baseline", 24.0), 1)),
                         step=0.05, format="%.2f")
    myopia_d = st.slider("Baseline myopia (D)", -12.0, -0.25,
                         float(round(medians.get("myopia_d", -3.0), 2)),
                         step=0.25, format="%.2f")

    st.markdown("#### 💊 Treatment")
    mi_rx = st.slider("MiSight Rx power (D)", -6.0, -0.5, -2.0, step=0.25, format="%.2f")
    atro  = st.selectbox("Adjunct atropine", [
        "None (0%)", "0.01%", "0.02%", "0.05%", "0.125%"
    ])

    st.markdown("#### 📈 Early AXL Measurements (0–6 months)")
    st.caption("Enter follow-up visits to auto-calculate slope")

    # 輸入方式選擇
    axl_input_mode = st.radio(
        "Input mode", ["Enter dates + AXL", "Enter months + AXL directly"],
        horizontal=True
    )

    # 最多輸入 4 個時間點
    NUM_PTS = 4
    months_list = []
    axl_list    = []

    if axl_input_mode == "Enter dates + AXL":
        start_date = st.date_input(
            "Treatment start date",
            value=date.today() - timedelta(days=180),
            help="The date the patient started MiSight treatment"
        )
        st.markdown("**Follow-up visits:**")
        for i in range(NUM_PTS):
            c1, c2 = st.columns(2)
            with c1:
                visit_date = st.date_input(
                    f"Visit {i+1} date",
                    value=None,
                    key=f"vdate_{i}",
                    label_visibility="collapsed" if i > 0 else "visible",
                )
            with c2:
                axl_val = st.number_input(
                    f"AXL {i+1} (mm)",
                    min_value=20.0, max_value=30.0,
                    value=None,
                    step=0.01, format="%.2f",
                    key=f"vaxl_{i}",
                    label_visibility="collapsed" if i > 0 else "visible",
                    placeholder="mm"
                )
            if visit_date is not None and axl_val is not None:
                months = (visit_date - start_date).days / 30.44
                if 0 < months <= 8:
                    months_list.append(months)
                    axl_list.append(axl_val)

    else:  # Enter months directly
        st.markdown("**Months since start + AXL value:**")
        for i in range(NUM_PTS):
            c1, c2 = st.columns(2)
            with c1:
                mo = st.number_input(
                    f"Month {i+1}",
                    min_value=0.5, max_value=8.0,
                    value=None, step=0.5, format="%.1f",
                    key=f"vmo_{i}",
                    placeholder="e.g. 3"
                )
            with c2:
                axl_val = st.number_input(
                    f"AXL {i+1} (mm)",
                    min_value=20.0, max_value=30.0,
                    value=None, step=0.01, format="%.2f",
                    key=f"vaxl2_{i}",
                    placeholder="mm"
                )
            if mo is not None and axl_val is not None:
                months_list.append(float(mo))
                axl_list.append(float(axl_val))

    # 自動計算斜率和 SD
    # 把 baseline（month 0）自動加入作為第一個點
    all_months = [0.0] + months_list
    all_axls   = [axl_base] + axl_list

    early_slope = float(medians.get("early_slope_6m", 0.015))
    axl_std     = float(medians.get("axl_std_6m", 0.05))
    slope_ok    = False

    n_followup = len(months_list)   # 不含 baseline 的後續點數

    if n_followup >= 1:             # baseline + 1 個後續點 = 可以算斜率
        m_arr = np.array(all_months)
        a_arr = np.array(all_axls)
        coeffs      = np.polyfit(m_arr, a_arr, 1)
        early_slope = float(coeffs[0])
        axl_std     = float(a_arr.std())
        slope_ok    = True
        st.success(
            f"✓ Baseline + {n_followup} follow-up visit(s)  |  "
            f"Slope = **{early_slope:+.4f} mm/month**  |  "
            f"SD = {axl_std:.4f} mm"
        )
        # 把算出的 mm/yr 給使用者看
        st.caption(f"≈ {early_slope * 12:+.3f} mm/yr annualized slope from entered data")
    else:
        st.info("Enter at least 1 follow-up visit — baseline (month 0) is used automatically")

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

# 年齡-眼軸常模
AGE_NORM = {6:22.5,7:22.8,8:23.0,9:23.2,10:23.4,11:23.6,12:23.8,
            13:24.0,14:24.1,15:24.2,16:24.3,17:24.4,18:24.5,19:24.5,20:24.5}
norm_axl = AGE_NORM.get(age, 24.0)
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
