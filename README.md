# Myopia Progression Prediction — ML Pipeline

> **Predicting Long-Term Axial Length Progression from Short-Term Trajectories in Pediatric Myopia Management**

🚀 **[Live Demo → Streamlit App](https://marklu-myopia.streamlit.app/)

A machine learning pipeline that uses early axial length (AXL) measurements (0–6 months) to predict annualized myopia progression (mm/yr) in children undergoing soft contact lens therapy (MiSight ± low-dose atropine). Built to support clinical decision-making and demonstrate cross-domain expertise in optometry and machine learning.

---

## Key Results

| Model | LOOCV RMSE (mm/yr) | R² | Bootstrap 95% CI |
|---|---|---|---|
| **XGBoost** | **0.090** | **0.892** | [0.072, 0.106] |
| Random Forest | 0.108 | 0.845 | [0.084, 0.128] |
| Lasso | 0.109 | 0.841 | [0.089, 0.130] |
| Ridge | 0.111 | 0.834 | [0.090, 0.132] |
| Linear Regression | 0.111 | 0.834 | [0.090, 0.132] |
| SVR | 0.131 | 0.771 | [0.105, 0.154] |

Real-world cohort median progression: **~0.17 mm/yr** — consistent with published MiSight RCT outcomes (Chamberlain et al. 2019: 0.13–0.15 mm/yr).

---

## Clinical Context

Myopia (short-sightedness) affects ~30% of the global population and is projected to reach 50% by 2050. Early prediction of progression rate allows clinicians to:
- Identify **fast progressors** who need more aggressive therapy (e.g., higher atropine concentration)
- Avoid over-treating **slow progressors** who respond well to MiSight alone
- Provide objective, data-driven counseling to families

---

## Data

- **Sources**: Two Taiwanese optometry clinics (n = 265 eyes; 171 with ≥2 measurement timepoints)
- **Treatment**: MiSight 1-day soft contact lens ± low-dose atropine (0.01%–0.125%)
- **Primary outcome**: Annualized axial length progression (mm/yr) via OLS slope across all follow-up timepoints
- **Age range**: 7–20 years (median ~13 years)

> ⚠️ Patient data is de-identified and not included in this repository. Contact the author for collaboration inquiries.

---

## Pipeline Overview

```
Raw Excel (2 clinics)
        │
        ▼
01_data_prep.py     ← Clean, merge, melt to long format → master.csv
        │
        ▼
02_features.py      ← Compute per-eye features + target y → features.csv
        │
        ▼
03_models.py        ← 6-model LOOCV benchmark → benchmark_table.xlsx + figure
        │
        ▼
04_interpret.py     ← SHAP, calibration plot, RCT comparison → figures/
        │
        ▼
05_visualize.py     ← Age analysis, atropine dose-response, clinic comparison
        │
        ▼
app.py              ← Streamlit interactive demo
```

---

## Features (Layer 1)

| Feature | Clinical Rationale |
|---|---|
| `early_slope_6m` | AXL velocity in first 6 months — strongest early signal |
| `axl_baseline` | Higher baseline AXL → typically faster progression |
| `axl_dev_from_norm` | Deviation from age-matched population norm (Tideman 2016) |
| `age_start` | Younger children (esp. 11–12 yrs) progress faster |
| `myopia_d` | Baseline refractive error (diopters) |
| `atropine_conc` | 0 / 0.01 / 0.05 / 0.125% — dose-response effect |
| `mi_rx` | MiSight prescription power |
| `axl_std_6m` | AXL variability — captures measurement noise vs real change |

---

## Figures

| File | Description |
|---|---|
| `figures/benchmark_bar.png` | LOOCV RMSE across 6 models with 95% CI |
| `figures/shap_summary.png` | SHAP beeswarm — which features drive predictions most |
| `figures/calibration_plot.png` | Predicted vs observed AXL progression |
| `figures/rct_benchmark_v2.png` | Real-world cohort vs RCT benchmarks (with age context) |
| `figures/age_boxplot.png` | Progression by age group — 11–12 yrs fastest |
| `figures/age_scatter.png` | Continuous age vs progression, colored by atropine dose |
| `figures/atropine_effect.png` | Dose-response analysis + age confound check |
| `figures/shap_age_interaction.png` | SHAP dependency: early slope × age interaction |
| `figures/source_comparison.png` | Two-clinic data consistency (Mann-Whitney U) |

---

## Repository Structure

```
myopia_ml/
├── app.py                # Streamlit demo app
├── save_model.py         # Train & serialize final model
├── 00_anonymize.py       # De-identification (run after 01 & 02)
├── 01_data_prep.py       # Data cleaning & merging
├── 02_features.py        # Feature engineering
├── 03_models.py          # Model benchmark (LOOCV)
├── 04_interpret.py       # SHAP + clinical visualization
├── 05_visualize.py       # Age & dose-response analysis
├── model.joblib          # Serialized XGBoost pipeline
├── requirements.txt
├── .gitignore            # Excludes all patient data
├── results/
│   └── benchmark_table.xlsx
└── figures/              # All output figures (9 total)
```

---

## Reproducing Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place raw data files:
#    Data Base/datebase.xlsx
#    Data Base/合安/合安病例紀錄.xlsx

# 3. Run pipeline in order
python 01_data_prep.py
python 02_features.py
python 00_anonymize.py   # de-identify before any sharing
python 03_models.py
python 04_interpret.py
python 05_visualize.py
python save_model.py

# 4. Launch demo app
streamlit run app.py
```

---

## Limitations

This project is a **retrospective exploratory analysis**, not a clinical trial. The following limitations should be considered before drawing clinical conclusions:

**Sample size**: n = 171 eyes from 2 clinics. While sufficient for proof-of-concept, this is underpowered for subgroup analyses (e.g., atropine dose-stratified modeling).

**No control group**: All patients received active treatment (MiSight ± atropine). Counterfactual progression rates are estimated from published RCT benchmarks, not observed controls.

**Retrospective design**: Target variable (annualized progression) is derived from the same follow-up data used to define the cohort. Prospective validation on unseen patients is required before clinical deployment.

**Single region / single ethnicity**: Both clinics are in Taiwan. Progression rates and treatment responses may differ in other populations.

**Measurement heterogeneity**: Follow-up intervals were not standardized across patients (opportunistic measurement timing), introducing variability in the early slope estimate.

**Atropine confounding**: Patients receiving higher atropine concentrations tended to be older (see `figures/atropine_effect.png`), making dose-response interpretation difficult without propensity score adjustment.

---

## Future Work

- **Prospective validation**: Enroll a new cohort prospectively, collect standardized 3-month and 6-month AXL measurements, and validate model predictions against 12-month outcomes.
- **Larger multi-clinic dataset**: Partner with additional optometry clinics to increase n and enable subgroup modeling (age-stratified, dose-stratified).
- **Treatment effect estimation**: With a proper control group, apply causal inference methods (e.g., propensity score matching, doubly-robust estimation) to estimate individualized treatment effects.
- **Longitudinal modeling**: Replace cross-sectional feature engineering with LSTM or mixed-effects models that natively handle irregular time series.
- **Clinical integration**: Package as a FHIR-compatible API for integration with electronic health record (EHR) systems.

---

## References

- Chamberlain P, et al. *A 3-year Randomized Clinical Trial of MiSight Lenses for Myopia Control.* Optom Vis Sci. 2019.
- Yam JC, et al. *Low-Concentration Atropine for Myopia Progression (LAMP).* Ophthalmology. 2019.
- Chia A, et al. *Atropine for the Treatment of Childhood Myopia (ATOM2).* Ophthalmology. 2012.
- Tideman JWL, et al. *Association of Axial Length With Risk of Uncorrectable Visual Impairment.* JAMA Ophthalmol. 2016.
- IMI – Myopia Management Guidelines. *Investigative Ophthalmology & Visual Science.* 2021.

---

## Author

Mark Lu | Optometrist × CS | [marklu0509@gmail.com](mailto:marklu0509@gmail.com)
