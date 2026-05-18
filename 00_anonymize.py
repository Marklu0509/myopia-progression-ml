"""
00_anonymize.py
───────────────
在 01_data_prep.py 和 02_features.py 跑完後執行一次。
移除所有可識別欄位，覆寫 master.csv 和 features.csv。

移除項目：姓名、西元出生年
eye_id 改成匿名編號 (eye_001, eye_002, ...)
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

# ── master.csv ────────────────────────────────────────────
m = pd.read_csv(BASE / "master.csv")

# 建立 eye_id → 匿名編號 對照表
mapping = {eid: f"eye_{i+1:03d}" for i, eid in enumerate(sorted(m["eye_id"].unique()))}
m["eye_id"] = m["eye_id"].map(mapping)

drop_cols = [c for c in ["姓名", "西元"] if c in m.columns]
m.drop(columns=drop_cols, inplace=True)

m.to_csv(BASE / "master.csv", index=False, encoding="utf-8-sig")
print(f"✓ master.csv 已去識別化（移除：{drop_cols}，eye_id 匿名化）")

# ── features.csv ──────────────────────────────────────────
f = pd.read_csv(BASE / "features.csv")

f["eye_id"] = f["eye_id"].map(mapping)

drop_cols_f = [c for c in ["姓名", "西元"] if c in f.columns]
f.drop(columns=drop_cols_f, inplace=True)

f.to_csv(BASE / "features.csv", index=False, encoding="utf-8-sig")
print(f"✓ features.csv 已去識別化")
