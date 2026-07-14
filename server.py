"""
server.py — FastAPI 後端 + 靜態前端
──────────────────────────────────
- GET  /          → 回傳橄欖綠 HTML 前端（design/tool-v1-medical.html）
- POST /predict   → 呼叫 myopia_core.predict()，回傳真實模型結果
- GET  /healthz   → 健康檢查

本機執行：  python -m uvicorn server:app --host 0.0.0.0 --port 8000
Azure 啟動：同上（把 App Service 的 Startup Command 設成這行）
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

import myopia_core as core

app = FastAPI(title="Myopia Progression Predictor", version="2.0")

_INDEX_PATH = Path(__file__).parent / "design" / "tool-v1-medical.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_PATH.read_text(encoding="utf-8")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_features": len(core.FEATURES), "n_train": int(core.N_TRAIN)}


@app.post("/predict")
async def predict(req: Request):
    try:
        payload = await req.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    try:
        return core.predict(payload)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:  # noqa: BLE001 — 回傳友善錯誤，細節寫入 log
        return JSONResponse({"error": f"prediction failed: {e}"}, status_code=500)
