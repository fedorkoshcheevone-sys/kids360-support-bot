"""
FastAPI-приложение Penny 2.0 — бот поддержки Kids360 (ветка возврата средств).

Архитектура (итерация 2): диалог — детерминированная форма-мастер на кнопках (логика на фронте).
LLM вызывается ровно в ДВУХ точках:
  POST /api/recognize — vision-распознавание чека;
  POST /api/chat      — разбор СВОБОДНОГО ТЕКСТА в поля формы (когда пользователь пишет вместо кнопки).
Кнопочные шаги формы НЕ обращаются к OpenAI — это прямая экономия токенов.

Роуты:
  GET  /               — отдаёт виджет static/index.html
  GET  /api/config     — приветствие формы-мастера (текст живёт в prompts.py)
  POST /api/chat       — {message, state} -> {reason, channel, amount, date, wants_human, reply}
  POST /api/recognize  — распознавание скриншота чека (multipart-файл ИЛИ base64 в JSON)
  POST /api/verify     — сверка двойного списания с (демо-)биллингом -> вердикт оператору (БЕЗ LLM)

Ключ OpenAI живёт только на сервере в .env. Фронт обращается к своему бэку, а бэк — к OpenAI.
"""

import traceback
from pathlib import Path

from dotenv import load_dotenv

# .env грузим ДО импорта модуля llm, чтобы ключ/модель были в окружении.
load_dotenv()

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app import billing, llm  # noqa: E402
from app.prompts import GREETING  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Kids360 Support Bot — Penny 2.0")

# Пустой ответ разбора текста (когда ничего не извлекли / ошибка) — чтобы форма не рвалась.
_EMPTY_INTERPRET = {
    "reason": None, "channel": None, "amount": None, "date": None, "wants_human": False,
}


@app.get("/")
def index():
    """Главная страница — виджет Penny."""
    path = STATIC_DIR / "index.html"
    if not path.is_file():
        # Явное сообщение при мисконфигурации деплоя вместо сырого 500.
        return JSONResponse(
            status_code=500,
            content={"error": "Виджет не найден: static/index.html отсутствует. Проверьте пути/деплой."},
        )
    return FileResponse(path)


@app.get("/api/config")
def config():
    """Конфиг для фронта: приветственное сообщение формы-мастера."""
    return {"greeting": GREETING}


@app.post("/api/chat")
async def chat(request: Request):
    """
    Разбор СВОБОДНОГО ТЕКСТА в поля формы возврата (вторая LLM-точка).
    Кнопочные шаги формы обрабатываются на фронте и НЕ вызывают LLM.
    Тело: {"message": "...", "state": {...}} -> {reason, channel, amount, date, wants_human, reply}.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={**_EMPTY_INTERPRET, "reply": "Не удалось прочитать запрос."})

    if not isinstance(body, dict):
        body = {}

    message = str(body.get("message", "")).strip()
    state = body.get("state") if isinstance(body.get("state"), dict) else {}

    if not message:
        return {**_EMPTY_INTERPRET, "reply": "Напишите, что случилось, или выберите вариант кнопкой."}

    try:
        return llm.interpret_free_text(message, state)
    except RuntimeError as exc:
        # Нет ключа / конфиг — явная адресная ошибка.
        return JSONResponse(status_code=500, content={**_EMPTY_INTERPRET, "reply": f"⚙️ {exc}"})
    except Exception:
        # Сбой OpenAI/сети — не роняем форму: пишем причину в лог, просим выбрать кнопкой.
        traceback.print_exc()
        return JSONResponse(
            status_code=200,
            content={**_EMPTY_INTERPRET, "reply": "Технические неполадки с разбором текста — выберите вариант кнопкой ниже."},
        )


@app.post("/api/recognize")
async def recognize(request: Request):
    """
    Распознавание чека (vision). Принимает:
      - multipart/form-data с полем file (скриншот), ИЛИ
      - application/json с полем image / image_b64 (base64 строка).
    Возвращает {amount, date, purpose, bank_or_channel, raw_text, confidence}.
    """
    content_type = request.headers.get("content-type", "")

    try:
        if "application/json" in content_type:
            body = await request.json()
            if not isinstance(body, dict):
                body = {}
            image_b64 = body.get("image") or body.get("image_b64")
            if not image_b64:
                return JSONResponse(
                    status_code=400,
                    content={"error": "В JSON нет поля image с base64 картинкой.", "confidence": 0},
                )
            # data:-URL: вычленяем mime и base64, чтобы не объявлять PNG/WebP как JPEG.
            mime = "image/jpeg"
            if image_b64.strip().startswith("data:") and "," in image_b64:
                header, image_b64 = image_b64.split(",", 1)
                if ":" in header and ";" in header:
                    mime = header.split(":", 1)[1].split(";", 1)[0] or "image/jpeg"
            result = llm.recognize_receipt(image_b64=image_b64, mime=mime)
        else:
            form = await request.form()
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Не найден файл в поле file.", "confidence": 0},
                )
            image_bytes = await upload.read()
            mime = getattr(upload, "content_type", None) or "image/jpeg"
            result = llm.recognize_receipt(image_bytes=image_bytes, mime=mime)
    except RuntimeError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc), "confidence": 0})
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=200,
            content={
                "error": "Не удалось распознать чек из-за технической ошибки. "
                         "Можно описать данные текстом, я передам специалисту.",
                "confidence": 0,
            },
        )

    return result


@app.post("/api/verify")
async def verify(request: Request):
    """
    Сверка двойного списания с биллингом (ДЕМО-мок). БЕЗ LLM — детерминированные правила.
    Тело: {account_id?, amount, date, channel, reason} -> вердикт для оператора на апрув.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    return billing.check_double_charge(
        account_id=body.get("account_id"),
        amount=body.get("amount"),
        date=body.get("date"),
        channel=body.get("channel"),
        reason=body.get("reason"),
    )


# Статику монтируем ПОСЛЕ роутов, чтобы /api/* не перехватывались.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
