"""
Обёртка над OpenAI (новый SDK). ДВЕ (и только две) LLM-точки прототипа:
- recognize_receipt(...)   — vision-распознавание скриншота чека -> строгий JSON.
- interpret_free_text(...) — разбор свободного текста пользователя в поля формы возврата.

Всё остальное (кнопочные шаги формы-мастера) детерминировано на фронте и НЕ вызывает LLM.
Ключ и модель берутся из окружения (.env). Ключ НИКОГДА не хардкодится и не уходит на фронт.
"""

import base64
import json
import os
import time

from openai import OpenAI

from app.knowledge_base import KNOWLEDGE_BASE
from app.prompts import INTERPRET_PROMPT

# Ключ подхватывается из окружения (.env загружается в main.py до импорта роутов).
# Если ключа нет — не роняем импорт, а отдаём понятную ошибку при первом вызове.
MODEL = os.getenv("MODEL", "gpt-4o")

# Ленивая инициализация клиента: чтобы приложение поднималось даже без ключа
# (например, при первом запуске без .env), а ошибка была явной и адресной.
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-your-key-here":
            raise RuntimeError(
                "OPENAI_API_KEY не задан. Создайте .env из .env.example и впишите реальный ключ."
            )
        # Опциональный base_url — для OpenAI-совместимых роутеров (напр. Polza.ai).
        # Если не задан, клиент ходит в обычный OpenAI.
        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("POLZA_BASE_URL")
            or os.getenv("LLM_BASE_URL")
        )
        _client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    return _client


# Ошибки, при которых имеет смысл повторить запрос (сеть/лимиты/временные сбои OpenAI).
def _is_retryable(exc: Exception) -> bool:
    name = exc.__class__.__name__
    return name in {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "APIError",
    }


def _with_retries(fn, attempts: int = 3, base_delay: float = 1.0):
    """Выполнить fn с экспоненциальными паузами при повторяемых ошибках (1s, 2s, ...)."""
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — сознательно широкий, решение о повторе ниже
            last_exc = exc
            if _is_retryable(exc) and i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
                continue
            raise
    # сюда не доходим, но на всякий случай
    raise last_exc


def interpret_free_text(message: str, state: dict = None) -> dict:
    """
    Разбор свободного текста пользователя в поля формы возврата — вторая (и последняя)
    LLM-точка помимо vision. Вызывается ТОЛЬКО когда пользователь пишет текст вместо кнопки.
    LLM здесь «переводчик»: извлекает поля, а не ведёт параллельный чат.
    Возвращает {reason, channel, amount, date, wants_human, reply}.
    """
    client = _get_client()

    system = INTERPRET_PROMPT + "\n\n" + KNOWLEDGE_BASE
    if state:
        known = {k: state.get(k) for k in ("reason", "channel", "amount", "date")}
        system += "\n\nУже известно из формы (не переспрашивай): " + json.dumps(known, ensure_ascii=False)

    def call(json_mode: bool):
        kwargs = {
            "model": MODEL,
            "temperature": 0.2,  # низкая: извлечение, не творчество
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": str(message)},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    try:
        raw = _with_retries(lambda: call(True))
    except Exception as exc:  # noqa: BLE001
        # роутер без поддержки response_format=json_object — повтор без него
        if exc.__class__.__name__ in ("BadRequestError", "UnprocessableEntityError", "NotFoundError"):
            raw = _with_retries(lambda: call(False))
        else:
            raise

    cleaned = _strip_json_fences(raw)
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("ожидался JSON-объект")
    except Exception:
        return {
            "reason": None, "channel": None, "amount": None, "date": None,
            "wants_human": False, "reply": "Не совсем понял — выберите вариант кнопкой ниже.",
        }

    def _enum(v, allowed):
        return v if v in allowed else None

    reply = data.get("reply")
    return {
        "reason": _enum(data.get("reason"), {"double", "after_cancel", "unknown_charge", "unused_time"}),
        "channel": _enum(data.get("channel"), {"appstore", "googleplay", "operator"}),
        "amount": data.get("amount") or None,
        "date": data.get("date") or None,
        "wants_human": bool(data.get("wants_human")),
        "reply": str(reply).strip() if reply else "Понял, продолжаем оформление.",
    }


def _strip_json_fences(text: str) -> str:
    """Снять обёртку ```json ... ``` или ``` ... ```, если модель её добавила."""
    t = text.strip()
    if t.startswith("```"):
        # убрать первую строку с ``` (и возможным 'json')
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


RECOGNIZE_PROMPT = (
    "Ты извлекаешь данные из скриншота банковского чека/справки об операции. "
    "Верни СТРОГО JSON с полями: "
    "amount (число или строка суммы), "
    "date (дата операции), "
    "purpose (назначение платежа/описание), "
    "bank_or_channel (банк или платёжный канал), "
    "raw_text (весь распознанный текст), "
    "confidence (0..1 — насколько уверен). "
    "Если это не чек — confidence=0 и пустые поля. "
    "Не добавляй ничего кроме JSON."
)


def recognize_receipt(image_bytes: bytes = None, image_b64: str = None,
                      mime: str = "image/jpeg") -> dict:
    """
    Vision-распознавание чека через gpt-4o.
    Принимает либо сырые байты картинки, либо готовый base64.
    Возвращает dict: {amount, date, purpose, bank_or_channel, raw_text, confidence}.
    При невалидном JSON — {confidence: 0, raw_text: <сырой ответ>}.
    """
    client = _get_client()

    if image_b64 is None:
        if image_bytes is None:
            raise ValueError("Нужны либо image_bytes, либо image_b64.")
        image_b64 = base64.b64encode(image_bytes).decode("ascii")

    data_url = f"data:{mime};base64,{image_b64}"

    vision_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": RECOGNIZE_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    def call(json_mode: bool):
        kwargs = {"model": MODEL, "temperature": 0, "messages": vision_messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    try:
        raw = _with_retries(lambda: call(True))
    except Exception as exc:  # noqa: BLE001
        # Некоторые OpenAI-совместимые роутеры не поддерживают response_format=json_object —
        # повторяем без него (безопасный парсинг ниже всё равно снимет ```-обёртки).
        if exc.__class__.__name__ in ("BadRequestError", "UnprocessableEntityError", "NotFoundError"):
            raw = _with_retries(lambda: call(False))
        else:
            raise

    # Безопасный парсинг: снять возможные ```-обёртки, затем json.loads.
    cleaned = _strip_json_fences(raw)
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("ожидался JSON-объект")
    except Exception:
        return {
            "amount": "",
            "date": "",
            "purpose": "",
            "bank_or_channel": "",
            "raw_text": raw,
            "confidence": 0,
        }

    # Нормализуем набор полей (модель могла что-то пропустить).
    return {
        "amount": data.get("amount", ""),
        "date": data.get("date", ""),
        "purpose": data.get("purpose", ""),
        "bank_or_channel": data.get("bank_or_channel", ""),
        "raw_text": data.get("raw_text", ""),
        "confidence": data.get("confidence", 0),
    }
