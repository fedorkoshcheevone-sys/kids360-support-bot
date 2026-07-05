"""
ДЕМО-биллинг Kids360 (мок).

Реального биллинга / стор-API у прототипа нет. Этот модуль ИМИТИРУЕТ ответ
биллинг-сервиса, чтобы показать сквозной путь ветки возврата:
    чек (vision) → сверка с биллингом → готовый ВЕРДИКТ → апрув оператора.

В боевой версии здесь был бы запрос к биллингу Kids360 по account_id
(и/или к App Store Server API / Google Play Developer API по каналу).
Вердикт считается ДЕТЕРМИНИРОВАННЫМИ правилами по структурированным данным — БЕЗ LLM.
Это третья причина, почему платёжная логика надёжна: деньги не отдаём на откуп генерации.
"""


def check_double_charge(account_id=None, amount=None, date=None, channel=None, reason=None):
    """
    Сверка заявленного двойного списания с биллингом. Возвращает готовый вердикт
    для оператора на апрув. Реального обращения к биллингу нет — это демо-мок.
    """
    charges = _lookup_charges(account_id, amount, date, reason)
    count = len(charges)
    amount_matches = bool(amount) and count >= 2

    if count >= 2:
        verdict, recommendation = "double_charge_confirmed", "refund"
    elif count == 1:
        verdict, recommendation = "single_charge_only", "needs_review"
    else:
        verdict, recommendation = "no_charge_found", "needs_review"

    return {
        "demo": True,
        "account_id": account_id or "demo-account",
        "channel": channel,
        "charges_found": count,
        "amount_matches": amount_matches,
        "charges": charges,
        "verdict": verdict,
        "recommendation": recommendation,
    }


def _lookup_charges(account_id, amount, date, reason):
    """
    Имитация выборки списаний из биллинга по аккаунту.
    Демо-правило: если чек распознан (есть сумма) и причина — двойное списание,
    «биллинг» возвращает два совпадающих списания; иначе одно; без суммы — ноль.
    """
    if not amount:
        return []
    charge = {"amount": amount, "date": date or "—", "product": "Kids360 Подписка", "status": "captured"}
    if reason == "double":
        return [dict(charge), dict(charge)]
    return [dict(charge)]
