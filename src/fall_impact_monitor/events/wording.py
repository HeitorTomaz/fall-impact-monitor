"""User-facing notification texts (English default, Brazilian Portuguese available).

Honest wording (CLAUDE.md §2): events are "possible"; the system never says "confirmed"
and never makes a claim about injury. Scores are shown as rule-based and uncalibrated.
Logs and JSON stay in English; only these texts are localized.
"""

from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from fall_impact_monitor.events.model import EventCategory
from fall_impact_monitor.events.record import EventRecord

Language = Literal["en", "pt-BR"]

_TEXTS: dict[str, dict[Language, str]] = {
    "possible_fall": {"en": "Possible fall", "pt-BR": "Possível queda"},
    "person_down_after_fall": {
        "en": "Person on the floor for {s} s",
        "pt-BR": "Pessoa no chão há {s} s",
    },
    "person_down_no_fall": {
        "en": "Person lying on the floor for {s} s (fall not observed)",
        "pt-BR": "Pessoa deitada no chão há {s} s (queda não observada)",
    },
    "possible_head_impact": {
        "en": "Possible head impact (experimental)",
        "pt-BR": "Possível impacto na cabeça (experimental)",
    },
    "manual_report": {"en": "Missed event reported", "pt-BR": "Evento perdido relatado"},
    "monitoring_degraded": {"en": "Monitoring degraded", "pt-BR": "Monitoramento degradado"},
    "monitoring_restored": {"en": "Monitoring restored", "pt-BR": "Monitoramento restabelecido"},
    "camera": {"en": "Camera", "pt-BR": "Câmera"},
    "time": {"en": "Time", "pt-BR": "Horário"},
    "confidence": {
        "en": "Confidence: {p}% (rule-based, uncalibrated)",
        "pt-BR": "Confiança: {p}% (por regras, não calibrada)",
    },
    "evidence": {"en": "Evidence", "pt-BR": "Evidências"},
    "reason": {"en": "Reason", "pt-BR": "Motivo"},
    "reason_camera_lost": {"en": "no video from the camera", "pt-BR": "sem vídeo da câmera"},
    "reason_camera_unstable": {
        "en": "camera keeps disconnecting",
        "pt-BR": "câmera desconectando repetidamente",
    },
    "reason_processing_failing": {
        "en": "detection failing on every frame",
        "pt-BR": "detecção falhando em todos os quadros",
    },
    "reason_analysis_slow": {
        "en": "analysis below the minimum rate",
        "pt-BR": "análise abaixo da taxa mínima",
    },
    "reason_disk_low": {
        "en": "disk below the free-space floor",
        "pt-BR": "disco abaixo do espaço livre mínimo",
    },
    "reason_alerts_backlog": {
        "en": "alerts are not being delivered",
        "pt-BR": "alertas não estão sendo entregues",
    },
    "reason_database_backlog": {
        "en": "database not reachable (events kept on disk)",
        "pt-BR": "banco de dados inacessível (eventos guardados em disco)",
    },
    "reason_worker_stalled": {
        "en": "processing stalled, restarting",
        "pt-BR": "processamento travado, reiniciando",
    },
    "late": {
        "en": "Late: sent {n} s after the event",
        "pt-BR": "Atrasado: enviado {n} s após o evento",
    },
    "repeats": {
        "en": "Repeated {n} times, last at {t}",
        "pt-BR": "Repetido {n} vezes, a última às {t}",
    },
    "footer": {
        "en": "Assistive alert: it does not replace supervision.",
        "pt-BR": "Alerta assistivo. Não substitui a supervisão.",
    },
    "drop": {"en": "hip drop {v} body heights", "pt-BR": "queda do quadril {v} alturas"},
    "speed": {"en": "speed {v} heights/s", "pt-BR": "velocidade {v} alturas/s"},
    "lying": {"en": "lying {v} s", "pt-BR": "deitada {v} s"},
    "recovered": {"en": "got up afterwards", "pt-BR": "levantou depois"},
    "track_lost": {"en": "person lost from view", "pt-BR": "pessoa saiu de vista"},
    "left_zone": {"en": "left the {v} lying", "pt-BR": "saiu deitada de {v}"},
    "test": {
        "en": "Test message from Fall Impact Monitor: delivery works.",
        "pt-BR": "Mensagem de teste do Fall Impact Monitor: a entrega funciona.",
    },
    "video_not_ready": {
        "en": "The video is not ready yet; try again in a minute.",
        "pt-BR": "O vídeo ainda não está pronto; tente de novo em um minuto.",
    },
    "unknown_event": {
        "en": "This event is no longer known.",
        "pt-BR": "Este evento não é mais conhecido.",
    },
    "feedback_saved": {"en": "Recorded, thank you.", "pt-BR": "Registrado, obrigado."},
    "video_failed": {
        "en": "The video could not be sent; it stays on the server.",
        "pt-BR": "Não foi possível enviar o vídeo; ele continua no servidor.",
    },
    "requested": {"en": "Requested", "pt-BR": "Pedido"},
    "recorded": {"en": "Recorded", "pt-BR": "Gravado"},
    "gaps": {"en": "{n} gap(s) in the recording", "pt-BR": "{n} lacuna(s) na gravação"},
    "truncated": {
        "en": "Shortened to fit the size limit",
        "pt-BR": "Encurtado para caber no limite de tamanho",
    },
}

_BUTTONS: dict[Language, dict[str, str]] = {
    "en": {
        "seen": "Seen",
        "false_alarm": "False alarm",
        "confirmed": "Confirmed",
        "video": "Watch video",
    },
    "pt-BR": {
        "seen": "Vi",
        "false_alarm": "Falso alarme",
        "confirmed": "Confirmado",
        "video": "Ver vídeo",
    },
}


def message(language: Language, key: str, **values: Any) -> str:
    return _TEXTS[key][language].format(**values)


def button_labels(language: Language) -> dict[str, str]:
    """Human feedback buttons (the person confirms, never the system)."""
    return dict(_BUTTONS[language])


def delivery_test_text(language: Language) -> str:
    return message(language, "test")


def _local(value: datetime, timezone: str) -> str:
    return value.astimezone(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S")


def _title(record: EventRecord, language: Language) -> str:
    category = record.event.category
    evidence = record.event.evidence
    if category is EventCategory.PERSON_DOWN:
        seconds = round(float(evidence.get("lying_s", 0)))
        key = "person_down_after_fall" if evidence.get("fall_observed") else "person_down_no_fall"
        return message(language, key, s=seconds)
    return message(language, category.value)


def _evidence(record: EventRecord, language: Language) -> list[str]:
    evidence = record.event.evidence
    parts = []
    if "max_drop_heights" in evidence:
        parts.append(message(language, "drop", v=f"{evidence['max_drop_heights']:.2f}"))
    if evidence.get("max_drop_speed_hps"):
        parts.append(message(language, "speed", v=f"{evidence['max_drop_speed_hps']:.1f}"))
    if evidence.get("lying_s") and record.event.category is not EventCategory.PERSON_DOWN:
        parts.append(message(language, "lying", v=f"{evidence['lying_s']:.0f}"))
    if evidence.get("recovered"):
        parts.append(message(language, "recovered"))
    if evidence.get("track_lost"):
        parts.append(message(language, "track_lost"))
    if evidence.get("left_safe_zone"):
        parts.append(message(language, "left_zone", v=evidence["left_safe_zone"]))
    return parts


def alert_text(
    record: EventRecord,
    *,
    language: Language,
    timezone: str,
    now_utc: datetime,
    late_after_s: float,
    repeats: int = 0,
    last_utc: datetime | None = None,
) -> str:
    category = record.event.category
    monitoring = category in (EventCategory.MONITORING_DEGRADED, EventCategory.MONITORING_RESTORED)
    icon = {"monitoring_degraded": "🔴", "monitoring_restored": "🟢"}.get(category.value, "⚠️")
    lines = [
        f"{icon} {_title(record, language)}",
        f"{message(language, 'camera')}: {record.event.camera_id}",
        f"{message(language, 'time')}: {_local(record.started_utc, timezone)} ({timezone})",
    ]
    if monitoring:
        reasons = [
            message(language, f"reason_{key}") if f"reason_{key}" in _TEXTS else str(key)
            for key in record.event.evidence.get("reasons", [])
        ]
        if record.event.evidence.get("reason"):
            reasons.append(str(record.event.evidence["reason"]))
        if reasons:
            lines.append(f"{message(language, 'reason')}: {'; '.join(reasons)}")
    else:
        lines.append(message(language, "confidence", p=round(record.event.confidence * 100)))
        parts = _evidence(record, language)
        if parts:
            lines.append(f"{message(language, 'evidence')}: {', '.join(parts)}")
    late_s = (now_utc - record.ended_utc).total_seconds()
    if late_s > late_after_s:
        lines.append(message(language, "late", n=round(late_s)))
    if repeats:
        last = (last_utc or now_utc).astimezone(ZoneInfo(timezone)).strftime("%H:%M:%S")
        lines.append(message(language, "repeats", n=repeats, t=last))
    if not monitoring:
        lines += ["", message(language, "footer")]
    return "\n".join(lines)


def video_caption(sidecar: dict[str, Any], *, language: Language, timezone: str) -> str:
    """Requested and recorded windows of a clip, from its JSON sidecar."""

    def hm(key: str) -> str:
        value = datetime.fromisoformat(sidecar[key]).astimezone(ZoneInfo(timezone))
        return value.strftime("%H:%M:%S")

    lines = [
        f"{message(language, 'requested')} {hm('requested_start')} - {hm('requested_end')}",
        f"{message(language, 'recorded')} {hm('actual_start')} - {hm('actual_end')}",
    ]
    if sidecar.get("gaps"):
        lines.append(message(language, "gaps", n=len(sidecar["gaps"])))
    if sidecar.get("truncated"):
        lines.append(message(language, "truncated"))
    return "\n".join(lines)


__all__ = [
    "Language",
    "alert_text",
    "button_labels",
    "delivery_test_text",
    "message",
    "video_caption",
]
