# ADR-0006: Optional event sinks: Telegram, webhook and JSON log

- Status: Accepted
- Date: 2026-11-08

## Context

The reference installation notifies a Telegram group; the system's responsibility ends at that group. Other adopters may want to integrate detection into their own tools instead. Notifications must be honest about uncertainty, avoid alert fatigue without hiding real events, and respect Telegram limits (about 50 MB per bot upload and roughly one message per second per chat).

## Decision

Events go to **sinks**, each enabled by configuration and independent of the database (ADR-0005):

- **JSON event log**: always on; one JSON object per event, for auditing and simple integrations.
- **Generic webhook**: HTTP POST of the event as JSON to a configured URL (e.g. `FIM_WEBHOOK_URL`), with retries.
- **Telegram** (e.g. `FIM_TELEGRAM_ENABLED`): implemented with **aiogram 3** (MIT) using outbound long polling (no inbound port, no webhook). The bot answers only in the configured chats.
  - Alert messages are **text**, with inline buttons **Seen / False alarm / Confirmed**; feedback is stored for evaluation.
  - A **View video** button sends the event clip to the group on demand, as a normal message, as MP4 (H.264 with AAC audio), within the upload limit.
  - `/status` reports the state of each camera and of the system; `/missed` (pt-BR alias `/perdido`) records a **missed event** at a given time and protects its recording from retention.
  - Degraded and restored monitoring states are sent as distinct messages, never confused with "no incident".

**Aggregation and repeats.** Repeated events of the same type from the same person with a **stable** identity within a few minutes update the existing message instead of creating a new one. The update is silent **until someone presses Seen**; after that, a repeat sends a **new reply that notifies**, so a second real event is never silent. A different person or an uncertain identity always produces a new alert.

**Wording.** Messages say "possible fall", "person on the floor for N s" or "possible head impact (experimental)", with time, camera, evidence and the confidence score as a percentage (only events at or above the configured threshold are sent; see ADR-0004). They never say "confirmed" and never make claims about injury; a short footer states the system does not replace supervision.

**Localization.** Notification texts are localized: English by default, Brazilian Portuguese provided, selected by configuration. Logs and CLI messages stay in English.

## Alternatives considered

- **python-telegram-bot** (LGPL-3.0) and a hand-written HTTP client: viable; aiogram chosen for its MIT licence and native asyncio.
- **pyTelegramBotAPI** (GPL-2.0): incompatible with the project licence.
- **MQTT with Home Assistant discovery**: valuable for the community but needs a broker service; planned after the MVP.
- **Images or clips attached to every alert**: rejected as default; video leaves the host only on explicit request.

## Consequences

- Clips sent to the group are stored on Telegram's servers (group chats are not end-to-end encrypted); this is documented.
- Sinks can be added without touching detection code.
- Rate limits and outages are handled by a retrying queue; alerts are queued while the internet is down.

## References

- https://docs.aiogram.dev/
- https://core.telegram.org/bots/api
- https://core.telegram.org/bots/faq
