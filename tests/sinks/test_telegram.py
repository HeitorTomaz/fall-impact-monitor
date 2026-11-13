import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fall_impact_monitor.events import EventCategory
from fall_impact_monitor.sinks.telegram.api import Button, callback_data, parse_callback
from fall_impact_monitor.sinks.telegram.controller import TelegramController
from fall_impact_monitor.sinks.telegram.sink import TelegramChatSink
from tests.sinks.helpers import T0, make_record

CHAT = -100
EID = "8f9a3c1e-2b4d-4e6f-8a0b-1c2d3e4f5a6b"


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._next_id = 10

    async def send_message(
        self, chat_id: int, text: str, *, buttons: list[Button], reply_to: int | None, silent: bool
    ) -> int:
        self._next_id += 1
        self.calls.append(
            (
                "send",
                {
                    "chat": chat_id,
                    "text": text,
                    "buttons": buttons,
                    "reply_to": reply_to,
                    "silent": silent,
                    "id": self._next_id,
                },
            )
        )
        return self._next_id

    async def edit_message(
        self, chat_id: int, message_id: int, text: str, *, buttons: list[Button]
    ) -> None:
        self.calls.append(("edit", {"chat": chat_id, "id": message_id, "text": text}))

    async def send_video(
        self, chat_id: int, path: Path, *, caption: str, reply_to: int | None
    ) -> int:
        self.calls.append(
            ("video", {"chat": chat_id, "path": path, "caption": caption, "reply_to": reply_to})
        )
        return 99

    async def answer(self, callback_id: str, text: str) -> None:
        self.calls.append(("answer", {"id": callback_id, "text": text}))


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def make_sink(
    api: FakeApi, clock: Clock, reports: list[dict[str, Any]], **kw: Any
) -> TelegramChatSink:
    async def no_sleep(seconds: float) -> None:
        kw.setdefault("_slept", []).append(seconds)

    return TelegramChatSink(
        api,
        CHAT,
        language="en",
        timezone="UTC",
        aggregation_window_s=kw.get("window", 300.0),
        late_after_s=60.0,
        report=reports.append,
        clock=clock,
        min_interval_s=0.0,
    )


def event_item(i: int = 0, **kw: Any) -> dict[str, Any]:
    return {"kind": "event", "record": make_record(i, **kw).to_json()}


def deliver(sink: TelegramChatSink, *items: dict[str, Any]) -> None:
    async def main() -> None:
        for item in items:
            await sink.deliver(item)

    asyncio.run(main())


def test_a_new_alert_has_feedback_and_video_buttons_and_is_reported() -> None:
    api, reports = FakeApi(), []
    sink = make_sink(api, Clock(T0 + timedelta(seconds=5)), reports)
    item = event_item(0)
    deliver(sink, item)
    kind, call = api.calls[0]
    assert kind == "send" and call["chat"] == CHAT and not call["silent"]
    assert "Possible fall" in call["text"]
    assert [b.action for b in call["buttons"]] == ["seen", "false_alarm", "confirmed", "video"]
    assert {b.event_id for b in call["buttons"]} == {item["record"]["event_id"]}
    assert reports == [
        {
            "kind": "notification",
            "event_id": item["record"]["event_id"],
            "sink": sink.name,
            "message_ref": str(call["id"]),
            "sent_utc": (T0 + timedelta(seconds=5)).isoformat(),
            "late_s": 4.0,
        }
    ]


def test_a_repeat_before_seen_edits_the_message_silently() -> None:
    api, reports = FakeApi(), []
    clock = Clock(T0 + timedelta(seconds=5))
    sink = make_sink(api, clock, reports)
    deliver(sink, event_item(0))
    clock.now = T0 + timedelta(seconds=60)
    deliver(sink, event_item(1))
    assert [kind for kind, _ in api.calls] == ["send", "edit"]
    assert "Repeated 1 times" in api.calls[1][1]["text"]
    assert len(reports) == 2


def test_a_repeat_after_seen_sends_a_new_notifying_reply() -> None:
    api, reports = FakeApi(), []
    clock = Clock(T0 + timedelta(seconds=5))
    sink = make_sink(api, clock, reports)
    deliver(sink, event_item(0))
    first_id = api.calls[0][1]["id"]
    sink.mark_seen(first_id)
    clock.now = T0 + timedelta(seconds=60)
    deliver(sink, event_item(1))
    kind, call = api.calls[1]
    assert kind == "send" and call["reply_to"] == first_id and not call["silent"]


def test_uncertain_identity_or_another_person_always_gets_a_new_alert() -> None:
    api, reports = FakeApi(), []
    sink = make_sink(api, Clock(T0 + timedelta(seconds=5)), reports)
    deliver(sink, event_item(0, identity="uncertain"), event_item(1, identity="uncertain"))
    deliver(sink, event_item(2, track_id=7))
    assert [kind for kind, _ in api.calls] == ["send", "send", "send"]


def test_a_repeat_outside_the_window_is_a_new_alert() -> None:
    api, reports = FakeApi(), []
    clock = Clock(T0 + timedelta(seconds=5))
    sink = make_sink(api, clock, reports)
    deliver(sink, event_item(0))
    clock.now = T0 + timedelta(seconds=400)
    deliver(sink, event_item(1, at=T0 + timedelta(seconds=395)))
    assert [kind for kind, _ in api.calls] == ["send", "send"]


def test_near_misses_are_never_sent() -> None:
    sink = make_sink(FakeApi(), Clock(T0), [])
    assert not sink.accepts(event_item(0, near_miss=True))
    assert sink.accepts({"kind": "test"})


def test_a_late_alert_is_sent_and_marked_late() -> None:
    api, reports = FakeApi(), []
    sink = make_sink(api, Clock(T0 + timedelta(seconds=1 + 75)), reports)
    deliver(sink, event_item(0))
    assert "Late: sent 75 s after the event" in api.calls[0][1]["text"]
    assert reports[0]["late_s"] == 75.0


def test_sends_to_one_chat_are_spaced() -> None:
    api = FakeApi()
    slept: list[float] = []
    ticks = iter([100.0, 100.2, 100.2, 101.0])

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    sink = TelegramChatSink(
        api,
        CHAT,
        language="en",
        timezone="UTC",
        aggregation_window_s=0.0,
        late_after_s=60.0,
        report=lambda item: None,
        clock=Clock(T0),
        min_interval_s=1.0,
        sleep=sleep,
        monotonic=lambda: next(ticks),
    )
    deliver(sink, event_item(0), event_item(1))
    assert len(slept) == 1 and abs(slept[0] - 0.8) < 1e-6


def controller(
    api: FakeApi, tmp_path: Path, submitted: list[dict[str, Any]], **kw: Any
) -> tuple[TelegramController, TelegramChatSink]:
    sink = make_sink(api, Clock(T0 + timedelta(seconds=5)), [])
    return (
        TelegramController(
            api,
            {CHAT: sink},
            submitted.append,
            clip_dir=tmp_path,
            language="en",
            timezone="UTC",
            status_text=lambda: "cam1: OK",
            clock=Clock(T0 + timedelta(seconds=9)),
            **{"clip_wait_s": 60.0, **kw},
        ),
        sink,
    )


def write_clip(clip_dir: Path, event_id: str = EID) -> Path:
    folder = clip_dir / "cam1" / "2026" / "11" / "09"
    folder.mkdir(parents=True, exist_ok=True)
    clip = folder / f"030000_{event_id}.mp4"
    clip.write_bytes(b"mp4")
    clip.with_suffix(".json").write_text(
        json.dumps(
            {
                "requested_start": "2026-11-09T02:59:40+00:00",
                "requested_end": "2026-11-09T03:01:01+00:00",
                "actual_start": "2026-11-09T02:59:39+00:00",
                "actual_end": "2026-11-09T03:01:01+00:00",
                "gaps": [],
                "truncated": False,
            }
        )
    )
    return clip


def test_buttons_and_commands_from_other_chats_are_ignored(tmp_path: Path) -> None:
    api, submitted = FakeApi(), []
    ctrl, _ = controller(api, tmp_path, submitted)
    asyncio.run(ctrl.on_button(12345, 11, "cb", callback_data("seen", EID)))
    assert asyncio.run(ctrl.on_command(12345, "status")) is None
    assert api.calls == [] and submitted == []


def test_feedback_is_recorded_and_marks_the_alert_seen(tmp_path: Path) -> None:
    api, submitted = FakeApi(), []
    ctrl, sink = controller(api, tmp_path, submitted)
    deliver(sink, event_item(0))
    message_id = api.calls[0][1]["id"]
    asyncio.run(ctrl.on_button(CHAT, message_id, "cb", callback_data("false_alarm", EID)))
    assert submitted == [
        {
            "kind": "feedback",
            "event_id": EID,
            "action": "false_alarm",
            "created_utc": (T0 + timedelta(seconds=9)).isoformat(),
        }
    ]
    assert api.calls[-1] == ("answer", {"id": "cb", "text": "Recorded, thank you."})
    assert sink.is_seen(message_id)


def test_a_press_before_the_clip_exists_sends_it_when_it_appears(tmp_path: Path) -> None:
    api, slept = FakeApi(), []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        if len(slept) == 2:  # the export finishes while the controller waits
            write_clip(tmp_path)

    ctrl, _ = controller(api, tmp_path, [], sleep=sleep, clip_wait_s=60.0, clip_poll_s=5.0)
    asyncio.run(ctrl.on_button(CHAT, 11, "cb", callback_data("video", EID)))
    assert api.calls[0] == (
        "answer",
        {"id": "cb", "text": "The video will be sent here as soon as it is ready."},
    )
    assert [kind for kind, _ in api.calls] == ["answer", "video"]
    assert slept == [5.0, 5.0]


def test_a_clip_that_never_appears_is_reported(tmp_path: Path) -> None:
    api, slept = FakeApi(), []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    ctrl, _ = controller(api, tmp_path, [], sleep=sleep, clip_wait_s=20.0, clip_poll_s=5.0)
    asyncio.run(ctrl.on_button(CHAT, 11, "cb", callback_data("video", EID)))
    assert slept == [5.0, 5.0, 5.0, 5.0]
    kind, sent = api.calls[-1]
    assert kind == "send" and sent["reply_to"] == 11
    assert sent["text"] == "The video is not available; it may be missing from the recordings."


def test_a_second_press_while_waiting_does_not_send_twice(tmp_path: Path) -> None:
    api = FakeApi()
    waiting = asyncio.Event()

    async def sleep(seconds: float) -> None:
        waiting.set()
        await asyncio.sleep(0)

    ctrl, _ = controller(api, tmp_path, [], sleep=sleep, clip_wait_s=60.0, clip_poll_s=5.0)

    async def scenario() -> None:
        first = asyncio.create_task(ctrl.on_button(CHAT, 11, "cb1", callback_data("video", EID)))
        await waiting.wait()
        await ctrl.on_button(CHAT, 11, "cb2", callback_data("video", EID))
        write_clip(tmp_path)
        await first

    asyncio.run(scenario())
    assert [kind for kind, _ in api.calls].count("video") == 1
    assert (
        "answer",
        {"id": "cb2", "text": "The video will be sent here as soon as it is ready."},
    ) in api.calls


def test_watch_video_sends_the_clip_with_its_windows(tmp_path: Path) -> None:
    api, submitted = FakeApi(), []
    ctrl, _ = controller(api, tmp_path, submitted)
    clip = write_clip(tmp_path)
    asyncio.run(ctrl.on_button(CHAT, 11, "cb", callback_data("video", EID)))
    kinds = [kind for kind, _ in api.calls]
    assert kinds == ["answer", "video"]
    assert api.calls[0][1]["text"] == "Sending the video..."
    video = api.calls[1][1]
    assert video["path"] == clip and video["reply_to"] == 11
    assert "Requested 02:59:40 - 03:01:01" in video["caption"]


def test_status_command_answers_configured_chats(tmp_path: Path) -> None:
    ctrl, _ = controller(FakeApi(), tmp_path, [])
    assert asyncio.run(ctrl.on_command(CHAT, "status")) == "cam1: OK"


def test_callback_data_round_trip_and_fits_telegram() -> None:
    event_id = "8f9a3c1e-2b4d-4e6f-8a0b-1c2d3e4f5a6b"
    data = callback_data("false_alarm", event_id)
    assert len(data.encode()) <= 64
    assert parse_callback(data) == ("false_alarm", event_id)
    assert parse_callback("other:thing") is None
    assert parse_callback("fim:delete:x") is None


def test_person_down_alerts_aggregate_by_camera_and_category() -> None:
    api, reports = FakeApi(), []
    clock = Clock(T0 + timedelta(seconds=5))
    sink = make_sink(api, clock, reports)
    deliver(
        sink,
        event_item(
            0, category=EventCategory.PERSON_DOWN, evidence={"fall_observed": True, "lying_s": 31}
        ),
    )
    deliver(sink, event_item(1))  # a fall is another category: new message
    assert [kind for kind, _ in api.calls] == ["send", "send"]


class GoneApi(FakeApi):
    async def edit_message(
        self, chat_id: int, message_id: int, text: str, *, buttons: list[Button]
    ) -> None:
        from fall_impact_monitor.sinks.telegram.api import MessageGone

        raise MessageGone("message to edit not found")


def test_a_deleted_alert_message_does_not_block_the_chat() -> None:
    api, reports = GoneApi(), []
    clock = Clock(T0 + timedelta(seconds=5))
    sink = make_sink(api, clock, reports)
    deliver(sink, event_item(0))
    clock.now = T0 + timedelta(seconds=60)
    deliver(sink, event_item(1))  # the edit target is gone: a new, notifying message
    assert [kind for kind, _ in api.calls] == ["send", "send"]
    assert not api.calls[1][1]["silent"]


def test_callbacks_must_carry_an_event_uuid() -> None:
    assert parse_callback("fim:seen:not-a-uuid") is None


def test_chat_ids_never_appear_in_sink_names() -> None:
    sink = make_sink(FakeApi(), Clock(T0), [])
    assert str(CHAT) not in sink.name and sink.name.startswith("telegram:")


def test_watch_video_waits_for_the_sidecar(tmp_path: Path) -> None:
    from fall_impact_monitor.recording.clips import find_clip

    folder = tmp_path / "cam1" / "2026" / "11" / "09"
    folder.mkdir(parents=True)
    (folder / "030000_e1.mp4").write_bytes(b"partial")
    assert find_clip(tmp_path, "e1") is None
    (folder / "030000_e1.json").write_text("{}")
    assert find_clip(tmp_path, "e1") == folder / "030000_e1.mp4"


def test_polling_is_restarted_after_a_failure() -> None:
    from fall_impact_monitor.sinks.telegram.bot import keep_polling

    attempts: list[int] = []

    async def poll() -> None:
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("no internet")

    async def no_sleep(seconds: float) -> None:
        return None

    asyncio.run(keep_polling(poll, sleep=no_sleep))
    assert len(attempts) == 3


def test_video_uploads_use_their_own_timeout(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from fall_impact_monitor.sinks.telegram.api import AiogramApi

    class FakeBot:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        async def send_video(self, chat_id: int, video: Any, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            return SimpleNamespace(message_id=5)

    bot = FakeBot()
    clip = write_clip(tmp_path)
    api = AiogramApi(bot, video_timeout_s=600.0)  # type: ignore[arg-type]
    assert asyncio.run(api.send_video(CHAT, clip, caption="", reply_to=None)) == 5
    assert bot.kwargs["request_timeout"] == 600


def test_pressing_again_during_an_upload_does_not_upload_twice(tmp_path: Path) -> None:
    write_clip(tmp_path)
    uploading = asyncio.Event()
    finish = asyncio.Event()

    class SlowApi(FakeApi):
        async def send_video(
            self, chat_id: int, path: Path, *, caption: str, reply_to: int | None
        ) -> int:
            uploading.set()
            await finish.wait()  # a large clip on a slow uplink
            return await super().send_video(chat_id, path, caption=caption, reply_to=reply_to)

    api = SlowApi()
    ctrl, _ = controller(api, tmp_path, [])

    async def scenario() -> None:
        first = asyncio.create_task(ctrl.on_button(CHAT, 11, "cb1", callback_data("video", EID)))
        await uploading.wait()
        second = asyncio.create_task(ctrl.on_button(CHAT, 11, "cb2", callback_data("video", EID)))
        await asyncio.sleep(0.05)
        finish.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=5)

    asyncio.run(scenario())
    assert [kind for kind, _ in api.calls].count("video") == 1
