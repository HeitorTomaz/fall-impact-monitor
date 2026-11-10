# Pilot: calibration, fault injection and go-live

This is the procedure for bringing an installation into service. Measurements from a
private installation stay in the git-ignored `local/` directory; publish only methods
and anonymized summaries.

## 1. Calibration period

1. Configure only the maintainer's chat in `FIM_TELEGRAM_CHAT_IDS`.
2. Run for at least a week of normal life. Answer every alert with False alarm or
   Confirmed, and note missed events with their time.
3. Stage falls on a mattress, with and without immediate recovery, in several
   directions and at several distances, plus everyday activities: sitting down fast,
   lying on the sofa or bed, crouching, picking an object off the floor. Record the
   time of each one.
4. Review the JSON event log and the database: alerts, near-misses and their
   confidence. Adjust `FIM_ALERT_MIN_CONFIDENCE`, the person-down limits and the zones
   (see [`fall-detection.md`](fall-detection.md#calibration-procedure)).
5. Add the other recipients only when the false alarms per day are acceptable and the
   staged falls are detected.

Targets for the staged scenes (measured, not guaranteed):

- at least 20 falls, of which at least 10 with immediate recovery;
- at least 16 alerted;
- p95 alert latency at most 30 s and none above 60 s;
- false alarms of the everyday activities counted per hour.

## 2. Fault injection

Repeat each test and write down the result. "Degraded" and "restored" refer to the
Telegram messages (they also appear in the JSON event log).

| Fault | How | Expected |
|---|---|---|
| Camera off (×3) | Unplug the camera's power | "Monitoring degraded: no video from the camera" within 60 s (default 30–35 s); "restored" about 30 s after video returns (plus the camera's boot time); no manual action |
| Network loss to the camera | Disconnect its Wi-Fi or cable | Same as above |
| PostgreSQL down | `docker compose stop postgres`, trigger a staged fall, then `docker compose start postgres` | The alert still arrives; after the restart the event appears in the `events` table; "degraded: database not reachable" if it stays down longer than `FIM_HEALTH_BACKLOG_MAX_AGE_S` |
| Disk full | Fill the recordings disk with a dummy file until less than `FIM_RECORDINGS_DISK_MIN_FREE_GB` is free (for example `fallocate -l <size> /path/on/that/disk/fill.tmp`), then delete it. **Do not raise the floor instead: maintenance would delete recordings until the floor is met, which may remove all of them.** | "degraded: disk below the free-space floor" within a minute; maintenance deletes the oldest recordings it can; event clips are kept; "restored" about 30 s after the dummy file is removed |
| Internet down | Unplug the router's uplink, trigger a staged fall, reconnect | Detection goes on; the alert arrives after reconnection, marked late; queued items survive a restart |
| App crash | `docker compose kill app` | Docker restarts it; pending alerts in the outbox are delivered after the restart |
| Stuck processing | (Not reproducible on demand) | The watchdog exits after `FIM_HEALTH_STALL_EXIT_S`; Docker restarts the service; "processing stalled, restarting" is delivered after the restart |

What CI covers automatically:

- the database being unreachable;
- outbox files damaged by a power cut;
- a camera source failing and being reopened;
- corrupt packets;
- a stuck worker;
- a camera that stops sending video;
- the low-disk and backlog conditions.

## 3. Long run

Run for several days on the target hardware and record:

- memory and CPU of `app`, disk growth of `recordings` and `events`;
- reconnections and recorder health (MediaMTX logs; `fim recordings index` lists gaps);
- false alarms per hour;
- analysis rate per camera (`/status`).

Items moved here from earlier issues that need the real camera:

- live timing experiments: direct vs relay, main stream vs substream, video-only vs all media;
- the stopwatch/LED ground-truth test at 15/20/25/30 fps;
- the 24–48 h recorder run;
- the retention fill test on the site disk, with `FIM_RECORDINGS_MAX_GB` sized to it;
- the capacity benchmark;
- two simultaneous cameras: one real and one looped file.

## 4. Go-live checklist

- [ ] `docker compose ps`: every service running; `app` and `postgres` healthy.
- [ ] `docker compose run --rm app fim alert test`: every enabled output delivered.
- [ ] `/status` in the group answers with every camera OK.
- [ ] A staged fall produces an alert, Seen/False alarm/Confirmed are recorded, and
      **Watch video** delivers the clip with its requested and recorded windows.
- [ ] Unplugging the camera produces "degraded"; plugging it back produces "restored".
- [ ] The previous app image is tagged for rollback (see [`deployment.md`](deployment.md#7-update)).
- [ ] Every zone file passes `fim zones --check` (run it again after each update) and its
      `zones.png` shows each safe zone on the soft surface only.
- [ ] CI is green on the installed commit, and no secrets or private media are in Git.
- [ ] Everyone in the group knows the limitations below and that the system does not
      replace supervision.
