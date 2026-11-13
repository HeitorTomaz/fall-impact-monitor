# Deployment on a Linux server

How to install, check, update, roll back and back up Fall Impact Monitor on a Linux server
with Docker Compose. The commands assume Ubuntu Server LTS on x86-64 and a user with
`sudo`. No GPU is needed.

Fall Impact Monitor is experimental assistive software, not a medical device. Read the
[known limitations](../README.md#known-limitations) before relying on it.

## 1. Requirements

- **CPU:** x86-64. Analysis cost grows with the number of cameras and people. On a desktop
  CPU with 4 cores, one camera analysed 5 times per second with a person in view used
  about 70 % of those cores, and the default of 8 times per second needs all four
  ([`benchmarks/pipeline.md`](benchmarks/pipeline.md)); a slower server analyses as fast as
  it can. Measure your server with `fim benchmark pipeline` (section 10) before adding
  cameras.
- **Memory:** about 1 GB for the service, plus PostgreSQL and MediaMTX.
- **Disk:** room for the recordings (`FIM_RECORDINGS_MAX_GB`, 200 GB by default) plus
  `FIM_RECORDINGS_DISK_MIN_FREE_GB` (10 GB) and about 10 GB for images and event clips.
- **Network:** the cameras' RTSP streams on the local network; outbound HTTPS for
  Telegram. Nothing needs to be reachable from the internet.
- **Time:** keep the server clock synchronised (NTP is on by default in Ubuntu).

## 2. Install Docker Engine

From the [official instructions](https://docs.docker.com/engine/install/ubuntu/):

```bash
sudo apt update
sudo apt install ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker   # start Docker at every boot
sudo usermod -aG docker "$USER"      # then log out and in again
```

## 3. Install Fall Impact Monitor

```bash
git clone https://github.com/HeitorTomaz/fall-impact-monitor.git
cd fall-impact-monitor
cp .env.example .env
chmod 600 .env                # it holds the camera passwords and the Telegram token
mkdir -p local/zones          # before the first start, so that the folder is yours
```

Edit `.env` (every variable is described in the [README](../README.md#configuration)):

- `FIM_CAMERAS` and `FIM_CAMERA_<ID>_MAIN_URL` / `_SUB_URL` for each camera. **Reserve
  each camera's address in the router first** (DHCP reservation, or a fixed address on the
  camera): when the router gives the camera a new address, the URLs stop working and
  monitoring stays degraded until someone edits `.env`;
- `FIM_DB_PASSWORD`: a long random value, for example from `openssl rand -base64 32`.
  PostgreSQL stores it on its first start: keep it;
- `FIM_ALERT_LANGUAGE` and `FIM_ALERT_TIMEZONE`;
- Telegram (`FIM_TELEGRAM_*`): leave `FIM_TELEGRAM_ENABLED=false` until the token and
  the chat ids are filled in.

Build and start (the first build downloads and verifies the model weights; it takes a
few minutes):

```bash
docker compose up -d --build
docker compose ps
```

After about two minutes `app` and `postgres` are `healthy`, `mediamtx` is `running`
(its official image has no shell, so it has no health check) and the one-shot services
`init-volumes` and `mediamtx-config` have exited with code 0.

## 4. First checks

1. `docker compose run --rm app fim alert test` sends a test message to every enabled
   output and reports each one.
2. In the Telegram group, `/status` lists every camera as connected and analysing. A
   camera on a ceiling may be mounted upside down: prefer its own image-flip setting (it
   also fixes the recordings and clips), or set `FIM_CAMERA_<ID>_ROTATE_DEG=180`. Once
   people have walked by, `/status` says when a camera's image looks upside down.
3. `docker compose logs --tail 50 app` shows no errors.
4. Set the safe zones of each camera (beds, sofas): see
   [`fall-detection.md`](fall-detection.md#zones). Then restart the service with
   `docker compose up -d`.
5. Follow the calibration period and the go-live checklist in [`pilot.md`](pilot.md).

## 5. Reboot recovery

Docker starts at boot (`systemctl enable docker`) and every long-running service has
`restart: unless-stopped`, so the system comes back on its own after a power cut or a
reboot. Check it once: `sudo reboot`, then `docker compose ps` and `/status` in Telegram.
The service reports "monitoring restored" for each camera that reconnects.

Services stopped with `docker compose stop` stay stopped after a reboot; start them again
with `docker compose up -d`.

## 6. Network

- No port is published: PostgreSQL and MediaMTX are reachable only on the internal Docker
  network. Do not add `ports:` to `compose.yaml` unless you need LAN access, and never
  forward a port from the internet.
- Telegram uses outbound long polling: no inbound connection, no webhook URL.
- Cameras: keep their addresses reserved in the router (section 3).
- Cameras: where the router allows it, block the cameras' internet access but keep NTP
  allowed (some cameras stop streaming without a time source). Do not let cameras update
  their firmware automatically, and write down the firmware version: a firmware change can
  change the stream's timing.

## 7. Update

Before every update, keep the current image and a database backup, so that you can go
back (section 8):

```bash
cd fall-impact-monitor
docker tag fall-impact-monitor-app:local fall-impact-monitor-app:previous
git rev-parse HEAD > local/previous-commit
docker compose exec -T postgres pg_dump -U fim -d fim -Fc > local/before-update.dump
git pull
docker compose up -d --build
```

Database migrations run when the service starts. Then check, before leaving the site:

- [ ] `docker compose ps`: `app` and `postgres` healthy, `mediamtx` running.
- [ ] `/status` in Telegram answers and lists every camera as connected.
- [ ] `docker compose run --rm app fim alert test` reaches every output.
- [ ] Every zone file passes `docker compose run --rm app fim zones --check <file>`.
- [ ] `docker compose logs --tail 100 app` shows no errors.
- [ ] Read the release's pull requests for actions needed (new variables, zone format).

## 8. Rollback

If the update does not work, go back to the previous commit and image:

```bash
cd fall-impact-monitor
git checkout "$(cat local/previous-commit)"
docker tag fall-impact-monitor-app:previous fall-impact-monitor-app:local
docker compose up -d --no-build
```

If the update changed the database schema (its pull request says so), restore the backup
taken before the update (section 9) after stopping the service:
`docker compose stop app`, restore, then `docker compose up -d --no-build`. Events
recorded between the update and the rollback are lost; the event log
(`/events/events.jsonl`) keeps them.

To return to the current version later: `git checkout main && git pull` and update again.

## 9. Backup and restore

The database holds events, notification states, feedback, the recording index and the
health history. It never holds video. Recordings and event clips are in the `recordings`
and `events` volumes.

Back up (any time; the service keeps running):

```bash
docker compose exec -T postgres pg_dump -U fim -d fim -Fc > local/fim-$(date +%F).dump
```

Copy the file off the server (it contains household events: keep it private).

Restore (replaces the database contents):

```bash
docker compose stop app
docker compose exec -T postgres pg_restore -U fim -d fim --clean --if-exists \
  < local/fim-YYYY-MM-DD.dump
docker compose up -d
```

Test a restore once after installation and after every update that changes the schema.

## 10. Benchmark

`fim benchmark pipeline` measures what one camera costs on this server: decoding, person
detection and pose per analysed frame, the highest sustainable analysis rate, CPU and
memory. Run it on a recording or a public video with people in it:

```bash
docker compose run --rm -v /path/to/videos:/media:ro app \
  fim benchmark pipeline --source /media/<video>.mp4
```

Published results per hardware class are in
[`benchmarks/pipeline.md`](benchmarks/pipeline.md).
