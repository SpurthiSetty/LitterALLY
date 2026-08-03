# LitterALLY: a smart bin built with Arduino UNO Q

Hold an item in front of the bin, and within three seconds it identifies the correct waste stream. A color, icon, and buzzer guide the user, while every classification is recorded in a log. Through a browser-based interface, you can then ask questions about what the bin has seen using plain English.
Built for the Arduino Intern Challenge on an Arduino UNO Q (4 GB).

---

## The board is two computers

That is the shape of the whole project, and most of the design follows from it.

| | MCU | MPU |
|---|---|---|
| Chip | STM32U585 Cortex-M33 | Qualcomm QRB2210, quad Cortex-A53 |
| Runs | Zephyr, `sketch/sketch.ino` | Debian, Python in a container |
| Owns | ToF sensor, buzzer, RGB LEDs, LED matrix | USB webcam, model, database, chat |

They talk over the **Arduino Bridge** — MessagePack RPC, both directions.

**The MCU owns everything the user sees.** Linux asks for a display change; the
MCU decides when and for how long. If the Linux side dies the bin still responds
— it just shows `unknown` after a four-second timeout. That property is why the
split is worth the trouble.

The Bridge contract is three calls:

| Call | Direction | Meaning |
|---|---|---|
| `on_trigger(mm)` | MCU → MPU | someone held something still; classify it |
| `set_feedback(index)` | MPU → MCU | show category *index*, 0–5 |
| `scene_clear()` | MCU → MPU | item withdrawn, drop any pending work |

`on_trigger` returns immediately and does no work — the result comes back later
as a separate call, so the MCU never blocks on Linux.

---

## What happens when you present something

```
  distance sensor  ─┐
   10 Hz, Kalman    │  held 3 s
                    ▼
              on_trigger(mm) ──────────────► capture 5 frames
                                                    │
                                          MobileNetV2 (wastenet)
                                                    │
                                        mean over the 5 frames
                                                    │
                                     label ──► disposal_rules.yaml
                                                    │
                        SQLite  ◄──── category ─────┤
                                                    ▼
                                            set_feedback(index)
                                                    │
                                     RGB LEDs + matrix + buzzer
```

**Distance is filtered, not thresholded.** The ToF sensor drops readings — it
returns 0 when it has no target — and a raw threshold read one dropped sample as
a withdrawn item and restarted the three-second hold. A small constant-velocity
Kalman filter tracks distance *and its rate of change*, so a dropped sample
coasts on the prediction instead of resetting the state machine. Presence uses
hysteresis (arm at 45–240 mm, disarm below 25 or above 265) because an item
resting exactly on a single threshold flips state ten times a second however
well it is filtered.

**Vision returns a label, not a category.** `label → category` lives in
[`python/disposal_rules.yaml`](python/disposal_rules.yaml), so changing disposal
policy never means retraining. Five categories, plus `unknown`.

**Five frames, not one.** A single frame catches motion blur or an awkward
angle. The five probability distributions are averaged, so a label has to do
well across the burst rather than get lucky once. Configurable via
`SMARTBIN_BURST`.

Measured on the board: **175 ms per frame, 102 MB RSS**, about 0.9 s for a
five-frame burst against the MCU's 4 s timeout.

### The five categories

| Category | Colour | Matrix icon | Tone | Reached by |
|---|---|---|---|---|
| `recycle` | blue | **R** | 880 Hz | cardboard, paper, plastic, metal, glass |
| `compost` | green | **C** | 660 Hz | biological |
| `trash` | white | bin | 440 Hz | clothes, shoes, trash |
| `hazardous` | red | **!** | 1320 Hz | battery |
| `ewaste` | magenta | bolt | 1100 Hz | *chat only — see below* |
| `unknown` | yellow | **?** | 220 Hz | confidence below 0.50 |

The onboard 13×8 matrix is monochrome, so the icon carries the meaning and the
two RGB LEDs carry the colour. The order of that table is a contract:
`set_feedback` sends the row index, and `sketch.ino` holds the matching table.

`ewaste` is deliberately unreachable from the camera — wastenet has no e-waste
class. It exists so the chatbot can answer "where does a laptop go", which it
otherwise handed to the language model, which replied *the third bin, with the
lid closed*.

---

## The chatbot

Two kinds of question — what this bin has seen, and how to dispose of something
— answered by whichever of three tiers can handle it. Served at
`http://<board>:8090`.

| Tier | Speed | Handles |
|---|---|---|
| **offline** | **0.0 s** | anything in the log or the rules file, no model at all |
| **cloud** | ~8 s | the long tail, only if you tick the box |
| **local** | ~9 s | the long tail, on-device |

**The offline tier runs first, always.** It pattern-matches the question, calls
one of four tools over SQLite and the rules file, and formats the result — so
the number goes from the database to the screen with no step in between that
could corrupt it. Asked what had been thrown out that day, the local model once
answered *20 items* when the log held *303*. A tier with no model in it cannot
make that mistake, and it is instant.

**Cloud and local are alternatives, not escalating tiers.** If cloud is
permitted and reachable it answers directly; running the 0.8 B local model first
would add ten seconds on the way to a worse answer. Local is what remains when
the toggle is off, the key is missing, or the network is down — so the bin
always answers something. When cloud fails mid-stream the answer is re-labelled
`local (cloud unavailable)` rather than silently attributed to the wrong model.

**The toggle is about consent, not capability.** The event log never leaves the
device. Only the question does, and only when you ask for it.

The bin knows where it is — `location` in `disposal_rules.yaml` — so cloud
answers name real local services instead of saying "check with your council".

Answers stream token by token over newline-delimited JSON, because a nine-second
wait with a blank screen reads as broken.

---

## Layout

```
app.yaml                     App Lab manifest: bricks and published ports
deploy.sh                    tar-over-ssh push from a laptop to the board
python/
  main.py                    orchestrator: trigger → classify → rules → store → feedback
  vision.py                  camera, burst capture, and the seam into the classifier
  detector.py                experiments in cropping to the item (off by default)
  rules.py                   label + confidence → category
  disposal_rules.yaml        THE disposal policy: labels, aliases, location, floor
  store.py                   SQLite event log
  chat.py                    three-tier routing
  chat_tools.py              what the chatbot is allowed to know
  chat_server.py             NDJSON streaming server for the chat UI
  chat_ui.html               the browser front end
  classification/            TFLite classifier + download_models.py
sketch/
  sketch.ino                 state machine, Kalman filter, LEDs, matrix, buzzer
```

The `python/` + `sketch/` split and the root `app.yaml` are required by App Lab,
not decoration.

---

## Running it

App Lab edits files **on the board** and keeps no local copy. Either clone into
`~/ArduinoApps/` on the UNO Q, or work in the repo on a laptop and push with
`./deploy.sh`.

Once, on the board, fetch the model weights (~20 MB, gitignored):

```bash
python3 python/classification/download_models.py
```

Then press **Run** in App Lab. The chat UI is at `http://<board>:8090`.

`python/requirements.txt` pulls `opencv-python-headless`, `pyyaml`, `openai` and
`ai-edge-litert`. **`tflite-runtime` will not install on this board** — there is
no wheel for Python 3.13 on aarch64 — so `ai-edge-litert`, Google's continuation
of it, runs the same `.tflite` files through the same `Interpreter` API.

### The cloud tier

Put an API key in `python/.cloud_key` (gitignored, and `deploy.sh` skips
dotfiles). It defaults to Google AI Studio's free tier; any OpenAI-compatible
endpoint works via `SMARTBIN_CLOUD_URL`.

The `arduino:cloud_llm` brick is **commented out** in `app.yaml` on purpose: it
hard-requires an `Api_key` variable and the app will not load without one,
appearing unnamed and unstartable in App Lab. Everything behind it — the UI
toggle, the routing, the fallback — is written and committed.

### Testing the chat without the board

The chat layer is plain Python over SQLite, so it runs on a laptop against a
copy of the database:

```bash
scp unoq:'~/ArduinoApps/<project>/python/smartbin.db' /tmp/smartbin.db
```

```bash
python3 python/chat_server.py --db /tmp/smartbin.db --port 8090
```

### Configuration

| Variable | Default | Does |
|---|---|---|
| `SMARTBIN_BURST` | `5` | frames per classification |
| `SMARTBIN_DETECT` | `off` | crop-to-item mode: `off`, `diff`, `yolox` |
| `SMARTBIN_CLOUD_URL` | Google AI Studio | any OpenAI-compatible endpoint |
| `SMARTBIN_CLOUD_MODEL` | `gemini-flash-lite-latest` | cloud model id |
| `SMARTBIN_CLOUD_MAX_TOKENS` | `1024` | cloud response cap |

Model ids are an alias, not a pinned version, because two pinned versions went
stale during development — one was already retired when it was written down.

---

## Hardware

- Arduino UNO Q (4 GB)
- Modulino Distance (ToF) and Modulino Buzzer, on the Qwiic bus — **MCU side**
- USB webcam, 640×480 — **MPU side**; the MCU cannot reach it
- Onboard 13×8 monochrome LED matrix and 2 MCU-driven RGB LEDs
