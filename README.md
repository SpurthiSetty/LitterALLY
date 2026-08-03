# LitterALLY — a smart bin on the Arduino UNO Q

Hold something in front of it. Three seconds later the bin tells you which
container it belongs in — with a colour, an icon and a tone — and writes what it
saw to a log you can then ask questions about, in plain English, in a browser.

Built for the Arduino Intern Challenge on an **Arduino UNO Q (4 GB)**.

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
| **router** | **0.0 s** | anything in the log or the rules file |
| **cloud** | ~8 s | the long tail, only if you tick the box |
| **local** | ~9 s | the long tail, offline and private |

**The router runs first, always.** It pattern-matches the question, calls one of
four tools over SQLite and the rules file, and formats the result — so the
number goes from the database to the screen with no step in between that could
corrupt it. Asked what had been thrown out that day, the local model once
answered *20 items* when the log held *303*. The router cannot make that
mistake, and it is instant.

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

## Things that turned out to matter

**A `String` parameter never reaches an MCU handler.** `set_feedback("recycle")`
timed out every single time, so the sketch fell back to its own timeout and
displayed `unknown` even when the classifier was certain. Zero-argument handlers
worked fine, which is exactly why the startup handshake succeeded and hid this
for hours. The category now travels as an integer index.

**`Bridge.notify` does not reach `provide_safe` handlers.** Swapping `call` for
`notify` to dodge a crash silently removed feedback altogether. The MPU→MCU
direction is `Bridge.call(..., timeout=2)` inside a `try`, so a dead MCU costs
two seconds rather than the process.

**The handshake has to repeat.** `mpu_ready` was announced once at startup —
then reflashing rebooted the MCU, which came up having never heard it. It is now
re-announced every 5 seconds, so the two sides resynchronise no matter which one
restarts.

**An AA battery is 6% of the frame.** It classified as `paper` — from the white
counter behind it — while the same battery filling a phone screen scored 0.99.
The model was never wrong about batteries; it was being shown a kitchen. A
classifier labels the whole image and pools features across all of it, so
whatever dominates the frame dominates the answer. **Presentation moved results
more than any model or preprocessing change.**

**Cropping to fix that made it worse.** A distance-scaled centre crop cut the
object out — the banana already filled the frame. Background subtraction found
the largest thing that changed, which is always the hand, because fingers touch
the item and become one connected region. Both paths are still in
`detector.py`, switched off, with what they measured written down.

**Only a square crop survived.** 640×480 → 480×480 before the resize to the
model's square input, which removes the aspect distortion without removing any
of the subject. That one stayed.

**A confidence floor of 0.65 was rejecting correct answers** about half the
time. Real items measured between 0.44 and 0.92, empty scenes between 0.30 and
0.60. The floor is 0.50 — the overlap is real, and no threshold separates them
cleanly.

**Softmax over an already-softmaxed output flattens everything.** The quantized
model emits probabilities, not logits; softmaxing them a second time drove every
class to 1/N ≈ 0.001. Quantized outputs are renormalised, float logits are
softmaxed, and nothing gets both.

**Labels are model vocabulary, not human vocabulary.** Nobody asks which bin
`biological` goes in, so "where does a banana go" fell straight through to the
language model, which said recycle. It is compost. `disposal_rules.yaml` now
carries an alias table, and it is correctness rather than convenience.

**Free tiers are shared pools.** The cloud path took 3 s, then 36 s, then 68 s
for the same question. Three models were swapped chasing it before the
provider's own error named the cause: `429 — limit_source:
upstream_provider_shared_pool`. Moving to a provider with a per-key quota fixed
what changing models could not.

---

## Hardware

- Arduino UNO Q (4 GB)
- Modulino Distance (ToF) and Modulino Buzzer, on the Qwiic bus — **MCU side**
- USB webcam, 640×480 — **MPU side**; the MCU cannot reach it
- Onboard 13×8 monochrome LED matrix and 2 MCU-driven RGB LEDs
