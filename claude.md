# Smart Bin

A smart trash bin on an **Arduino UNO Q (4 GB)**. A distance sensor detects
someone approaching, a USB webcam classifies the item, LEDs and a buzzer signal
which bin it belongs in, and a web chatbot answers questions about what's been
thrown out and how to dispose of specific things.

Team of four. Nothing is built yet.

## The board is two computers

| | MPU | MCU |
|---|---|---|
| Chip | Qualcomm QRB2210, quad A53 | STM32U585 Cortex-M33 |
| Runs | Debian Linux, Python in Docker | Zephyr, `sketch.ino` |
| Owns | USB webcam, model, database, web UI | Qwiic bus, all timing-critical work |

They talk over the **Arduino Bridge** (MessagePack RPC, bidirectional).
`/dev/ttyHS1` and `Serial1` are reserved by the router — never opened directly.

Peripherals: Modulino Distance (ToF, mm) and Modulino Buzzer on Qwiic, so
**MCU-side**. Modulino Pixels (8 RGB LEDs) carries the colour signal. The
onboard 8x13 matrix is **monochrome** — icons and brightness only, no colour.
There are also 4 onboard RGB LEDs, 2 driven by each processor. The webcam is USB
and therefore **MPU-only** — the MCU cannot reach it.

## Components

**Vision** (Task 1) — owns the camera and the classifier. Takes a frame, returns
a raw object label plus a confidence score. Has no opinion about disposal.

**Firmware** (Task 2) — `sketch.ino`. Samples distance, debounces, tells the MPU
someone arrived, and drives all the lights and sound.

**Chat** (Task 3) — answers two kinds of question: what this bin has seen, and
how to dispose of a given item. Reads the event database and the rules file.
Local model by default, cloud escalation as a stretch goal.

**Integration** (Task 4) — the orchestrator, the rules engine, the event store,
and the web server. The only component that talks to all the others.

## How it fits together

```
distance trips  ->  MCU debounces  ->  Bridge: on_trigger()
                                            |
                     capture -> classify -> label + confidence
                                            |
                            rules engine -> category
                                            |
                        log to SQLite  ->  Bridge: set_feedback()
                                            |
                                    MCU lights + buzzer
```

The chatbot is a **separate path** with a separate trigger (a browser, not the
sensor). The two share only the database.

## Decided architecture

- **The MCU owns everything the user sees.** The MPU requests a display change;
  the MCU decides when and how long. If the Linux side dies, the bin still
  responds — it just shows `unknown`.
- **`on_trigger` returns immediately** and does no work; the classification
  result comes back later as a separate call. The MCU never blocks on it and
  self-times-out if nothing arrives.
- **Vision returns a label, not a category.** `label -> category` lives in
  `disposal_rules.yaml`, so disposal policy changes without retraining.
- **The chat path never blocks the real-time path.** Different deadlines,
  different threads, shared only through the database.

## Platform gotchas

- `Bridge.provide()` is thread-unsafe; there is a `provide_safe()` variant.
  Most published examples use the unsafe one.
- The Python side starts inside a Docker container and comes up well after the
  MCU, so the two need a startup handshake.
- Anything written to the container filesystem may not survive a redeploy.

## Open questions

- The label vocabulary and whether five disposal categories is the right number.
- Whether the local model is large enough to be useful at 4 GB.
