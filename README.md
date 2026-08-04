# mcufit

**Check if an AI model fits on a microcontroller — before you flash it.**

<p align="center">
  <a href="https://avionicharshit-byte.github.io/mcufit/"><b>🌐 Try it in your browser</b></a>
  &nbsp;·&nbsp;
  <a href="https://pypi.org/project/mcufit/"><b>📦 pip install mcufit</b></a>
</p>

You trained a model. You have a board. Will it run, or will it crash with a
cryptic allocation failure after an hour of toolchain setup? Today the
official answer from the TensorFlow Lite Micro docs is that arena size
"may need to be determined by experimentation." `mcufit` replaces the
experimentation with an answer in one second:

```
$ mcufit check wake_word.tflite --board esp32-s3

  Model:  wake_word.tflite  (int8, 14 layers, 340 KB)
  Board:  ESP32-S3 DevKit  (362 KB usable SRAM · 8 MB flash)

  ✅ FITS

  RAM   ████████████░░░░░░░░  ~289 KB arena / 362 KB   (80%)
  Flash ██░░░░░░░░░░░░░░░░░░  490 KB total  / 8 MB     (6%)

  Peak memory moment: layer 9 (DEPTHWISE_CONV_2D) — 118 KB live tensors

   • Leaves ~73 KB RAM for your application, sensor buffers, and network stack.
```

No hardware required. No vendor lock-in. Works with any `.tflite` model and
any board in the database (ESP32, RP2040, STM32, Teensy, Arduino, ...).

## Install

```
pip install mcufit
```

## Commands

| Command | What it does |
|---|---|
| `mcufit check model.tflite -b esp32-s3` | Fit verdict for one board (exit code 1 if it doesn't fit — CI-friendly) |
| `mcufit check model.tflite -b rp2040 --json` | Same, as JSON for scripts and CI |
| `mcufit compare model.tflite` | Verdict matrix across every board in the database |
| `mcufit inspect model.tflite` | Layer-by-layer memory profile — see *where* the peak is |
| `mcufit boards` | List all known boards |

## How it works

The RAM bottleneck on microcontrollers is the **tensor arena**: every
intermediate activation tensor that is alive at the same moment must fit in
SRAM simultaneously. `mcufit`:

1. **Parses** the `.tflite` flatbuffer directly — layers, tensor shapes,
   dtypes, and which tensors are baked-in weights (flash) vs. runtime
   activations (RAM).
2. **Computes tensor lifetimes** across the execution schedule and finds the
   peak of simultaneously-live activation memory — the same quantity TFLite
   Micro's memory planner must pack into the arena.
3. **Adds honest overhead** for interpreter structures and a safety margin
   for per-op scratch buffers that static analysis cannot see, and labels
   the result as an estimate.
4. **Compares** against a curated board database that accounts for the RAM
   your RTOS/Wi-Fi stack already eats before your app gets any.

A *measurement mode* — running your model through the real TFLM interpreter
compiled for your host machine, for exact-to-the-byte arena numbers with
zero hardware — is the next milestone on the roadmap.

## Supported boards

ESP32, ESP32-S3, ESP32-C3, Raspberry Pi Pico (RP2040), Pico 2 (RP2350),
STM32F411 BlackPill, STM32F746 Discovery, STM32H743 Nucleo, Arduino Nano 33
BLE Sense, Teensy 4.1, Seeed XIAO ESP32S3 Sense, and the Arduino Uno (so
the tool can politely tell you *no*).

**Adding a board is a 10-line PR** to
[`boards.yaml`](src/mcufit/boards/data/boards.yaml) — contributions very
welcome.

## Roadmap

- [ ] Measurement mode: exact arena numbers via host-compiled TFLM
- [ ] ONNX model support
- [ ] Latency estimation per board
- [ ] GitHub Action (`mcufit-action`) to guard model size in CI
- [x] Web UI: [mcufit in the browser](https://avionicharshit-byte.github.io/mcufit/) — same package, running via Pyodide

## Why this exists

Pre-deployment arena estimation has been requested in the TensorFlow repos
since [2019](https://github.com/tensorflow/tensorflow/issues/35070)
([and again in 2024](https://github.com/tensorflow/tflite-micro/issues/2474))
and never shipped. Vendor tools (STM32Cube.AI, eIQ, ...) answer it only for
their own silicon. `mcufit` is the neutral, open version.

## License

MIT
