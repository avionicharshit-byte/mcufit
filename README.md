# mcufit

[![PyPI](https://img.shields.io/pypi/v/mcufit)](https://pypi.org/project/mcufit/)
[![CI](https://github.com/avionicharshit-byte/mcufit/actions/workflows/ci.yml/badge.svg)](https://github.com/avionicharshit-byte/mcufit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

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
| `mcufit check model.tflite -b esp32-s3 --exact` | Same, measured by the real TFLM runtime (see Exact mode) |
| `mcufit setup-exact` | One-time build of the TFLM runtime for `--exact` |
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

## Exact mode

Static analysis is instant but approximate. For exact-to-the-byte numbers,
`mcufit` can run your model through the **real TFLite Micro interpreter
compiled for your machine** and read the recorded allocations — the same
numbers the device would report, with zero hardware:

```
mcufit setup-exact                                  # one-time build (~5 min)
mcufit check model.tflite -b esp32-s3 --exact       # measured, not estimated
```

On the person-detection reference model: static analysis estimates ~74 KB,
exact mode measures 89,248 bytes — the difference is per-operator buffers
that only the real runtime knows about. Requires git, a C++ toolchain, and
GNU make >= 3.82 (`brew install make` on macOS).

## Supported boards

31 boards across 7 vendor groups (run `mcufit boards` for the full table):

| Vendor | Boards |
|---|---|
| Arduino | Uno R3/R4, Mega 2560, Nano 33 BLE Sense, Nano 33 IoT, Portenta H7 |
| Espressif | ESP32, S2, S3, C3, C6, P4, ESP8266, ESP32-CAM, M5Stack Core2 |
| Raspberry Pi | Pico, Pico W, Pico 2 |
| STM32 | F103 Blue Pill, F411 BlackPill, F407 Discovery, F746 Discovery, H743 Nucleo |
| Seeed Studio | XIAO ESP32S3 Sense, XIAO nRF52840 Sense, Wio Terminal |
| Teensy | 4.0, 4.1 |
| Other | SparkFun Edge, BBC micro:bit v2, nRF52832 DK |

**Adding a board is a 10-line PR** to
[`boards.yaml`](src/mcufit/boards/data/boards.yaml) — contributions very
welcome.

## Roadmap

- [x] Exact mode: measured arena numbers via host-compiled TFLM (`--exact`)
- [ ] ONNX model support
- [ ] Latency estimation per board
- [ ] GitHub Action (`mcufit-action`) to guard model size in CI
- [x] Web UI: [mcufit in the browser](https://avionicharshit-byte.github.io/mcufit/) — same package, running via Pyodide

## Why this exists

Pre-deployment arena estimation has been requested in the TensorFlow repos
since [2019](https://github.com/tensorflow/tensorflow/issues/35070)
([and again in 2024](https://github.com/tensorflow/tflite-micro/issues/2474))
and never shipped. In a TFLM maintainer's
[own words](https://github.com/tensorflow/tflite-micro/issues/2474#issuecomment-2015406610)
(March 2024):

> "We don't have a python based tool for determining arena size, but we do
> have a C++ one. [...] This would be fairly easy to estimate via Python.
> However, there are additional allocations from each operator [...]"

That Python tool is what `mcufit` is — including a labelled safety margin
for exactly those per-operator allocations, until measurement mode makes
them exact. Vendor tools (STM32Cube.AI, eIQ, ...) answer the question only
for their own silicon; `mcufit` is the neutral, open version.

## License

MIT
