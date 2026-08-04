<div align="center">

# 📟 mcufit

**Know if your AI model fits your microcontroller - before you flash it.**

[![PyPI](https://img.shields.io/pypi/v/mcufit)](https://pypi.org/project/mcufit/)
[![CI](https://github.com/avionicharshit-byte/mcufit/actions/workflows/ci.yml/badge.svg)](https://github.com/avionicharshit-byte/mcufit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/mcufit)](https://pypi.org/project/mcufit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

### [🌐 **Try it in your browser - drop a model, get a verdict, nothing to install**](https://avionicharshit-byte.github.io/mcufit/)

*Runs entirely client-side. Your model never leaves your device.*

<br>

<img src="docs/demo.svg" alt="mcufit demo - ✅ fits on ESP32-S3, ❌ won't fit on Arduino Uno" width="720">

</div>

---

Deploying ML to a microcontroller today works like this: train, convert,
flash, watch it crash with `Arena size is too small`, guess a bigger
number, re-flash, repeat. The official TFLite Micro docs literally say the
memory size *"may need to be determined by experimentation."*

**mcufit replaces the experimentation with an answer in one second.**

```bash
pip install mcufit
mcufit check model.tflite --board esp32-s3
```

## Features

- ⚡ **Instant fit verdict** - RAM & flash bars, headroom, and the exact layer where memory peaks
- 🎯 **Exact mode** - runs your model through the *real* TFLite Micro runtime compiled for your machine: measured bytes, not estimates, zero hardware
- 🌐 **Browser version** - same engine via WebAssembly, fully private, no install
- 🔌 **31 boards** - ESP32 family, Pico, STM32, Teensy, Arduino, and more
- 🤖 **CI guard** - a GitHub Action that fails the PR when your model outgrows the chip
- 💡 **Actionable suggestions** - int8 quantization preview (simulated, not guessed) and which boards *do* fit
- ⏱️ **Speed ballpark** - rough ms/inference per board from the model's MAC count
- 📦 **ONNX support** - `pip install 'mcufit[onnx]'` for the PyTorch world

## Commands

| Command | What you get |
|---|---|
| `mcufit check model.tflite -b esp32-s3` | Fit verdict (exit code 1 on ❌ - CI-friendly) |
| `mcufit check ... --exact` | Measured by the real TFLM runtime |
| `mcufit compare model.tflite` | Verdict matrix across all 31 boards |
| `mcufit inspect model.tflite` | Layer-by-layer memory profile |
| `mcufit boards` | The board database |
| `mcufit setup-exact` | One-time build for exact mode (~5 min) |
| `... --json` | Machine-readable output for scripts & CI |

## Exact mode

Static analysis is instant but approximate - real runtimes allocate
per-operator working memory no file analysis can see. Exact mode compiles
the actual TFLite Micro interpreter for your machine and reads its
recorded allocations:

```bash
mcufit setup-exact                              # one-time (needs git, C++, gmake)
mcufit check model.tflite -b esp32-s3 --exact   # measured, byte-accurate
```

On the person-detection reference model: estimate ~74 KB → measured
**89,248 bytes**. The website does the same trick with TFLM compiled to
WebAssembly - verdicts flip from "estimate" to "measured" right in the page.

## Guard your model in CI

```yaml
- uses: avionicharshit-byte/mcufit@main
  with:
    model: models/wake_word.tflite
    board: esp32-s3
```

A model that grows past the board's RAM now fails the pull request instead
of the field deployment.

<details>
<summary><b>Supported boards</b> (31 across 7 vendors - click to expand)</summary>
<br>

| Vendor | Boards |
|---|---|
| Arduino | Uno R3/R4, Mega 2560, Nano 33 BLE Sense, Nano 33 IoT, Portenta H7 |
| Espressif | ESP32, S2, S3, C3, C6, P4, ESP8266, ESP32-CAM, M5Stack Core2 |
| Raspberry Pi | Pico, Pico W, Pico 2 |
| STM32 | F103 Blue Pill, F411 BlackPill, F407 Discovery, F746 Discovery, H743 Nucleo |
| Seeed Studio | XIAO ESP32S3 Sense, XIAO nRF52840 Sense, Wio Terminal |
| Teensy | 4.0, 4.1 |
| Other | SparkFun Edge, BBC micro:bit v2, nRF52832 DK |

Missing yours? **Adding a board is a 10-line PR** to
[`boards.yaml`](src/mcufit/boards/data/boards.yaml) - CI validates it
automatically.

</details>

<details>
<summary><b>How it works</b></summary>
<br>

The RAM bottleneck on microcontrollers is the **tensor arena**: every
intermediate tensor alive at the same moment must fit in SRAM at once.
mcufit parses the model file directly (weights → flash, activations →
RAM), computes tensor lifetimes across the execution schedule, and finds
the peak - the same quantity TFLM's memory planner must pack. Exact mode
skips the math and asks the real runtime. Board verdicts account for the
RAM your RTOS/Wi-Fi stack already eats before your app gets any.

</details>

## Why this exists

Pre-deployment memory estimation has been requested in the TensorFlow
repos since [2019](https://github.com/tensorflow/tensorflow/issues/35070) -
never shipped. A TFLite Micro maintainer,
[March 2024](https://github.com/tensorflow/tflite-micro/issues/2474#issuecomment-2015406610):

> *"We don't have a python based tool for determining arena size…"*

Vendor tools (STM32Cube.AI, eIQ) answer only for their own silicon.
**mcufit is the neutral, open version.**

## License

MIT
