# LedgerLens — mobile (React Native)

Offline, on-device receipt/invoice extraction — the **active app** (Android verified
on real hardware; iOS is blocked on this project's build machine lacking full Xcode).
Supersedes [`android/`](../android/) (a Kotlin/Compose prototype, kept for reference
only). A quantized **Qwen2-VL-2B** GGUF model runs entirely on-device via
[`llama.rn`](https://github.com/mybigday/llama.rn) — no server, no `INTERNET`
permission requested anywhere in the release manifest.

## Flow

1. **Scan** — capture a receipt photo (or, in dev builds, inject a fixed test image via
   the `(dev) queue test image` button, for reliable testing without a real camera
   pointed at something useful). Capture is decoupled from extraction: tapping Capture
   enqueues the photo and the camera stays live for the next shot — a
   [`scanQueue`](src/model/scanQueue.ts) worker processes one item at a time in the
   background (one shared model context, so no real parallelism to exploit).
2. **Review card** — once extraction finishes, the queue row expands into editable
   Vendor / Date / Total / Tax fields pre-filled from the model's read, plus a category
   picker. **Category is never model-derived** — it's a fixed, user-picked list, by
   explicit product decision, not a technical limitation.
3. **Save** — writes to a local SQLite ledger (`op-sqlite`). Dashboard, Overview, and
   receipt-detail screens (with its own post-save edit mode) read from the same store.

## Prerequisites

- Node 20+, a React Native Android toolchain (JDK 17, Android SDK/NDK — see
  [the RN environment setup guide](https://reactnative.dev/docs/set-up-your-environment))
- A physical **arm64** device is strongly recommended over an emulator: on-device
  inference is CPU/GPU-bound and slow enough on an emulated CPU that testing the real
  latency story needs real hardware (see the GPU-acceleration note below)

## First-time setup

```sh
npm install
npm start                                   # Metro dev server, separate terminal
npm run android -- --device <adb-serial>    # builds, installs, launches
```

## Model files

Two GGUF files, staged manually into app-private storage (there is no
bundling/download-on-first-launch flow yet — see the TODO in
[`src/model/extract.ts`](src/model/extract.ts)):

```sh
adb shell run-as com.ledgerlensapp mkdir -p files/models
adb push model.gguf  /data/local/tmp/model.gguf
adb push mmproj.gguf /data/local/tmp/mmproj.gguf
adb shell run-as com.ledgerlensapp cp /data/local/tmp/model.gguf  files/models/model.gguf
adb shell run-as com.ledgerlensapp cp /data/local/tmp/mmproj.gguf files/models/mmproj.gguf
```

Get the GGUFs from `scripts/export_gguf.py`'s output (see the root [README](../README.md)),
or a compatible Qwen2-VL-2B-Instruct GGUF export with a matching mmproj.

## GPU acceleration — real, but chip-gated

`android/app/src/main/AndroidManifest.xml`'s `<uses-native-library android:name="libOpenCL.so">`
declaration controls whether `llama.rn` can load the OpenCL backend. **Read the comment
above that declaration before touching it** — it carries the full history of a real
regression: GPU acceleration measured **~19–31s per extraction** on verified-compatible
hardware (Adreno 750+, e.g. a Nothing Phone 3), versus **~2 minutes CPU-only fallback**
on the same phone, but the identical declaration causes a **hard native crash
(`SIGABRT`)** on unsupported GPU tiers (confirmed on a Snapdragon 778G+/Adreno 642L
device). This is a single shared manifest with no runtime chip-capability detection —
**do not enable this for a build you intend to install on hardware you haven't
personally verified**, and see [`CLAUDE.md`](../CLAUDE.md)'s mobile caveat section for
the full evidence chain, including a thermal-throttling incident that was originally
(wrongly) blamed on this same flag.

## Testing

```sh
npm test                                          # Jest unit tests
npx maestro test .maestro/01_scan_capture_categorize_save.yaml   # on-device E2E (real accuracy check)
```

See [`.maestro/README.md`](.maestro/README.md) for what each flow covers and known
flakiness (e.g. `assertVisible` racing an off-screen `ScrollView` — use
`scrollUntilVisible`, not a bare assert, for anything below the fold).

## Project layout

```
mobile/
├── src/
│   ├── model/            # extract.ts (llama.rn wiring), scanQueue.ts (background worker),
│   │                     # prompt.ts (byte-identical to src/ledgerlens/schema.py's PROMPT),
│   │                     # schema.ts (ports schema.py's parse_model_output to TS)
│   ├── screens/          # Scan, Dashboard, Overview, Settings, ReceiptDetail, ...
│   ├── storage/          # op-sqlite wrapper, receipts/settings/budget stores
│   ├── navigation/       # RootNavigator (tabs + a root stack for full-screen detail)
│   └── components/       # Primitives, DonutChart
├── android/               # native Android project (manifest, Gradle, JNI libs)
└── .maestro/              # on-device E2E flows
```
