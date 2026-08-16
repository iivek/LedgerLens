# Acceptance tests (Maestro)

```sh
maestro test .maestro/                          # all flows, against whatever device is connected
maestro --device <serial> test .maestro/01_*.yaml
```

Install: https://maestro.mobile.dev (`curl -fsSL "https://get.maestro.mobile.dev" | bash`).

## Flows

- `01_scan_capture_categorize_save.yaml` -- real on-device extraction accuracy test: extracts a
  known SROIE test receipt and asserts the exact gold vendor name, category-required gate, save,
  shows up on Dashboard. **Requires manual one-time device setup, see below.**
- `02_retake_photo.yaml` -- retake actually returns to a capturable *camera*, not just a visual
  reset. Deliberately still camera-based (that's what it's testing); everything else uses the
  dev test-image button instead, see below.
- `03_receipt_detail_delete_to_empty_state.yaml` -- detail view, delete confirmation, deleting
  everything lands on the real empty state (regression coverage for the seed-on-every-empty-boot
  bug).

## Setup required for `01`

`01` drives real on-device model inference, not a stub -- it needs two things pushed onto the
device first, which `clearState` would wipe, so `01` deliberately doesn't use it:

```sh
# 1. Model + vision projector, into the app's private files/models/ dir
adb push models/gguf/sft2b-teacher-agree-q4_k_m.gguf /data/local/tmp/model.gguf
adb push models/gguf/mmproj-sft2b-teacher-agree-f16.gguf /data/local/tmp/mmproj.gguf
adb shell run-as com.ledgerlensapp mkdir -p /data/data/com.ledgerlensapp/files/models
adb shell run-as com.ledgerlensapp cp /data/local/tmp/model.gguf /data/data/com.ledgerlensapp/files/models/model.gguf
adb shell run-as com.ledgerlensapp cp /data/local/tmp/mmproj.gguf /data/data/com.ledgerlensapp/files/models/mmproj.gguf

# 2. A real receipt photo with known gold labels, as the fixed test image
adb push data/processed/images/sroie_test/sroie_test_X00016469670.jpg /data/local/tmp/test-receipt.jpg
adb shell run-as com.ledgerlensapp cp /data/local/tmp/test-receipt.jpg /data/data/com.ledgerlensapp/files/test-receipt.jpg
```

`01` taps the `__DEV__`-only "use test image instead of camera" button on the Scan screen instead
of driving the real camera -- real capture depends on physical camera positioning (dark rooms,
proximity-sensor locks mid-capture) that made this flow unreliable for repeatable automated
testing. The button runs the exact same `runExtraction()` code path a real capture does.

## Known caveat: on-device extraction speed is hardware-gated

`01` asserts a 120s timeout waiting for extraction to finish -- deliberately generous, because
this genuinely varies by an order of magnitude depending on the device's GPU/NPU support. Real
measurements from this project, both against the identical SROIE test image:

- **Nothing Phone (3)** (Snapdragon 8s Gen 4, Adreno 825, GPU+Hexagon path both engaged): **18.8s**
  total, every scored field (vendor/date/total) an exact match against gold.
- **Honor phone** (Snapdragon 778G+, Adreno 642L, below llama.cpp's supported GPU tier): **11.3
  minutes**, CPU-only fallback, same accuracy.

See CLAUDE.md's on-device latency caveat for the full breakdown and why. If `01` times out, check
which device it's running against before assuming it's broken.

## Known caveat: emulator camera capture

`02` calls the real `capturePhoto()` API. On a physical device this reliably resolves in under a
second. On the Android Studio emulator's virtual camera, capture is intermittently unreliable --
it can silently never resolve (no error, no success, `capturing` state quietly reverts without a
visible cause) rather than actually failing loudly. This is emulator/AVD camera backend flakiness,
not an app bug -- confirmed by testing the identical flow by hand against physical hardware, where
it passes every time.

If `02` hangs or fails only on the emulator, don't chase it there -- run it against a real device
(`maestro --device <serial>`) instead. `03` has no camera dependency and is reliable on either.
