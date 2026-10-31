# Acceptance tests (Maestro)

```sh
maestro test .maestro/                          # all flows, against whatever device is connected
maestro --device <serial> test .maestro/01_*.yaml
```

Install: https://maestro.mobile.dev (`curl -fsSL "https://get.maestro.mobile.dev" | bash`).

## Flows

- `01_scan_capture_categorize_save.yaml` -- real on-device extraction accuracy test: queues a
  known SROIE test receipt and asserts the exact gold vendor name, category-required gate, save,
  shows up on Dashboard. **Requires manual one-time device setup, see below.**
- `02_capture_queue_discard.yaml` -- capturing twice in a row never blocks (no freeze-frame/retake
  step exists anymore -- see "Scan queue" below), and discarding a queued/still-extracting item
  actually removes it without crashing. Deliberately still camera-based (that's what it's
  testing); everything else uses the dev test-image button instead, see below.
- `03_receipt_detail_delete_to_empty_state.yaml` -- detail view, delete confirmation, deleting
  everything lands on the real empty state (regression coverage for the seed-on-every-empty-boot
  bug).
- `04_scan_edit_save_demo.yaml` -- full demo flow: launch, queue the test image, wait for real
  extraction, expand the row, edit fields, categorize, save, confirm on Dashboard. Same setup
  requirements as `01`. Generously timed (extraction timeout up to 900s) because it's tuned for
  running on an emulator under heavy host load, not just real hardware — see the timeout comment
  in the file itself for measured variance.
- `04b_continuation_after_extraction.yaml` -- companion to `04`, split out so a slow cold model
  load (minutes, on a loaded emulator) doesn't have to be repeated just to re-demo the fast,
  interactive expand → edit → save part. Skips `launchApp`, so it only works picking up from a
  state where an item has *already* finished extracting and is sitting in the queue.

## Scan queue

Capture (or the dev test-image button) no longer blocks on a single review screen -- it enqueues
immediately and the camera stays live for the next shot. A background worker
(`src/model/scanQueue.ts`) processes the queue strictly one item at a time (there's only one
on-device model context to share, no real parallelism), updating each row's status as it goes.
Tapping a finished (`done`) row expands it into the category-picker-then-save card; `error` rows
get a Retry; any row can be discarded via its `✕` (`testID="queue-discard-<index>"`), which also
cancels the native inference if that item happens to be the one currently extracting.

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

`01` taps the `__DEV__`-only "queue test image instead of camera" button on the Scan screen instead
of driving the real camera -- real capture depends on physical camera positioning (dark rooms,
proximity-sensor locks mid-capture) that made this flow unreliable for repeatable automated
testing. The button runs the exact same `enqueue()` code path a real capture does.

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

Also note `02`'s two `queue-discard-0` taps race the background worker (the item at index 0 may
still be `queued` or already `processing`/`done` by the time each tap lands) -- that's deliberate,
not flaky-test tolerance: discarding at any of those stages is exactly the behavior being tested.

## Known false lead: "the expand toggle is broken" (it wasn't)

If a flow's `tapOn` a finished queue row succeeds but a following `assertVisible: "Vendor"` (or
similar) fails, **check `scrollUntilVisible` before assuming the tap or the app is broken.** This
exact symptom cost hours across multiple sessions: the row's expand toggle looked broken (bare
`assertVisible` failing right after a successful tap, on both Maestro's element tap and raw
`adb shell input tap`, with the code reviewed clean and state confirmed correct via a temporary
CDP-exposed debug hook). The real cause was much simpler -- the queue card sits near the bottom
of a `ScrollView`, so the expanded content renders correctly but off-screen below the fold, and
`uiautomator dump` doesn't report off-screen content, so `assertVisible` was correctly reporting
it as not visible (it genuinely wasn't, on screen, without scrolling). `04`/`04b` use
`scrollUntilVisible` for exactly this reason -- don't revert that to a bare assert.
