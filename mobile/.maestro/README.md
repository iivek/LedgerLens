# Acceptance tests (Maestro)

```sh
maestro test .maestro/                          # all flows, against whatever device is connected
maestro --device <serial> test .maestro/01_*.yaml
```

Install: https://maestro.mobile.dev (`curl -fsSL "https://get.maestro.mobile.dev" | bash`).

## Flows

- `01_scan_capture_categorize_save.yaml` -- capture, category-required gate, save, shows up on Dashboard.
- `02_retake_photo.yaml` -- retake actually returns to a capturable camera, not just a visual reset.
- `03_receipt_detail_delete_to_empty_state.yaml` -- detail view, delete confirmation, deleting
  everything lands on the real empty state (regression coverage for the seed-on-every-empty-boot bug).

## Known caveat: emulator camera capture

`01` and `02` call the real `capturePhoto()` API. On a physical device this reliably resolves in
under a second. On the Android Studio emulator's virtual camera, capture is intermittently
unreliable -- it can silently never resolve (no error, no success, `capturing` state quietly
reverts without a visible cause) rather than actually failing loudly. This is emulator/AVD camera
backend flakiness, not an app bug -- confirmed by testing the identical flow by hand against
physical hardware, where it passes every time.

If `01`/`02` hang or fail only on the emulator, don't chase it there -- run them against a real
device (`maestro --device <serial>`) instead. `03` has no camera dependency and is reliable on
either.
