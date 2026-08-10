# LedgerLens — Android

Offline, on-device receipt/invoice extraction. A quantized **Qwen2-VL-2B** GGUF model runs
via llama.cpp's multimodal support (the `mtmd` library) entirely on the phone.

**No `INTERNET` permission.** That is deliberate and load-bearing: the manifest simply does
not declare it, so the OS guarantees the app cannot make a network request. Receipts never
leave the device — demo it in airplane mode and watch the on-screen latency timer.

## Flow

1. **Home** — photograph a receipt (CameraX) or pick one (Photo Picker).
2. **Extraction** — shows the image, a running elapsed-time counter during inference, then
   pretty-printed JSON (`vendor`, `date`, `currency`, `total`, `tax`, `line_items[]`) with
   copy-to-clipboard. If the model output is not valid JSON, the raw output is shown instead.
3. **Model setup** — model files are **not** bundled; this screen shows the expected path,
   detects the files, shows their sizes, and loads the model.

## Prerequisites

- **Android Studio** (Koala / 2024.1+) with:
  - **NDK 26.3.11579264** (pinned in `app/build.gradle.kts` — install via SDK Manager,
    or change `ndkVersion` to another r26+ LTS)
  - **CMake 3.22.1**
  - Android SDK Platform 34
- **JDK 17**
- A standalone **Gradle 8.7+** on your PATH, once, to bootstrap the wrapper (below)
- A physical **arm64-v8a** device with ~8 GB RAM recommended (the build targets arm64 only;
  emulators are impractical for on-device inference)

## First-time setup

All commands run from this `android/` directory.

### 1. Add llama.cpp as a git submodule

```sh
git submodule add --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp fetch --depth 1 origin refs/tags/b10344:refs/tags/b10344
git -C third_party/llama.cpp checkout b10344
```

> **PINNED TAG — READ THIS.** llama.cpp's public API, and the `mtmd` API in particular,
> moves fast and breaks without notice. The JNI code in
> `app/src/main/cpp/ledgerlens_jni.cpp` is currently pinned to **tag `b10344`
> (2026-08-10)** — every mtmd/llama.h symbol it calls was checked directly against
> that tag's headers (not assumed), and three real breaking changes versus an
> earlier mid-2025 draft were found and fixed this way: `mtmd_context_params`
> dropped its `verbosity` field, `mtmd_helper_bitmap_init_from_file()` now returns
> a wrapper struct by value instead of a raw pointer, and `mtmd_input_text.text_len`
> must be set explicitly (was left uninitialized — a real bug, not just drift).
> See the file's own header comment for specifics.
>
> ```
> VALIDATED AGAINST: llama.cpp tag b10344 -- BUILD CONFIRMED 2026-08-10.
> `./gradlew assembleDebug` succeeds end to end (native CMake/NDK compile +
> link + Kotlin + dex + package), producing a real app-debug.apk. Not yet
> confirmed: install + run on a physical device (needs `adb` + a connected
> phone) and actual model inference through the built JNI bridge.
> ```
>
> If you bump the submodule to a newer tag and the native build fails, see
> [Troubleshooting](#troubleshooting) — API drift is the expected failure mode; diff
> the three symbols above first, they're the ones most likely to have moved again.

### 2. Generate the Gradle wrapper

The wrapper **jar** cannot be committed by this scaffold, only
`gradle/wrapper/gradle-wrapper.properties` (already pinned to Gradle 8.7). Bootstrap it once
with any locally installed Gradle:

```sh
gradle wrapper --gradle-version 8.7
```

From then on, use `./gradlew` as usual.

### 3. Build

```sh
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

Or open `android/` in Android Studio and run. The first build compiles llama.cpp from
source (CPU backend, arm64) and takes several minutes.

## Model files

Two GGUF files are required, with **exactly these names**:

| File | Contents | Typical size |
|---|---|---|
| `ledgerlens-q4_k_m.gguf` | Qwen2-VL-2B-Instruct text model, Q4_K_M quantization | ~1.0 GB |
| `mmproj-f16.gguf` | Qwen2-VL vision projector (mmproj), F16 | ~1.4 GB |

Get them from a Qwen2-VL-2B-Instruct GGUF distribution (or convert yourself with
llama.cpp's `convert_hf_to_gguf.py` — use `--mmproj` for the projector), rename to the
filenames above, then push **after installing the app** (installing creates the
app-private directory):

```sh
adb shell mkdir -p /sdcard/Android/data/com.ledgerlens.app/files/models
adb push ledgerlens-q4_k_m.gguf /sdcard/Android/data/com.ledgerlens.app/files/models/
adb push mmproj-f16.gguf       /sdcard/Android/data/com.ledgerlens.app/files/models/
```

Then open **Model setup** in the app → **Rescan** → **Load model**. The screen shows the
exact expected directory for your device (it is the app's external files dir, so no storage
permission is needed).

> On some Android 13+ builds, `adb push` into `Android/data` of a non-debuggable app is
> restricted. The debug build is debuggable, so this works; if it still fails, use
> `adb push file /data/local/tmp/ && adb shell run-as com.ledgerlens.app ...` or a file
> manager on the device.

## Project layout

```
android/
├── app/
│   ├── build.gradle.kts               # arm64-only, CMake wiring, Compose
│   └── src/main/
│       ├── cpp/
│       │   ├── CMakeLists.txt         # add_subdirectory(third_party/llama.cpp), links llama+mtmd+common
│       │   └── ledgerlens_jni.cpp     # JNI: loadModel / extract / unload
│       └── java/com/ledgerlens/app/
│           ├── LlamaBridge.kt         # external funs + System.loadLibrary
│           ├── Prompts.kt             # fixed extraction prompt (ChatML + <__media__> marker)
│           ├── MainActivity.kt        # single activity, state-based screen switching
│           ├── data/                  # repository (Mutex-serialized), JSON models/parser, files, image prep
│           └── ui/                    # ViewModel + Home / Camera / Extraction / ModelSetup screens
├── gradle/libs.versions.toml          # version catalog
└── third_party/llama.cpp              # git submodule (you add this)
```

## Troubleshooting

### Native build fails in `ledgerlens_jni.cpp` (most likely failure)

**JNI ↔ llama.cpp/mtmd API drift is the expected way this project breaks.** The fix is
almost always mechanical: open the headers in your checked-out submodule
(`tools/mtmd/mtmd.h`, `tools/mtmd/mtmd-helper.h`, `include/llama.h`) and the reference
client `tools/mtmd/mtmd-cli.cpp`, and adjust the call sites. Every llama.cpp symbol the
JNI file uses, and its assumed shape:

| Symbol | Assumption |
|---|---|
| `mtmd_context_params_default()` / `mtmd_init_from_file(mmproj, model, params)` | params struct has `use_gpu`, `print_timings`, `n_threads`, `verbosity` |
| `mtmd_default_marker()` | returns `"<__media__>"`; must match `MTMD_MEDIA_MARKER` in `Prompts.kt` |
| `mtmd_helper_bitmap_init_from_file(ctx, path)` | takes the mtmd context first (older versions took only the path); declared in `mtmd-helper.h` |
| `mtmd_input_text` | fields `text`, `add_special`, `parse_special` |
| `mtmd_input_chunks_init()` / `mtmd_tokenize(ctx, chunks, &text, bitmaps, n)` | returns 0 on success |
| `mtmd_helper_eval_chunks(ctx, lctx, chunks, n_past, seq_id, n_batch, logits_last, &new_n_past)` | returns 0 on success |
| `mtmd-helper.h` as a separate header | helpers were split out of `mtmd.h` in 2025; on older tags include only `mtmd.h` |
| `llama_model_load_from_file` / `llama_init_from_model` / `llama_model_free` | the post-2024 names (formerly `llama_load_model_from_file` / `llama_new_context_with_model`) |
| `llama_memory_clear(llama_get_memory(ctx), true)` | the 2025 memory API (formerly `llama_kv_self_clear` / `llama_kv_cache_clear`) |
| `llama_sampler_chain_init` + `llama_sampler_init_greedy` + `llama_sampler_sample(smpl, ctx, -1)` | the sampler-chain API |
| `llama_batch_get_one(&token, 1)` | two-argument form (no pos/seq params) |
| `llama_token_to_piece(vocab, tok, buf, len, lstrip, special)` / `llama_vocab_is_eog(vocab, tok)` | vocab-first signatures |
| CMake: `mtmd` target under `tools/mtmd`, gated by `LLAMA_BUILD_TOOLS=ON` | on older tags mtmd lived in `examples/llava`; adjust `CMakeLists.txt` |

### Other common issues

- **`libmtmd`/`common` link errors** — make sure the submodule is checked out
  (`git submodule update --init`) and that a stale `.cxx/` dir isn't caching an old
  configure: `./gradlew clean` and delete `app/.cxx/`.
- **Model loads but output is garbage** — the mmproj must match the text model
  (same Qwen2-VL-2B family); also verify both files fully transferred (compare sizes on
  the Model setup screen with the originals).
- **Out of memory on load** — close other apps; the Q4_K_M model + F16 mmproj + KV cache
  need roughly 3 GB free. Use a device with 8 GB RAM.
- **Slow inference** — expected on small phones; the image is downscaled to 1536 px
  long-edge before encoding, and generation is greedy CPU decoding. Latency is the point
  of the on-screen timer, not something to hide.
- **`gradle wrapper` step skipped** — `./gradlew` will fail with "no wrapper jar"; run
  step 2 above.
