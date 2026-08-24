import { initLlama, type LlamaContext } from 'llama.rn';
import { EXTRACTION_PROMPT, MAX_NEW_TOKENS } from './prompt';
import { parseModelOutput, type RawExtraction } from './schema';

// Dev-only bridge: these files were pushed straight into the app's private
// storage via `adb push` + `run-as cp` for testing before real model
// wiring existed. A shipped build needs a proper bundling or
// download-on-first-launch flow instead of a hardcoded path -- this only
// works on this one already-provisioned device.
const MODEL_PATH = '/data/user/0/com.ledgerlensapp/files/models/model.gguf';
const MMPROJ_PATH = '/data/user/0/com.ledgerlensapp/files/models/mmproj.gguf';

let contextPromise: Promise<LlamaContext> | null = null;

/** Lazily loads the model once per app session and reuses it -- re-loading
 * a ~1GB GGUF per receipt would make every scan pay multi-second load time
 * on top of inference. */
function loadModel(): Promise<LlamaContext> {
  if (!contextPromise) {
    contextPromise = (async () => {
      // n_gpu_layers intentionally omitted: the measured Pareto numbers
      // (results/efficiency.csv) are CPU-only, and this build's GPU backend
      // support hasn't been verified -- forcing GPU offload here would run
      // an untested code path instead of the one actually benchmarked.
      const context = await initLlama({
        model: MODEL_PATH,
        n_ctx: 4096,
      });
      // Bounds match src/ledgerlens/schema.py's MIN_PIXELS/MAX_PIXELS
      // (256*28*28 .. 1280*28*28 -- Qwen2-VL's vision tokens are one per
      // 28x28 patch, so dividing by 28*28 gives the token-count equivalent).
      // Left unset, mtmd's own default picked ~4000 image tokens for a
      // single receipt photo -- 3x+ more than the model was ever trained or
      // measured on, and slow enough on phone CPU to make extraction take
      // several minutes instead of seconds.
      //
      // use_gpu: false is deliberate, not the library default (true).
      // initMultimodal() returns a boolean rather than always throwing on
      // failure -- found by direct on-device investigation (a CDP capture
      // of the app's real console output, since neither Android logcat nor
      // Metro's own terminal could surface it on this device) after 8+
      // reproducible extractions that ran to completion, at normal timing,
      // with zero errors, but returned all-null/empty fields regardless of
      // which image was queued. A /proc/<pid>/maps memory-mapping check
      // confirmed model.gguf was mapped but mmproj.gguf never was -- the
      // vision tower silently never loaded even though this call's Promise
      // resolved without throwing. Forcing CPU for multimodal specifically
      // (independent of the main LLM's own backend) avoids whatever GPU
      // registration path was failing silently for the vision tower; the
      // explicit success check below is the actual fix -- if this returns
      // false, loading is treated as a hard failure instead of proceeding
      // with a context that can't see images.
      const multimodalOk = await context.initMultimodal({
        path: MMPROJ_PATH,
        use_gpu: false,
        image_min_tokens: 256,
        image_max_tokens: 1280,
      });
      // Belt-and-braces: on-device evidence (a /proc/<pid>/maps check
      // showing mmproj.gguf never actually memory-mapped) showed
      // initMultimodal() can resolve true even when the projector didn't
      // really load, so its own return value isn't trustworthy alone --
      // cross-check with the dedicated status query too.
      const reallyEnabled = await context.isMultimodalEnabled();
      if (!multimodalOk || !reallyEnabled) {
        throw new Error(
          `initMultimodal did not actually enable vision (initMultimodal=${multimodalOk}, isMultimodalEnabled=${reallyEnabled})`,
        );
      }
      return context;
    })();
    // On failure, clear the memoized promise so the next extraction attempt
    // gets a fresh load rather than permanently replaying this rejection --
    // without this, one bad load would brick every future scan for the rest
    // of the app session.
    contextPromise.catch(() => {
      contextPromise = null;
    });
  }
  return contextPromise;
}

/** Runs the real on-device extraction prompt against a captured photo.
 * Returns null if the model produced no parseable JSON -- the caller
 * decides how to surface that (never throws for a bad-but-recoverable
 * model response, only for genuine load/inference failures). */
export async function extractReceipt(
  photoPath: string,
): Promise<RawExtraction | null> {
  const context = await loadModel();
  const result = await context.completion({
    prompt: EXTRACTION_PROMPT,
    media_paths: [photoPath],
    n_predict: MAX_NEW_TOKENS,
    temperature: 0, // deterministic extraction, not creative generation
  });
  return parseModelOutput(result.text);
}

/** Stops an in-flight extraction. On CPU-only hardware with no GPU/NPU
 * path (common -- see the on-device latency caveat in CLAUDE.md), a single
 * extraction can run for many minutes; without this, tapping Retake mid-run
 * would only reset the UI while the native inference kept burning CPU and
 * battery in the background, unseen and unstoppable. */
export async function cancelExtraction(): Promise<void> {
  if (!contextPromise) return;
  const context = await contextPromise;
  await context.stopCompletion();
}
