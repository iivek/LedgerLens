package com.ledgerlens.app

/**
 * JNI bridge to libledgerlens.so (llama.cpp + mtmd, statically linked).
 *
 * All functions are blocking and must be called off the main thread.
 * [ExtractionRepository][com.ledgerlens.app.data.ExtractionRepository] serializes access;
 * never call [extract] concurrently for the same handle.
 */
object LlamaBridge {

    init {
        System.loadLibrary("ledgerlens")
    }

    /**
     * Loads the text model and the multimodal projector.
     *
     * @return an opaque native handle (never 0 on success).
     * @throws RuntimeException if either file fails to load.
     */
    external fun loadModel(modelPath: String, mmprojPath: String, nCtx: Int, nThreads: Int): Long

    /**
     * Runs one full multimodal extraction: image embedding + prompt eval + greedy generation.
     *
     * @param prompt must contain [MTMD_MEDIA_MARKER] where the image should be inserted.
     * @return the raw generated text.
     * @throws RuntimeException on any native failure.
     */
    external fun extract(handle: Long, imagePath: String, prompt: String, maxTokens: Int): String

    /** Frees all native resources associated with [handle]. */
    external fun unload(handle: Long)
}
