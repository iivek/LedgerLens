package com.ledgerlens.app.data

import com.ledgerlens.app.EXTRACTION_PROMPT
import com.ledgerlens.app.LlamaBridge
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Owns the native model handle and serializes all native calls:
 * exactly one load/extract/unload may run at a time.
 */
class ExtractionRepository {

    private val mutex = Mutex()

    @Volatile
    private var handle: Long = 0L

    val isLoaded: Boolean
        get() = handle != 0L

    suspend fun loadModel(modelPath: String, mmprojPath: String) {
        mutex.withLock {
            if (handle != 0L) return
            withContext(Dispatchers.Default) {
                handle = LlamaBridge.loadModel(
                    modelPath = modelPath,
                    mmprojPath = mmprojPath,
                    nCtx = N_CTX,
                    nThreads = defaultThreadCount(),
                )
            }
            check(handle != 0L) { "Native loadModel returned a null handle" }
        }
    }

    /** Runs one extraction. Throws if the model is not loaded or inference fails. */
    suspend fun extract(imagePath: String): String {
        mutex.withLock {
            val h = handle
            check(h != 0L) { "Model is not loaded" }
            return withContext(Dispatchers.Default) {
                LlamaBridge.extract(h, imagePath, EXTRACTION_PROMPT, MAX_TOKENS)
            }
        }
    }

    suspend fun unload() {
        mutex.withLock {
            val h = handle
            if (h != 0L) {
                handle = 0L
                withContext(Dispatchers.Default) { LlamaBridge.unload(h) }
            }
        }
    }

    /**
     * Fire-and-forget unload for ViewModel.onCleared(), which Android calls
     * synchronously on the main thread. Native `unload()` blocks on the same
     * C++-side mutex `extract()` holds for its entire run, so calling it
     * directly here would freeze the UI thread for as long as an in-flight
     * extraction has left to run (worst case tens of seconds per our own
     * latency measurements) -- a real ANR, not a rare edge case, if the user
     * backs out mid-inference. Runs on a plain daemon thread instead: no
     * coroutine scope survives onCleared() to launch one properly, and
     * cleanup finishing a moment after the screen is gone is harmless.
     */
    fun unloadBlocking() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            Thread({ LlamaBridge.unload(h) }, "ledgerlens-unload").apply { isDaemon = true }.start()
        }
    }

    private fun defaultThreadCount(): Int =
        (Runtime.getRuntime().availableProcessors() / 2).coerceIn(2, 6)

    companion object {
        /** Enough for Qwen2-VL image tokens (fewer now at ImageUtils' 1280px max dim than
         *  the previous 1536px) + prompt + JSON output. */
        private const val N_CTX = 4096
        /** Matches MAX_NEW_TOKENS in src/ledgerlens/schema.py -- keep both in sync. */
        private const val MAX_TOKENS = 1024
    }
}
