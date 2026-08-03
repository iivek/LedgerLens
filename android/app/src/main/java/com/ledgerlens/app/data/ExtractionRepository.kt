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
     * Best-effort synchronous unload for ViewModel.onCleared(). Skips the mutex:
     * only call when no other work can be in flight (scope already cancelled).
     */
    fun unloadBlocking() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            LlamaBridge.unload(h)
        }
    }

    private fun defaultThreadCount(): Int =
        (Runtime.getRuntime().availableProcessors() / 2).coerceIn(2, 6)

    companion object {
        /** Enough for Qwen2-VL image tokens (~750 at 1536px max dim) + prompt + JSON output. */
        private const val N_CTX = 4096
        private const val MAX_TOKENS = 1024
    }
}
