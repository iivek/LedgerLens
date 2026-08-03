package com.ledgerlens.app.ui

import android.app.Application
import android.net.Uri
import android.os.SystemClock
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.ledgerlens.app.data.ExtractionRepository
import com.ledgerlens.app.data.ImageUtils
import com.ledgerlens.app.data.ModelFiles
import com.ledgerlens.app.data.Receipt
import com.ledgerlens.app.data.ReceiptParser
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed interface UiState {
    /** Model not loaded. */
    data object Idle : UiState

    data object ModelLoading : UiState

    /** Model loaded; ready to accept an image. */
    data object Ready : UiState

    /** Inference in flight; [elapsedMs] drives the on-screen latency timer. */
    data class Running(val elapsedMs: Long) : UiState

    /**
     * @param receipt parsed result, or null if the model output was not valid JSON
     *                (in which case the UI falls back to showing [raw]).
     */
    data class Result(val receipt: Receipt?, val raw: String, val elapsedMs: Long) : UiState

    data class Error(val message: String) : UiState
}

class ExtractionViewModel(application: Application) : AndroidViewModel(application) {

    private val repository = ExtractionRepository()

    private val _uiState = MutableStateFlow<UiState>(UiState.Idle)
    val uiState: StateFlow<UiState> = _uiState

    /** Normalized JPEG shown on the extraction screen. */
    var currentImagePath by mutableStateOf<String?>(null)
        private set

    private var extractionJob: Job? = null

    fun loadModel() {
        if (_uiState.value is UiState.ModelLoading || repository.isLoaded) return
        viewModelScope.launch {
            _uiState.value = UiState.ModelLoading
            try {
                val context = getApplication<Application>()
                repository.loadModel(
                    modelPath = ModelFiles.modelFile(context).absolutePath,
                    mmprojPath = ModelFiles.mmprojFile(context).absolutePath,
                )
                _uiState.value = UiState.Ready
            } catch (t: Throwable) {
                _uiState.value = UiState.Error("Model load failed: ${t.message}")
            }
        }
    }

    fun extractFromUri(uri: Uri) = startExtraction {
        ImageUtils.importImage(getApplication(), uri)
    }

    fun extractFromFile(file: File) = startExtraction {
        ImageUtils.importImage(getApplication(), file)
    }

    private fun startExtraction(prepareImage: suspend () -> File) {
        if (extractionJob?.isActive == true) return
        extractionJob = viewModelScope.launch {
            val startMs = SystemClock.elapsedRealtime()
            _uiState.value = UiState.Running(0L)

            // End-to-end latency timer: image prep + prompt eval + generation.
            val ticker = launch {
                while (isActive) {
                    _uiState.value = UiState.Running(SystemClock.elapsedRealtime() - startMs)
                    delay(100L)
                }
            }

            try {
                val imageFile = withContext(Dispatchers.IO) { prepareImage() }
                currentImagePath = imageFile.absolutePath
                val raw = repository.extract(imageFile.absolutePath)
                ticker.cancel()
                val elapsed = SystemClock.elapsedRealtime() - startMs
                _uiState.value = UiState.Result(
                    receipt = ReceiptParser.parse(raw),
                    raw = raw,
                    elapsedMs = elapsed,
                )
            } catch (t: Throwable) {
                ticker.cancel()
                _uiState.value = UiState.Error(t.message ?: "Extraction failed")
            }
        }
    }

    /** Returns to Home in the appropriate state after a result or error. */
    fun reset() {
        if (extractionJob?.isActive == true) return
        currentImagePath = null
        _uiState.value = if (repository.isLoaded) UiState.Ready else UiState.Idle
    }

    override fun onCleared() {
        repository.unloadBlocking()
    }
}
