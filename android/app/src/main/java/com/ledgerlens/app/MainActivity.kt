package com.ledgerlens.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.ledgerlens.app.ui.CameraScreen
import com.ledgerlens.app.ui.ExtractionScreen
import com.ledgerlens.app.ui.ExtractionViewModel
import com.ledgerlens.app.ui.HomeScreen
import com.ledgerlens.app.ui.ModelSetupScreen

/** Simple state-based screen switching — deliberately no navigation library. */
private enum class Screen { Home, Camera, Extraction, ModelSetup }

class MainActivity : ComponentActivity() {

    private val viewModel: ExtractionViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface {
                    LedgerLensApp(viewModel)
                }
            }
        }
    }
}

@Composable
private fun LedgerLensApp(viewModel: ExtractionViewModel) {
    var screen by remember { mutableStateOf(Screen.Home) }
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    when (screen) {
        Screen.Home -> HomeScreen(
            uiState = uiState,
            onOpenCamera = { screen = Screen.Camera },
            onImagePicked = { uri ->
                viewModel.extractFromUri(uri)
                screen = Screen.Extraction
            },
            onOpenModelSetup = { screen = Screen.ModelSetup },
        )

        Screen.Camera -> CameraScreen(
            onCaptured = { file ->
                viewModel.extractFromFile(file)
                screen = Screen.Extraction
            },
            onClose = { screen = Screen.Home },
        )

        Screen.Extraction -> ExtractionScreen(
            uiState = uiState,
            imagePath = viewModel.currentImagePath,
            onDone = {
                viewModel.reset()
                screen = Screen.Home
            },
        )

        Screen.ModelSetup -> ModelSetupScreen(
            uiState = uiState,
            onLoadModel = { viewModel.loadModel() },
            onBack = { screen = Screen.Home },
        )
    }
}
