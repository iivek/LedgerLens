package com.ledgerlens.app.ui

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat

@Composable
fun HomeScreen(
    uiState: UiState,
    onOpenCamera: () -> Unit,
    onImagePicked: (Uri) -> Unit,
    onOpenModelSetup: () -> Unit,
) {
    val context = LocalContext.current
    val modelReady = uiState is UiState.Ready

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) onOpenCamera()
    }

    val photoPickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        uri?.let(onImagePicked)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text("LedgerLens", style = MaterialTheme.typography.headlineLarge)
        Text(
            "Offline receipt extraction — nothing ever leaves your device",
            style = MaterialTheme.typography.bodyMedium,
        )
        Spacer(Modifier.height(32.dp))

        when (uiState) {
            is UiState.Idle -> Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text("Model not loaded", style = MaterialTheme.typography.titleMedium)
                    Text(
                        "Copy the model files to the device, then load them in Model setup.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
            is UiState.ModelLoading -> Text("Loading model…")
            is UiState.Ready -> Text("Model loaded — ready to scan")
            else -> Unit
        }

        Spacer(Modifier.height(24.dp))

        Button(
            onClick = {
                val granted = ContextCompat.checkSelfPermission(
                    context, Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED
                if (granted) onOpenCamera()
                else cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            },
            enabled = modelReady,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Photograph receipt")
        }

        Spacer(Modifier.height(12.dp))

        OutlinedButton(
            onClick = {
                photoPickerLauncher.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                )
            },
            enabled = modelReady,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Pick from gallery")
        }

        Spacer(Modifier.height(24.dp))

        TextButton(onClick = onOpenModelSetup) {
            Text("Model setup")
        }
    }
}
