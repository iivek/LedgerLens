package com.ledgerlens.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.ledgerlens.app.data.ModelFiles

@Composable
fun ModelSetupScreen(
    uiState: UiState,
    onLoadModel: () -> Unit,
    onBack: () -> Unit,
) {
    val context = LocalContext.current
    var refreshKey by remember { mutableIntStateOf(0) }
    val fileStatuses = remember(refreshKey) { ModelFiles.status(context) }
    val bothPresent = fileStatuses.all { it.exists && it.sizeBytes > 0L }
    val loading = uiState is UiState.ModelLoading
    val loaded = uiState is UiState.Ready

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
    ) {
        Text("Model setup", style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(16.dp))

        Text(
            "Model files are not bundled with the app. Copy both files into:",
            style = MaterialTheme.typography.bodyMedium,
        )
        Spacer(Modifier.height(8.dp))
        Card(Modifier.fillMaxWidth()) {
            Text(
                ModelFiles.modelsDir(context).absolutePath,
                fontFamily = FontFamily.Monospace,
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(12.dp),
            )
        }
        Spacer(Modifier.height(8.dp))
        Text(
            "e.g. adb push ${ModelFiles.MODEL_FILE_NAME} " +
                "\"${ModelFiles.modelsDir(context).absolutePath}/\"",
            fontFamily = FontFamily.Monospace,
            style = MaterialTheme.typography.bodySmall,
        )

        Spacer(Modifier.height(24.dp))
        Text("Expected files", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(8.dp))

        fileStatuses.forEach { status ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(
                    status.name,
                    fontFamily = FontFamily.Monospace,
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    if (status.exists) ModelFiles.formatSize(status.sizeBytes) else "missing",
                    color = if (status.exists) {
                        MaterialTheme.colorScheme.primary
                    } else {
                        MaterialTheme.colorScheme.error
                    },
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        Spacer(Modifier.height(24.dp))

        OutlinedButton(
            onClick = { refreshKey++ },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Rescan")
        }

        Spacer(Modifier.height(8.dp))

        Button(
            onClick = onLoadModel,
            enabled = bothPresent && !loading && !loaded,
            modifier = Modifier.fillMaxWidth(),
        ) {
            when {
                loading -> Text("Loading…")
                loaded -> Text("Model loaded")
                else -> Text("Load model")
            }
        }

        if (uiState is UiState.Error) {
            Spacer(Modifier.height(12.dp))
            Text(
                uiState.message,
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall,
            )
        }

        Spacer(Modifier.height(16.dp))
        TextButton(onClick = onBack, modifier = Modifier.fillMaxWidth()) {
            Text("Back")
        }
    }
}
