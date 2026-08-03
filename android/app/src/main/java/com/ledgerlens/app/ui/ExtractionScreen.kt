package com.ledgerlens.app.ui

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.ledgerlens.app.data.ReceiptParser
import java.util.Locale

@Composable
fun ExtractionScreen(
    uiState: UiState,
    imagePath: String?,
    onDone: () -> Unit,
) {
    val clipboard = LocalClipboardManager.current
    val bitmap = remember(imagePath) {
        imagePath?.let { BitmapFactory.decodeFile(it) }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        bitmap?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = "Receipt",
                modifier = Modifier
                    .fillMaxWidth()
                    .height(220.dp),
                contentScale = ContentScale.Fit,
            )
            Spacer(Modifier.height(16.dp))
        }

        when (uiState) {
            is UiState.Running -> {
                CircularProgressIndicator()
                Spacer(Modifier.height(16.dp))
                // The latency timer: kept visible for the airplane-mode demo.
                Text(
                    formatElapsed(uiState.elapsedMs),
                    style = MaterialTheme.typography.displayMedium,
                )
                Text(
                    "Extracting on device…",
                    style = MaterialTheme.typography.bodyMedium,
                )
            }

            is UiState.Result -> {
                Text(
                    "Done in ${formatElapsed(uiState.elapsedMs)}",
                    style = MaterialTheme.typography.titleLarge,
                )
                Spacer(Modifier.height(16.dp))

                val receipt = uiState.receipt
                val displayText = if (receipt != null) {
                    ReceiptParser.pretty(receipt)
                } else {
                    uiState.raw
                }

                if (receipt == null) {
                    Text(
                        "Could not parse JSON — showing raw model output",
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Spacer(Modifier.height(8.dp))
                }

                Surface(
                    tonalElevation = 2.dp,
                    shape = MaterialTheme.shapes.medium,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(
                        displayText,
                        fontFamily = FontFamily.Monospace,
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier
                            .horizontalScroll(rememberScrollState())
                            .padding(12.dp),
                    )
                }

                Spacer(Modifier.height(16.dp))
                OutlinedButton(
                    onClick = { clipboard.setText(AnnotatedString(displayText)) },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("Copy to clipboard")
                }
                Spacer(Modifier.height(8.dp))
                Button(onClick = onDone, modifier = Modifier.fillMaxWidth()) {
                    Text("New scan")
                }
            }

            is UiState.Error -> {
                Text(
                    "Error",
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.titleLarge,
                )
                Spacer(Modifier.height(8.dp))
                Text(uiState.message, style = MaterialTheme.typography.bodyMedium)
                Spacer(Modifier.height(16.dp))
                Button(onClick = onDone, modifier = Modifier.fillMaxWidth()) {
                    Text("Back")
                }
            }

            else -> {
                // Idle / ModelLoading / Ready should not normally be visible here.
                Button(onClick = onDone, modifier = Modifier.fillMaxWidth()) {
                    Text("Back")
                }
            }
        }
    }
}

private fun formatElapsed(elapsedMs: Long): String =
    String.format(Locale.US, "%.1f s", elapsedMs / 1000.0)
