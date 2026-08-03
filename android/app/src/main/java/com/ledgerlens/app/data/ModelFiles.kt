package com.ledgerlens.app.data

import android.content.Context
import java.io.File
import java.util.Locale

/**
 * Model files are NOT bundled with the APK. The user pushes them (e.g. via adb)
 * into the app's external files dir, which requires no extra permissions:
 *
 *   /sdcard/Android/data/com.ledgerlens.app/files/models/ledgerlens-q4_k_m.gguf
 *   /sdcard/Android/data/com.ledgerlens.app/files/models/mmproj-f16.gguf
 */
object ModelFiles {

    const val MODEL_FILE_NAME = "ledgerlens-q4_k_m.gguf"
    const val MMPROJ_FILE_NAME = "mmproj-f16.gguf"

    fun modelsDir(context: Context): File =
        File(context.getExternalFilesDir(null), "models")

    fun modelFile(context: Context): File = File(modelsDir(context), MODEL_FILE_NAME)

    fun mmprojFile(context: Context): File = File(modelsDir(context), MMPROJ_FILE_NAME)

    fun bothPresent(context: Context): Boolean =
        modelFile(context).length() > 0L && mmprojFile(context).length() > 0L

    data class FileStatus(val name: String, val exists: Boolean, val sizeBytes: Long)

    fun status(context: Context): List<FileStatus> =
        listOf(modelFile(context), mmprojFile(context)).map { file ->
            FileStatus(name = file.name, exists = file.isFile, sizeBytes = file.length())
        }

    fun formatSize(bytes: Long): String = when {
        bytes >= 1L shl 30 -> String.format(Locale.US, "%.2f GB", bytes.toDouble() / (1L shl 30))
        bytes >= 1L shl 20 -> String.format(Locale.US, "%.1f MB", bytes.toDouble() / (1L shl 20))
        bytes >= 1L shl 10 -> String.format(Locale.US, "%.1f KB", bytes.toDouble() / (1L shl 10))
        else -> "$bytes B"
    }
}
