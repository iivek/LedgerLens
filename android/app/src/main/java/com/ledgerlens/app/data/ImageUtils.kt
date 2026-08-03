package com.ledgerlens.app.data

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.IOException

/**
 * Normalizes any input image (camera capture or gallery pick) into an
 * EXIF-upright, bounded-size JPEG in the app cache. The native side decodes
 * with stb_image, which ignores EXIF orientation and cannot read HEIC, so this
 * step guarantees the model always sees a plain, correctly-rotated JPEG.
 */
object ImageUtils {

    /** Max long-edge in pixels. Bounds vision-encoder cost and memory. */
    private const val MAX_DIMENSION = 1536
    private const val JPEG_QUALITY = 90

    @Throws(IOException::class)
    fun importImage(context: Context, uri: Uri): File {
        // Pass 1: bounds only, to pick a power-of-two sample size.
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, bounds)
        } ?: throw IOException("Cannot open image: $uri")

        var sampleSize = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / (sampleSize * 2) >= MAX_DIMENSION) {
            sampleSize *= 2
        }

        // Pass 2: real decode.
        val decodeOptions = BitmapFactory.Options().apply { inSampleSize = sampleSize }
        var bitmap = context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, decodeOptions)
        } ?: throw IOException("Failed to decode image: $uri")

        bitmap = applyExifRotation(context, uri, bitmap)
        bitmap = scaleToFit(bitmap, MAX_DIMENSION)

        val outFile = File(context.cacheDir, "receipt_${System.currentTimeMillis()}.jpg")
        outFile.outputStream().use { stream ->
            if (!bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, stream)) {
                throw IOException("JPEG encode failed")
            }
        }
        return outFile
    }

    @Throws(IOException::class)
    fun importImage(context: Context, file: File): File = importImage(context, Uri.fromFile(file))

    private fun applyExifRotation(context: Context, uri: Uri, bitmap: Bitmap): Bitmap {
        val orientation = try {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                ExifInterface(stream).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL,
                )
            } ?: ExifInterface.ORIENTATION_NORMAL
        } catch (_: IOException) {
            ExifInterface.ORIENTATION_NORMAL
        }

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            else -> return bitmap
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun scaleToFit(bitmap: Bitmap, maxDim: Int): Bitmap {
        val longEdge = maxOf(bitmap.width, bitmap.height)
        if (longEdge <= maxDim) return bitmap
        val scale = maxDim.toFloat() / longEdge
        return Bitmap.createScaledBitmap(
            bitmap,
            (bitmap.width * scale).toInt().coerceAtLeast(1),
            (bitmap.height * scale).toInt().coerceAtLeast(1),
            true,
        )
    }
}
