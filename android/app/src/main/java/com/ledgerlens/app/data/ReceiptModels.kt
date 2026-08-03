package com.ledgerlens.app.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

@Serializable
data class LineItem(
    val description: String? = null,
    val qty: Double? = null,
    @SerialName("unit_price") val unitPrice: Double? = null,
    val total: Double? = null,
)

@Serializable
data class Receipt(
    val vendor: String? = null,
    val date: String? = null,
    val currency: String? = null,
    val total: Double? = null,
    val tax: Double? = null,
    @SerialName("line_items") val lineItems: List<LineItem> = emptyList(),
)

/**
 * Tolerant parser for the model's output. Small VLMs occasionally wrap JSON in
 * markdown fences or add stray prose despite instructions, so we extract the
 * outermost `{...}` before decoding.
 */
object ReceiptParser {

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        coerceInputValues = true
    }

    private val prettyJson = Json {
        prettyPrint = true
        encodeDefaults = true
    }

    /** @return the parsed [Receipt], or null if no valid JSON object could be extracted. */
    fun parse(raw: String): Receipt? {
        var text = raw.trim()

        // Strip markdown code fences (```json ... ``` or ``` ... ```).
        if (text.startsWith("```")) {
            text = text.substringAfter('\n', missingDelimiterValue = "")
        }
        if (text.endsWith("```")) {
            text = text.removeSuffix("```").trimEnd()
        }

        // Fall back to the outermost brace pair to shed any remaining prose.
        val start = text.indexOf('{')
        val end = text.lastIndexOf('}')
        if (start == -1 || end <= start) return null

        return try {
            json.decodeFromString<Receipt>(text.substring(start, end + 1))
        } catch (_: Exception) {
            null
        }
    }

    fun pretty(receipt: Receipt): String = prettyJson.encodeToString(receipt)
}
