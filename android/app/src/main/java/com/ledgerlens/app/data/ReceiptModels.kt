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

        val jsonText = firstJsonObject(text) ?: return null
        return try {
            json.decodeFromString<Receipt>(jsonText)
        } catch (_: Exception) {
            null
        }
    }

    /**
     * Extracts the first balanced top-level `{...}` object, tracking string
     * literals so braces inside quoted values (e.g. an address field) don't
     * throw off the match, and stopping at the first balanced close rather
     * than grabbing everything up to the *last* `}` in the text (which would
     * swallow trailing prose containing its own brace, e.g. a URL fragment).
     * Mirrors `_first_json_object` in src/ledgerlens/schema.py -- the two
     * parsers should treat identical model output identically.
     */
    private fun firstJsonObject(text: String): String? {
        val start = text.indexOf('{')
        if (start == -1) return null
        var depth = 0
        var inString = false
        var escape = false
        for (i in start until text.length) {
            val c = text[i]
            if (inString) {
                when {
                    escape -> escape = false
                    c == '\\' -> escape = true
                    c == '"' -> inString = false
                }
            } else when (c) {
                '"' -> inString = true
                '{' -> depth++
                '}' -> {
                    depth--
                    if (depth == 0) return text.substring(start, i + 1)
                }
            }
        }
        return null
    }

    fun pretty(receipt: Receipt): String = prettyJson.encodeToString(receipt)
}
