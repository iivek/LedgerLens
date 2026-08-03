package com.ledgerlens.app

/**
 * Default media marker understood by llama.cpp's mtmd tokenizer
 * (`mtmd_default_marker()`). The marker is replaced by the image embedding
 * at tokenization time.
 */
const val MTMD_MEDIA_MARKER = "<__media__>"

/**
 * The user-turn text MUST stay byte-identical to `PROMPT` in
 * `src/ledgerlens/schema.py`: the student is fine-tuned on that exact prompt and
 * every harness number is measured with it, so any drift here makes on-phone
 * results incomparable with the Pareto grid. The ChatML wrapper below mirrors
 * what `AutoProcessor.apply_chat_template` produces for Qwen2-VL (default system
 * message included); the JNI layer tokenizes with `parse_special = true`.
 */
private val USER_PROMPT = """
Extract the following fields from this receipt or invoice image and answer with a single JSON object, nothing else:
{"vendor": string|null, "date": "YYYY-MM-DD"|null, "currency": ISO-4217 string|null, "total": number|null, "tax": number|null, "line_items": [{"description": string, "qty": number, "unit_price": number, "total": number}]}
Use null for fields not present. Do not invent values.
""".trimIndent()

val EXTRACTION_PROMPT: String =
    "<|im_start|>system\n" +
    "You are a helpful assistant.<|im_end|>\n" +
    "<|im_start|>user\n" +
    "$MTMD_MEDIA_MARKER\n" +
    "$USER_PROMPT<|im_end|>\n" +
    "<|im_start|>assistant\n"
