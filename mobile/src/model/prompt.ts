// MUST stay byte-identical to `PROMPT` in src/ledgerlens/schema.py -- the
// student model is fine-tuned on that exact prompt and every harness number
// (results.csv) is measured with it, so any drift here makes on-device
// output incomparable with the measured Pareto grid. Mirrors the ChatML
// wrapper the old android/Prompts.kt used, which itself mirrors what
// `AutoProcessor.apply_chat_template` produces for Qwen2-VL.
export const MTMD_MEDIA_MARKER = '<__media__>';

const USER_PROMPT = `Extract the following fields from this receipt or invoice image and answer with a single JSON object, nothing else:
{"vendor": string|null, "date": "YYYY-MM-DD"|null, "currency": ISO-4217 string|null, "total": number|null, "tax": number|null, "line_items": [{"description": string, "qty": number, "unit_price": number, "total": number}]}
Use null for fields not present. Do not invent values.`;

export const EXTRACTION_PROMPT =
  '<|im_start|>system\n' +
  'You are a helpful assistant.<|im_end|>\n' +
  '<|im_start|>user\n' +
  `${MTMD_MEDIA_MARKER}\n` +
  `${USER_PROMPT}<|im_end|>\n` +
  '<|im_start|>assistant\n';

// src/ledgerlens/schema.py's MAX_NEW_TOKENS -- keep in sync by hand until
// there's a shared config file both Python and RN can read.
export const MAX_NEW_TOKENS = 1024;
