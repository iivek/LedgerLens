import type { LineItem, Receipt } from '../types';

// Mirrors src/ledgerlens/schema.py's Extraction/LineItem + parse_model_output.
// Every field is nullable because the model is explicitly told to use null
// for fields not present -- the ScanScreen UI is responsible for turning
// this into sane display defaults, not this module.

export interface RawLineItem {
  description: string | null;
  qty: number | null;
  unit_price: number | null;
  total: number | null;
}

export interface RawExtraction {
  vendor: string | null;
  date: string | null;
  currency: string | null;
  total: number | null;
  tax: number | null;
  line_items: RawLineItem[];
}

const FENCE_RE = /^```(?:json)?\s*|\s*```$/gm;

/** Take the first balanced top-level {...} object, ignoring braces inside
 * string literals -- mirrors schema.py's _first_json_object exactly, since
 * models sometimes wrap the JSON in prose ("Here's the extraction: {...}"). */
function firstJsonObject(text: string): string | null {
  const start = text.indexOf('{');
  if (start === -1) return null;
  let depth = 0;
  let inStr = false;
  let escape = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (inStr) {
      if (escape) escape = false;
      else if (c === '\\') escape = true;
      else if (c === '"') inStr = false;
    } else if (c === '"') {
      inStr = true;
    } else if (c === '{') {
      depth++;
    } else if (c === '}') {
      depth--;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}

function asStringOrNull(v: unknown): string | null {
  return typeof v === 'string' ? v : null;
}

function asNumberOrNull(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) {
    return Number(v);
  }
  return null;
}

function toLineItem(v: unknown): RawLineItem {
  const o = v && typeof v === 'object' ? (v as Record<string, unknown>) : {};
  return {
    description: asStringOrNull(o.description),
    qty: asNumberOrNull(o.qty),
    unit_price: asNumberOrNull(o.unit_price),
    total: asNumberOrNull(o.total),
  };
}

/** Parse raw model text into a RawExtraction, or null if the model didn't
 * return valid, well-shaped JSON -- callers should treat null as "extraction
 * failed", not throw it away silently. */
export function parseModelOutput(text: string): RawExtraction | null {
  if (!text) return null;
  const cleaned = text.trim().replace(FENCE_RE, '').trim();
  const candidate = firstJsonObject(cleaned);
  if (candidate === null) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(candidate);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== 'object') return null;
  const o = parsed as Record<string, unknown>;

  return {
    vendor: asStringOrNull(o.vendor),
    date: asStringOrNull(o.date),
    currency: asStringOrNull(o.currency),
    total: asNumberOrNull(o.total),
    tax: asNumberOrNull(o.tax),
    line_items: Array.isArray(o.line_items) ? o.line_items.map(toLineItem) : [],
  };
}

/** Turns a (possibly mostly-null) model extraction into something the
 * ledger UI can always render -- the model is told to use null for absent
 * fields, but a receipt row on screen needs a real vendor string and a real
 * total, not undefined. Category is deliberately excluded: that's always
 * the user's own tap, never derived from the extraction. */
export function toReceiptFields(
  raw: RawExtraction,
): Omit<Receipt, 'id' | 'category'> {
  const lineItems: LineItem[] = raw.line_items.map(li => ({
    description: li.description ?? 'Item',
    qty: li.qty ?? 1,
    unitPrice: li.unit_price ?? li.total ?? 0,
    total: li.total ?? 0,
  }));
  const lineItemsSum = lineItems.reduce((sum, li) => sum + li.total, 0);

  return {
    vendor: raw.vendor ?? 'Unknown vendor',
    date: raw.date ?? new Date().toISOString().slice(0, 10),
    currency: raw.currency ?? 'USD',
    total: raw.total ?? lineItemsSum,
    tax: raw.tax ?? 0,
    lineItems,
  };
}
