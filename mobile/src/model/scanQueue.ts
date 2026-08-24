import { useSyncExternalStore } from 'react';
import { cancelExtraction, extractReceipt } from './extract';
import type { RawExtraction } from './schema';

// Module-level, not React state: a bottom tab screen stays mounted across
// tab switches, but this makes "keep scanning while a queue grinds through
// in the background" true regardless of navigator config, and matches the
// singleton-model pattern extract.ts already uses for the llama context.

export interface QueueItem {
  id: string;
  path: string;
  status: 'queued' | 'processing' | 'done' | 'error';
  extraction: RawExtraction | null;
  error: string | null;
}

let items: QueueItem[] = [];
const listeners = new Set<() => void>();

function setItems(next: QueueItem[]) {
  items = next;
  for (const listener of listeners) listener();
}

export function useScanQueue(): QueueItem[] {
  return useSyncExternalStore(
    listener => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => items,
  );
}

let processing = false;

/** Sequential worker: every receipt shares one on-device model context (see
 * extract.ts), so there's no real parallelism to exploit -- one extraction
 * runs at a time regardless of how many are queued. Safe to call repeatedly;
 * a no-op while already running. */
async function processQueue() {
  if (processing) return;
  processing = true;
  try {
    let next: QueueItem | undefined;
    while ((next = items.find(i => i.status === 'queued'))) {
      const id = next.id;
      setItems(items.map(i => (i.id === id ? { ...i, status: 'processing' } : i)));
      try {
        const raw = await extractReceipt(next.path);
        setItems(
          items.map(i =>
            i.id !== id
              ? i
              : raw
              ? { ...i, status: 'done', extraction: raw }
              : { ...i, status: 'error', error: 'Could not read this receipt.' },
          ),
        );
      } catch (err) {
        console.warn('[LedgerLens] queue extraction failed:', err);
        setItems(
          items.map(i =>
            i.id === id ? { ...i, status: 'error', error: 'Could not read this receipt.' } : i,
          ),
        );
      }
    }
  } finally {
    processing = false;
  }
}

export function enqueue(path: string): void {
  const id = `scan-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  setItems([...items, { id, path, status: 'queued', extraction: null, error: null }]);
  processQueue();
}

export function retry(id: string): void {
  setItems(items.map(i => (i.id === id ? { ...i, status: 'queued', error: null } : i)));
  processQueue();
}

/** Removes an item from the queue. If it's the one currently being
 * extracted, stops the native inference too -- otherwise a discarded item
 * would keep burning CPU/battery for a result nobody will see. The worker
 * loop picks up the next queued item on its own once the in-flight
 * extraction settles, so no separate resume call is needed here. */
export function discard(id: string): void {
  const item = items.find(i => i.id === id);
  if (item?.status === 'processing') {
    cancelExtraction().catch(err =>
      console.warn('[LedgerLens] cancelExtraction failed:', err),
    );
  }
  setItems(items.filter(i => i.id !== id));
}
