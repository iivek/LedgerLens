import { getDB } from './db';

const BUDGET_KEY = 'monthlyBudget';
export const DEFAULT_BUDGET = 600;

/** Stored in app_meta (same key/value table the seed-flag uses) rather than
 * a dedicated settings table -- one persisted number doesn't earn its own
 * schema. Falls back to DEFAULT_BUDGET for a fresh install or a corrupted
 * value, never throws. */
export async function getBudget(): Promise<number> {
  const db = getDB();
  const { rows } = await db.execute('SELECT value FROM app_meta WHERE key = ?;', [BUDGET_KEY]);
  if (rows.length === 0) return DEFAULT_BUDGET;
  const n = Number(rows[0].value);
  return Number.isFinite(n) && n > 0 ? n : DEFAULT_BUDGET;
}

export async function setBudget(value: number): Promise<void> {
  const db = getDB();
  await db.execute(
    'INSERT INTO app_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value;',
    [BUDGET_KEY, String(value)],
  );
}
