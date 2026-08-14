import { open, type DB } from '@op-engineering/op-sqlite';

let _db: DB | null = null;

/** Lazy singleton -- op-sqlite's open() is synchronous and cheap to call
 * once, but every screen needing the DB should go through this rather than
 * opening its own handle. */
export function getDB(): DB {
  if (_db) return _db;
  _db = open({ name: 'ledgerlens.db' });
  return _db;
}

export async function initSchema(): Promise<void> {
  const db = getDB();
  // SQLite has foreign-key enforcement OFF by default even when a table
  // declares REFERENCES ... ON DELETE CASCADE -- without this pragma,
  // deleting a receipt would silently leave its line_items orphaned rather
  // than cascading, since the constraint is parsed but never enforced.
  await db.execute('PRAGMA foreign_keys = ON;');
  await db.execute(`
    CREATE TABLE IF NOT EXISTS receipts (
      id TEXT PRIMARY KEY,
      vendor TEXT NOT NULL,
      date TEXT NOT NULL,
      currency TEXT NOT NULL,
      total REAL NOT NULL,
      tax REAL NOT NULL,
      category TEXT NOT NULL
    );
  `);
  await db.execute(`
    CREATE TABLE IF NOT EXISTS line_items (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      receipt_id TEXT NOT NULL REFERENCES receipts(id) ON DELETE CASCADE,
      description TEXT NOT NULL,
      qty REAL NOT NULL,
      unit_price REAL NOT NULL,
      total REAL NOT NULL
    );
  `);
  await db.execute(`
    CREATE INDEX IF NOT EXISTS idx_line_items_receipt_id ON line_items(receipt_id);
  `);
}
