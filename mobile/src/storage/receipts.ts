import type { LineItem, Receipt } from '../types';
import { sampleReceipts } from '../data/sampleData';
import { getDB } from './db';

export async function listReceipts(): Promise<Receipt[]> {
  const db = getDB();
  const receiptRows = (await db.execute('SELECT * FROM receipts ORDER BY date DESC;')).rows;
  const itemRows = (await db.execute('SELECT * FROM line_items;')).rows;

  const itemsByReceipt = new Map<string, LineItem[]>();
  for (const row of itemRows) {
    const receiptId = row.receipt_id as string;
    const list = itemsByReceipt.get(receiptId) ?? [];
    list.push({
      description: row.description as string,
      qty: row.qty as number,
      unitPrice: row.unit_price as number,
      total: row.total as number,
    });
    itemsByReceipt.set(receiptId, list);
  }

  return receiptRows.map(row => ({
    id: row.id as string,
    vendor: row.vendor as string,
    date: row.date as string,
    currency: row.currency as string,
    total: row.total as number,
    tax: row.tax as number,
    category: row.category as string,
    lineItems: itemsByReceipt.get(row.id as string) ?? [],
  }));
}

export async function addReceipt(receipt: Receipt): Promise<void> {
  const db = getDB();
  // Wrapped in a transaction: a receipt with only some of its line items
  // saved (e.g. the app is killed mid-insert) is worse than not saving it
  // at all -- transaction.commit() is implicit on the callback returning
  // without throwing, per op-sqlite's transaction() contract.
  await db.transaction(async tx => {
    await tx.execute(
      'INSERT INTO receipts (id, vendor, date, currency, total, tax, category) VALUES (?, ?, ?, ?, ?, ?, ?);',
      [receipt.id, receipt.vendor, receipt.date, receipt.currency, receipt.total, receipt.tax, receipt.category],
    );
    for (const li of receipt.lineItems) {
      await tx.execute(
        'INSERT INTO line_items (receipt_id, description, qty, unit_price, total) VALUES (?, ?, ?, ?, ?);',
        [receipt.id, li.description, li.qty, li.unitPrice, li.total],
      );
    }
  });
}

export async function deleteReceipt(id: string): Promise<void> {
  const db = getDB();
  await db.execute('DELETE FROM receipts WHERE id = ?;', [id]);
}

/** First-launch only: an empty ledger reads as "is this broken" rather than
 * "you haven't scanned anything yet" -- seed it once so the app never opens
 * to a blank state, without ever overwriting real user data on later
 * launches (only runs when the table is genuinely empty). */
export async function seedIfEmpty(): Promise<void> {
  const db = getDB();
  const { rows } = await db.execute('SELECT COUNT(*) as n FROM receipts;');
  const count = (rows[0]?.n as number) ?? 0;
  if (count > 0) return;
  for (const receipt of sampleReceipts) {
    await addReceipt(receipt);
  }
}
