// Seed data: inserted into the real op-sqlite-backed store on first launch
// only (see storage/receipts.ts:seedIfEmpty), so the app never opens to a
// blank ledger, but everything after that first seed is genuinely persisted
// user data, not this static array.

import type { Receipt } from '../types';

export type { LineItem, Receipt } from '../types';

export const sampleReceipts: Receipt[] = [
  {
    id: 'r1',
    vendor: 'Blue Bottle Coffee',
    date: '2026-08-03',
    currency: 'USD',
    total: 6.75,
    tax: 0.5,
    category: 'Food & Drink',
    lineItems: [{ description: 'Oat Latte', qty: 1, unitPrice: 6.25, total: 6.25 }],
  },
  {
    id: 'r2',
    vendor: "Trader Joe's",
    date: '2026-08-05',
    currency: 'USD',
    total: 84.32,
    tax: 4.12,
    category: 'Groceries',
    lineItems: [
      { description: 'Rotisserie Chicken', qty: 1, unitPrice: 6.99, total: 6.99 },
      { description: 'Organic Whole Milk, 1gal', qty: 1, unitPrice: 4.29, total: 4.29 },
      { description: 'Sourdough Bread', qty: 1, unitPrice: 4.99, total: 4.99 },
      { description: 'Extra Virgin Olive Oil', qty: 1, unitPrice: 10.99, total: 10.99 },
      { description: 'Free-Range Eggs, dozen', qty: 1, unitPrice: 5.49, total: 5.49 },
      { description: 'Avocados', qty: 4, unitPrice: 1.24, total: 4.96 },
      { description: 'Basmati Rice, 2lb', qty: 1, unitPrice: 6.99, total: 6.99 },
      { description: 'Cheddar Cheese Block', qty: 1, unitPrice: 7.49, total: 7.49 },
      { description: 'Baby Spinach, 5oz', qty: 1, unitPrice: 3.49, total: 3.49 },
    ],
  },
  {
    id: 'r3',
    vendor: 'Ace Hardware',
    date: '2026-08-06',
    currency: 'USD',
    total: 23.1,
    tax: 1.6,
    category: 'Home',
    lineItems: [
      { description: 'Picture hooks', qty: 2, unitPrice: 3.5, total: 7.0 },
      { description: 'Wood glue', qty: 1, unitPrice: 14.5, total: 14.5 },
    ],
  },
  {
    id: 'r4',
    vendor: 'CVS Pharmacy',
    date: '2026-08-08',
    currency: 'USD',
    total: 18.44,
    tax: 0.94,
    category: 'Health',
    lineItems: [
      { description: 'Ibuprofen', qty: 1, unitPrice: 8.5, total: 8.5 },
      { description: 'Vitamins', qty: 1, unitPrice: 9.0, total: 9.0 },
    ],
  },
  {
    id: 'r5',
    vendor: 'Shell',
    date: '2026-08-09',
    currency: 'USD',
    total: 52.0,
    tax: 0,
    category: 'Transport',
    lineItems: [{ description: 'Fuel', qty: 1, unitPrice: 52.0, total: 52.0 }],
  },
];

