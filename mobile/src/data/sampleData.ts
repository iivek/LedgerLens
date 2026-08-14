// Placeholder data matching the design sketch exactly, until real on-device
// extraction + local storage (op-sqlite) are wired up. Kept as one module so
// it's obvious what's sample vs. real once storage lands.

export interface LineItem {
  description: string;
  qty: number;
  unitPrice: number;
  total: number;
}

export interface Receipt {
  id: string;
  vendor: string;
  date: string; // ISO
  currency: string;
  total: number;
  tax: number;
  category: string;
  lineItems: LineItem[];
}

export const CATEGORIES = [
  'Food & Drink',
  'Groceries',
  'Home',
  'Health',
  'Transport',
] as const;

export const CATEGORY_ICON: Record<string, string> = {
  'Food & Drink': '☕',
  Groceries: '🛒',
  Home: '⚒',
  Health: '✚',
  Transport: '⛽',
};

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

export const monthlyBudget = 600;

export function totalSpent(receipts: Receipt[]): number {
  return receipts.reduce((sum, r) => sum + r.total, 0);
}

export function categoryBreakdown(
  receipts: Receipt[],
): { category: string; amount: number; pct: number }[] {
  const total = totalSpent(receipts);
  const byCategory = new Map<string, number>();
  for (const r of receipts) {
    byCategory.set(r.category, (byCategory.get(r.category) ?? 0) + r.total);
  }
  return [...byCategory.entries()]
    .map(([category, amount]) => ({ category, amount, pct: total ? amount / total : 0 }))
    .sort((a, b) => b.amount - a.amount);
}
