import {
  totalSpent,
  categoryBreakdown,
  monthlyTrend,
} from '../src/utils/receiptMath';
import type { Receipt } from '../src/types';

function receipt(overrides: Partial<Receipt>): Receipt {
  return {
    id: 'r1',
    vendor: 'Test Vendor',
    date: '2026-08-01',
    currency: 'USD',
    total: 10,
    tax: 1,
    category: 'Groceries',
    lineItems: [],
    ...overrides,
  };
}

describe('totalSpent', () => {
  it('sums receipt totals', () => {
    expect(
      totalSpent([receipt({ total: 10 }), receipt({ total: 5.5 })]),
    ).toBeCloseTo(15.5);
  });

  it('returns 0 for an empty list', () => {
    expect(totalSpent([])).toBe(0);
  });
});

describe('categoryBreakdown', () => {
  it('groups by category and sorts descending by amount', () => {
    const receipts = [
      receipt({ category: 'Groceries', total: 20 }),
      receipt({ category: 'Transport', total: 60 }),
      receipt({ category: 'Groceries', total: 20 }),
    ];
    const breakdown = categoryBreakdown(receipts);
    expect(breakdown).toEqual([
      { category: 'Transport', amount: 60, pct: 0.6 },
      { category: 'Groceries', amount: 40, pct: 0.4 },
    ]);
  });

  it('returns an empty array for no receipts, without dividing by zero', () => {
    expect(categoryBreakdown([])).toEqual([]);
  });
});

describe('monthlyTrend', () => {
  it('buckets receipts into their calendar month, oldest first', () => {
    const now = new Date();
    const thisMonth = `${now.getFullYear()}-${String(
      now.getMonth() + 1,
    ).padStart(2, '0')}-05`;
    const lastMonthDate = new Date(now.getFullYear(), now.getMonth() - 1, 5);
    const lastMonth = `${lastMonthDate.getFullYear()}-${String(
      lastMonthDate.getMonth() + 1,
    ).padStart(2, '0')}-05`;

    const trend = monthlyTrend(
      [
        receipt({ date: thisMonth, total: 40 }),
        receipt({ date: lastMonth, total: 25 }),
      ],
      3,
    );
    expect(trend).toEqual([0, 25, 40]);
  });

  it('drops receipts older than the requested window', () => {
    const ancient = receipt({ date: '2000-01-01', total: 999 });
    const trend = monthlyTrend([ancient], 6);
    expect(trend.reduce((a, b) => a + b, 0)).toBe(0);
  });
});
