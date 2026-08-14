import type { Receipt } from '../types';

export function totalSpent(receipts: Receipt[]): number {
  return receipts.reduce((sum, r) => sum + r.total, 0);
}

/** Monthly totals for the trailing `months` months, oldest first, current
 * month last -- so callers can read `trend[trend.length - 1]` as "this
 * month" the same way the old hardcoded illustrative array was indexed. */
export function monthlyTrend(receipts: Receipt[], months = 6): number[] {
  const now = new Date();
  const monthKeys: string[] = [];
  for (let i = months - 1; i >= 0; i--) {
    const d = new Date(now.getFullYear(), now.getMonth() - i, 1);
    monthKeys.push(
      `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`,
    );
  }
  const totals = new Array(months).fill(0);
  for (const r of receipts) {
    const idx = monthKeys.indexOf(r.date.slice(0, 7));
    if (idx !== -1) totals[idx] += r.total;
  }
  return totals;
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
    .map(([category, amount]) => ({
      category,
      amount,
      pct: total ? amount / total : 0,
    }))
    .sort((a, b) => b.amount - a.amount);
}
