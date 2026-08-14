import type { Receipt } from '../types';

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
