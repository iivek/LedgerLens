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

export const monthlyBudget = 600;
