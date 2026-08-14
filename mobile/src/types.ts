// Shared across storage, sample data, and screens -- kept in one place so
// storage code doesn't have to import from a "sample data" module.

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
