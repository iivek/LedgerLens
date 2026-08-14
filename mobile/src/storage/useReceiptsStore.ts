import { useCallback, useEffect, useState } from 'react';
import type { Receipt } from '../types';
import { addReceipt, listReceipts } from './receipts';

/** Single hook every screen reads the ledger through, so "save a receipt on
 * Scan" and "see it on Dashboard/Overview" stay in sync without a bigger
 * global-state library -- each screen re-fetches on focus via the caller
 * (see ScanScreen/DashboardScreen), not via a shared subscription, which is
 * enough for this app's scale. */
export function useReceiptsStore() {
  const [receipts, setReceipts] = useState<Receipt[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    const rows = await listReceipts();
    setReceipts(rows);
  }, []);

  useEffect(() => {
    refresh().finally(() => setLoading(false));
  }, [refresh]);

  const save = useCallback(
    async (receipt: Receipt) => {
      await addReceipt(receipt);
      await refresh();
    },
    [refresh],
  );

  return { receipts, loading, refresh, save };
}
