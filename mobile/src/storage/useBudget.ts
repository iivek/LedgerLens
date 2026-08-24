import { useCallback, useEffect, useState } from 'react';
import { DEFAULT_BUDGET, getBudget, setBudget } from './settings';

/** Same refresh-on-focus convention as useReceiptsStore -- budget starts at
 * the default so the Dashboard progress bar has something sane to render
 * before the first DB read resolves, rather than flashing empty/zero. */
export function useBudget() {
  const [budget, setBudgetState] = useState<number>(DEFAULT_BUDGET);

  const refresh = useCallback(async () => {
    setBudgetState(await getBudget());
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const update = useCallback(
    async (value: number) => {
      await setBudget(value);
      await refresh();
    },
    [refresh],
  );

  return { budget, update, refresh };
}
