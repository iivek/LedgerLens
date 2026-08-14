// No native binary in the Jest environment -- App.test.tsx only needs the
// schema/seed calls in App.tsx to resolve without throwing, not real storage.
module.exports = {
  open: () => ({
    execute: async () => ({ rows: [] }),
    transaction: async fn => fn({ execute: async () => ({ rows: [] }) }),
  }),
};
