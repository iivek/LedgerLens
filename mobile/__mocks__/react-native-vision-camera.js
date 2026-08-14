// Jest has no native camera binary -- App.test.tsx only needs ScanScreen to
// mount without throwing, not a working camera feed.
module.exports = {
  Camera: () => null,
  useCameraDevice: () => undefined,
  useCameraPermission: () => ({
    hasPermission: false,
    requestPermission: async () => false,
  }),
  usePhotoOutput: () => ({
    capturePhoto: async () => ({
      saveToTemporaryFileAsync: async () => '',
      dispose: () => {},
    }),
  }),
};
