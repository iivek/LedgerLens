module.exports = {
  preset: '@react-native/jest-preset',
  setupFiles: ['./node_modules/@shopify/react-native-skia/jestSetup.js'],
  // The preset's default pattern only transforms react-native/@react-native
  // packages -- everything else under node_modules ships pre-built ESM
  // (import/export syntax) that Jest's CommonJS-only transform can't parse.
  // These are the ESM-only deps App.tsx's import graph actually reaches.
  transformIgnorePatterns: [
    'node_modules/(?!((jest-)?react-native|@react-native(-community)?|@react-navigation|react-native-screens|react-native-safe-area-context|react-native-gesture-handler|react-native-reanimated|react-native-worklets|@shopify/react-native-skia|@op-engineering/op-sqlite|react-native-vision-camera|react-native-nitro-modules)/)',
  ],
};
