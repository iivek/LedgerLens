module.exports = {
  presets: ['module:@react-native/babel-preset'],
  // Must be listed last -- react-native-reanimated v4 delegates its worklets
  // transform to this separate plugin package, and it needs to run after
  // every other Babel transform in the chain.
  plugins: ['react-native-worklets/plugin'],
};
