/**
 * LedgerLens — private, local-first receipt & costs ledger.
 * @format
 */

import React, { useEffect, useState } from 'react';
import { ActivityIndicator, StatusBar, useColorScheme, View } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import RootNavigator from './src/navigation/RootNavigator';
import { initSchema } from './src/storage/db';
import { seedIfEmpty } from './src/storage/receipts';
import { lightColors, darkColors } from './src/theme/tokens';

function App() {
  const isDarkMode = useColorScheme() === 'dark';
  const [ready, setReady] = useState(false);

  useEffect(() => {
    initSchema()
      .then(seedIfEmpty)
      .then(() => setReady(true))
      .catch(err => {
        // Surface loudly rather than leave the app stuck on a spinner --
        // a schema/seed failure means every screen would silently show an
        // empty ledger, which reads as "you have no receipts" instead of
        // "storage is broken".
        console.error('[LedgerLens] storage init failed:', err);
        setReady(true);
      });
  }, []);

  const colors = isDarkMode ? darkColors : lightColors;

  if (!ready) {
    return (
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.paperRaised }}>
        <ActivityIndicator color={colors.moss} />
      </View>
    );
  }

  return (
    <SafeAreaProvider>
      <StatusBar barStyle={isDarkMode ? 'light-content' : 'dark-content'} />
      <NavigationContainer>
        <RootNavigator />
      </NavigationContainer>
    </SafeAreaProvider>
  );
}

export default App;
