import React from 'react';
import { ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { Card } from '../components/Primitives';
import { fonts, spacing } from '../theme/tokens';
import { useTheme } from '../theme/useTheme';
import type { SettingsStackParamList } from '../navigation/RootNavigator';

type Props = NativeStackScreenProps<SettingsStackParamList, 'SettingsHome'>;

const ROWS: { label: string; screen: keyof SettingsStackParamList }[] = [
  { label: 'Privacy', screen: 'Privacy' },
  { label: 'Support LedgerLens', screen: 'Support' },
];

export default function SettingsScreen({ navigation }: Props) {
  const { colors } = useTheme();
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.paperRaised }} edges={['top']}>
      <ScrollView contentContainerStyle={styles.body}>
        <Text style={[styles.title, { color: colors.ink, fontFamily: fonts.display }]}>
          Settings
        </Text>
        <Text style={[styles.sub, { color: colors.inkFaint }]}>
          Everything below is free — always
        </Text>
        <Card style={{ paddingVertical: 4, paddingHorizontal: spacing.md }}>
          {ROWS.map((row, i) => (
            <TouchableOpacity
              key={row.screen}
              onPress={() => navigation.navigate(row.screen as 'Privacy')}
              style={[
                styles.row,
                { borderBottomColor: colors.line, borderBottomWidth: i === ROWS.length - 1 ? 0 : 1 },
              ]}>
              <Text style={{ color: colors.ink, fontSize: 13 }}>{row.label}</Text>
              <Text style={{ color: colors.inkFaint, fontSize: 14 }}>›</Text>
            </TouchableOpacity>
          ))}
        </Card>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  body: { paddingHorizontal: spacing.md, paddingTop: spacing.md, paddingBottom: spacing.xl },
  title: { fontSize: 26 },
  sub: { fontSize: 12, marginBottom: spacing.md },
  row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 14 },
});
