import React, { useState } from 'react';
import {
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import { Card } from '../components/Primitives';
import { useBudget } from '../storage/useBudget';
import { fonts, radius, spacing } from '../theme/tokens';
import { useTheme } from '../theme/useTheme';
import type { SettingsStackParamList } from '../navigation/RootNavigator';

type Props = NativeStackScreenProps<SettingsStackParamList, 'SettingsHome'>;

const ROWS: { label: string; screen: keyof SettingsStackParamList }[] = [
  { label: 'Privacy', screen: 'Privacy' },
  { label: 'Support LedgerLens', screen: 'Support' },
];

export default function SettingsScreen({ navigation }: Props) {
  const { colors } = useTheme();
  const { budget, update, refresh } = useBudget();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');

  useFocusEffect(
    React.useCallback(() => {
      refresh();
    }, [refresh]),
  );

  function startEdit() {
    setDraft(String(budget));
    setEditing(true);
  }

  async function saveBudget() {
    const value = Number(draft);
    if (Number.isFinite(value) && value > 0) {
      await update(value);
    }
    setEditing(false);
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.paperRaised }} edges={['top']}>
      <ScrollView contentContainerStyle={styles.body}>
        <Text style={[styles.title, { color: colors.ink, fontFamily: fonts.display }]}>
          Settings
        </Text>
        <Text style={[styles.sub, { color: colors.inkFaint }]}>
          Everything below is free — always
        </Text>

        <Card style={{ paddingVertical: 4, paddingHorizontal: spacing.md, marginBottom: spacing.md }}>
          <View style={[styles.row, { borderBottomColor: colors.line, borderBottomWidth: 1 }]}>
            <Text style={{ color: colors.ink, fontSize: 13 }}>Monthly budget</Text>
            {editing ? (
              <View style={styles.editRow}>
                <Text style={{ color: colors.inkFaint, fontSize: 13 }}>$</Text>
                <TextInput
                  value={draft}
                  onChangeText={setDraft}
                  keyboardType="number-pad"
                  autoFocus
                  style={[styles.input, { color: colors.ink, borderColor: colors.lineStrong }]}
                />
                <TouchableOpacity onPress={saveBudget}>
                  <Text style={{ color: colors.moss, fontSize: 13, fontWeight: '700' }}>
                    Save
                  </Text>
                </TouchableOpacity>
              </View>
            ) : (
              <TouchableOpacity onPress={startEdit} style={styles.editRow}>
                <Text style={{ color: colors.inkFaint, fontSize: 13 }}>${budget}</Text>
                <Text style={{ color: colors.inkFaint, fontSize: 14 }}>›</Text>
              </TouchableOpacity>
            )}
          </View>

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
  editRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  input: {
    borderWidth: 1,
    borderRadius: radius.sm,
    paddingHorizontal: 8,
    paddingVertical: 4,
    fontSize: 13,
    minWidth: 60,
    textAlign: 'right',
  },
});
