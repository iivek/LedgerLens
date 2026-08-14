import React from 'react';
import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import {
  Card,
  EmptyState,
  LocalBadge,
  Money,
  SectionLabel,
} from '../components/Primitives';
import { DonutChart } from '../components/DonutChart';
import { CATEGORY_ICON, monthlyBudget } from '../constants';
import { categoryBreakdown, totalSpent } from '../utils/receiptMath';
import { useReceiptsStore } from '../storage/useReceiptsStore';
import { fonts, spacing } from '../theme/tokens';
import { useTheme } from '../theme/useTheme';
import type { RootStackParamList } from '../navigation/RootNavigator';

const LEGEND_COLORS = [
  'moss',
  'ochre',
  'mossDeep',
  'oxblood',
  'lineStrong',
] as const;

export default function DashboardScreen() {
  const { colors } = useTheme();
  const navigation =
    useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const { receipts, refresh } = useReceiptsStore();
  // Refetch whenever this tab regains focus -- a receipt saved on the Scan
  // tab wouldn't otherwise appear here until some other state change
  // happened to re-render this screen.
  useFocusEffect(
    React.useCallback(() => {
      refresh();
    }, [refresh]),
  );

  const spent = totalSpent(receipts);
  const budgetPct = monthlyBudget > 0 ? Math.min(spent / monthlyBudget, 1) : 0;
  const breakdown = categoryBreakdown(receipts);
  const recent = [...receipts].sort((a, b) => (a.date < b.date ? 1 : -1));

  return (
    <SafeAreaView
      style={{ flex: 1, backgroundColor: colors.paperRaised }}
      edges={['top']}
    >
      <View style={styles.statusStrip}>
        <LocalBadge label="On-device · airplane mode" />
      </View>
      <ScrollView contentContainerStyle={styles.body}>
        <Text
          style={[
            styles.title,
            { color: colors.ink, fontFamily: fonts.display },
          ]}
        >
          August
        </Text>
        <Text style={[styles.sub, { color: colors.inkFaint }]}>14 days in</Text>

        {receipts.length === 0 ? (
          <EmptyState
            icon="🧾"
            title="No receipts yet"
            body="Scan your first receipt to start tracking spend — everything stays on this device."
            actionLabel="Scan a receipt"
            onAction={() => navigation.navigate('Tabs', { screen: 'Scan' })}
          />
        ) : (
          <>
            <Card style={styles.budgetCard}>
              <View style={styles.budgetRow}>
                <SectionLabel>Spent of budget</SectionLabel>
                <Text
                  style={{
                    fontFamily: fonts.mono,
                    color: colors.inkSoft,
                    fontSize: 13,
                  }}
                >
                  <Money value={spent} size={15} emphasis /> / ${monthlyBudget}
                </Text>
              </View>
              <View
                style={[
                  styles.barTrack,
                  { backgroundColor: colors.paperSunken },
                ]}
              >
                <View
                  style={[
                    styles.barFill,
                    {
                      backgroundColor: colors.moss,
                      width: `${budgetPct * 100}%`,
                    },
                  ]}
                />
              </View>
            </Card>

            <View style={styles.donutRow}>
              <DonutChart
                size={84}
                strokeWidth={14}
                holeColor={colors.paperRaised}
                segments={breakdown.map((b, i) => ({
                  pct: b.pct,
                  color: colors[LEGEND_COLORS[i % LEGEND_COLORS.length]],
                }))}
              />
              <View style={styles.legend}>
                {breakdown.map((b, i) => (
                  <View key={b.category} style={styles.legendRow}>
                    <View
                      style={[
                        styles.sw,
                        {
                          backgroundColor:
                            colors[LEGEND_COLORS[i % LEGEND_COLORS.length]],
                        },
                      ]}
                    />
                    <Text
                      style={{ color: colors.inkSoft, fontSize: 11.5, flex: 1 }}
                    >
                      {b.category}
                    </Text>
                    <Text
                      style={{
                        fontFamily: fonts.mono,
                        color: colors.ink,
                        fontSize: 11.5,
                      }}
                    >
                      {(b.pct * 100).toFixed(1)}%
                    </Text>
                  </View>
                ))}
              </View>
            </View>

            <SectionLabel>Recent</SectionLabel>
            {recent.map(r => (
              <TouchableOpacity
                key={r.id}
                onPress={() =>
                  navigation.navigate('ReceiptDetail', { id: r.id })
                }
                style={[styles.txn, { borderBottomColor: colors.line }]}
              >
                <View
                  style={[
                    styles.txnIcon,
                    { backgroundColor: colors.paperSunken },
                  ]}
                >
                  <Text style={{ fontSize: 14 }}>
                    {CATEGORY_ICON[r.category]}
                  </Text>
                </View>
                <View style={{ flex: 1 }}>
                  <Text
                    style={{ color: colors.ink, fontSize: 13 }}
                    numberOfLines={1}
                  >
                    {r.vendor}
                  </Text>
                  <Text style={{ color: colors.inkFaint, fontSize: 10.5 }}>
                    {r.category} ·{' '}
                    {new Date(r.date).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                    })}
                  </Text>
                </View>
                <Money value={r.total} size={13} />
              </TouchableOpacity>
            ))}
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  statusStrip: { alignItems: 'center', paddingVertical: spacing.sm },
  body: { paddingHorizontal: spacing.md, paddingBottom: spacing.xl },
  title: { fontSize: 26, marginTop: spacing.xs },
  sub: { fontSize: 12, marginBottom: spacing.md },
  budgetCard: { marginBottom: spacing.md },
  budgetRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    marginBottom: 10,
  },
  barTrack: { height: 7, borderRadius: 4, overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: 4 },
  donutRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    marginBottom: spacing.lg,
  },
  legend: { flex: 1, gap: 6 },
  legendRow: { flexDirection: 'row', alignItems: 'center', gap: 7 },
  sw: { width: 8, height: 8, borderRadius: 2 },
  txn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 9,
    borderBottomWidth: 1,
  },
  txnIcon: {
    width: 32,
    height: 32,
    borderRadius: 9,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
