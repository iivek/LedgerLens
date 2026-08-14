import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { Canvas, Path, Skia } from '@shopify/react-native-skia';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card, LocalBadge, Money, SectionLabel } from '../components/Primitives';
import { categoryBreakdown, sampleReceipts } from '../data/sampleData';
import { fonts, spacing } from '../theme/tokens';
import { useTheme } from '../theme/useTheme';

const LEGEND_COLORS = ['moss', 'ochre', 'mossDeep', 'oxblood', 'lineStrong'] as const;

// Illustrative six-month trend around the current sample total -- becomes a
// real query over stored receipts once op-sqlite is wired up.
const TREND = [1620, 1740, 1690, 1960, 1802, 1842];

function Sparkline({ values, color }: { values: number[]; color: string }) {
  const w = 240;
  const h = 46;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = values.map((v, i) => ({
    x: (i / (values.length - 1)) * w,
    y: h - ((v - min) / range) * h,
  }));
  const pathBuilder = Skia.PathBuilder.Make();
  points.forEach((p, i) => (i === 0 ? pathBuilder.moveTo(p.x, p.y) : pathBuilder.lineTo(p.x, p.y)));
  const path = pathBuilder.build();
  const last = points[points.length - 1];
  const dot = Skia.PathBuilder.Make().addCircle(last.x, last.y, 3.5).build();
  return (
    <Canvas style={{ width: w, height: h }}>
      <Path path={path} style="stroke" strokeWidth={2.5} strokeCap="round" strokeJoin="round" color={color} />
      <Path path={dot} style="fill" color={color} />
    </Canvas>
  );
}

export default function OverviewScreen() {
  const { colors } = useTheme();
  const breakdown = categoryBreakdown(sampleReceipts);
  const delta = ((TREND[TREND.length - 1] - TREND[TREND.length - 2]) / TREND[TREND.length - 2]) * 100;
  const vendorTotals = new Map<string, { amount: number; visits: number }>();
  for (const r of sampleReceipts) {
    const cur = vendorTotals.get(r.vendor) ?? { amount: 0, visits: 0 };
    vendorTotals.set(r.vendor, { amount: cur.amount + r.total, visits: cur.visits + 1 });
  }
  const topVendors = [...vendorTotals.entries()].sort((a, b) => b[1].amount - a[1].amount);

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.paperRaised }} edges={['top']}>
      <View style={styles.statusStrip}>
        <LocalBadge label="0 requests sent" />
      </View>
      <ScrollView contentContainerStyle={styles.body}>
        <Text style={[styles.title, { color: colors.ink, fontFamily: fonts.display }]}>
          Overview
        </Text>
        <Text style={[styles.sub, { color: colors.inkFaint }]}>Last 6 months</Text>

        <Card style={styles.trendCard}>
          <View style={styles.trendHead}>
            <Money value={TREND[TREND.length - 1]} size={17} emphasis />
            {/* Semantic, not decorative: spending down is the "good" green;
                spending up gets a distinct warning tone, not the same
                reassuring color a decrease would use. */}
            <Text
              style={{
                fontFamily: fonts.mono,
                fontSize: 11,
                color: delta <= 0 ? colors.moss : colors.ochre,
              }}>
              {delta <= 0 ? '↓' : '↑'} {Math.abs(delta).toFixed(0)}% vs. last month
            </Text>
          </View>
          <Sparkline values={TREND} color={colors.moss} />
        </Card>

        <SectionLabel>By category</SectionLabel>
        <View style={{ gap: 12, marginBottom: spacing.lg }}>
          {breakdown.map((b, i) => (
            <View key={b.category} style={{ gap: 5 }}>
              <View style={styles.catTop}>
                <Text style={{ color: colors.inkSoft, fontSize: 12 }}>{b.category}</Text>
                <Money value={b.amount} size={12} />
              </View>
              <View style={[styles.catTrack, { backgroundColor: colors.paperSunken }]}>
                <View
                  style={[
                    styles.catFill,
                    {
                      width: `${(b.amount / breakdown[0].amount) * 100}%`,
                      backgroundColor: colors[LEGEND_COLORS[i % LEGEND_COLORS.length]],
                    },
                  ]}
                />
              </View>
            </View>
          ))}
        </View>

        <SectionLabel>Top vendors</SectionLabel>
        <Card style={{ paddingVertical: 4, paddingHorizontal: spacing.md }}>
          {topVendors.map(([vendor, v], i) => (
            <View
              key={vendor}
              style={[
                styles.vrow,
                { borderBottomColor: colors.line, borderBottomWidth: i === topVendors.length - 1 ? 0 : 1 },
              ]}>
              <View>
                <Text style={{ color: colors.ink, fontSize: 12.5 }}>{vendor}</Text>
                <Text style={{ color: colors.inkFaint, fontSize: 10.5 }}>
                  {v.visits} visit{v.visits === 1 ? '' : 's'}
                </Text>
              </View>
              <Money value={v.amount} size={12.5} />
            </View>
          ))}
        </Card>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  statusStrip: { alignItems: 'center', paddingVertical: spacing.sm },
  body: { paddingHorizontal: spacing.md, paddingBottom: spacing.xl },
  title: { fontSize: 26, marginTop: spacing.xs },
  sub: { fontSize: 12, marginBottom: spacing.md },
  trendCard: { marginBottom: spacing.lg },
  trendHead: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 },
  catTop: { flexDirection: 'row', justifyContent: 'space-between' },
  catTrack: { height: 8, borderRadius: 4, overflow: 'hidden' },
  catFill: { height: '100%', borderRadius: 4 },
  vrow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 10 },
});
