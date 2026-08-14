import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LocalBadge } from '../components/Primitives';
import { fonts, radius, spacing } from '../theme/tokens';
import { useTheme } from '../theme/useTheme';

const CHECKS = [
  {
    text: 'No INTERNET permission requested',
    detail:
      "Not a setting you could accidentally change — the app's release build cannot send or receive data, period.",
  },
  {
    text: 'No analytics, no crash reporting',
    detail: 'Not even anonymized — we don’t know you opened the app.',
  },
  {
    text: 'No account, no login',
    detail: 'The ledger only ever exists on this phone, unless you export it.',
  },
  {
    text: 'Model, storage, and search — all on-device',
    detail: 'Qwen2-VL-2B, quantized, runs on this phone’s own chip.',
  },
];

export default function PrivacyScreen() {
  const { colors } = useTheme();
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.paperRaised }} edges={['top']}>
      <View style={styles.statusStrip}>
        <LocalBadge label="Verifiable, not just stated" />
      </View>
      <ScrollView contentContainerStyle={styles.body}>
        <Text style={[styles.title, { color: colors.ink, fontFamily: fonts.display }]}>
          Privacy
        </Text>
        <Text style={[styles.sub, { color: colors.inkFaint }]}>
          What "local-only" actually means here
        </Text>

        <View style={[styles.zeroStat, { borderBottomColor: colors.lineStrong }]}>
          <Text style={{ fontFamily: fonts.mono, fontSize: 44, color: colors.moss, lineHeight: 48 }}>
            0
          </Text>
          <Text style={{ fontSize: 11.5, color: colors.inkFaint, marginTop: 6 }}>
            bytes sent since install
          </Text>
        </View>

        {CHECKS.map((c, i) => (
          <View
            key={c.text}
            style={[
              styles.checkRow,
              { borderBottomColor: colors.line, borderBottomWidth: i === CHECKS.length - 1 ? 0 : 1 },
            ]}>
            <View style={[styles.ck, { backgroundColor: colors.mossTint }]}>
              <Text style={{ color: colors.mossDeep, fontFamily: fonts.mono, fontSize: 11 }}>✓</Text>
            </View>
            <View style={{ flex: 1 }}>
              <Text style={{ color: colors.ink, fontSize: 12.5 }}>{c.text}</Text>
              <Text style={{ color: colors.inkFaint, fontSize: 10.5, marginTop: 2 }}>{c.detail}</Text>
            </View>
          </View>
        ))}

        <View style={[styles.verifyNote, { backgroundColor: colors.paper, borderColor: colors.lineStrong }]}>
          <Text style={{ color: colors.inkSoft, fontSize: 11, lineHeight: 17 }}>
            Don't take our word for it — check Settings → Apps → LedgerLens → Permissions on your
            phone. Network access won't be listed, because it was never requested. (One honest
            exception: a foundational Android library present in virtually every app bundles a
            read-only "is there a connection" permission that can't send or receive a single byte
            without INTERNET also being granted — which it isn't.)
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  statusStrip: { alignItems: 'center', paddingVertical: spacing.sm },
  body: { paddingHorizontal: spacing.md, paddingBottom: spacing.xl },
  title: { fontSize: 26, marginTop: spacing.xs },
  sub: { fontSize: 12, marginBottom: spacing.md },
  zeroStat: { alignItems: 'center', paddingVertical: 18, borderBottomWidth: 1, marginBottom: 6 },
  checkRow: { flexDirection: 'row', gap: 10, paddingVertical: 10 },
  ck: { width: 18, height: 18, borderRadius: 9, alignItems: 'center', justifyContent: 'center', marginTop: 1 },
  verifyNote: { marginTop: 14, padding: 12, borderRadius: radius.sm, borderWidth: 1, borderStyle: 'dashed' },
});
