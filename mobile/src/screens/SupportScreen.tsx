import React from 'react';
import { Linking, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { fonts, radius, spacing } from '../theme/tokens';
import { useTheme } from '../theme/useTheme';

// Deliberately NOT an embedded IAP SDK: react-native-iap's underlying
// openiap-google library declares INTERNET unconditionally in its own
// manifest (confirmed via Gradle's manifest-merger-blame report), which
// would compromise the app's core "zero network permission" claim for a
// feature that gates nothing. Opening the user's own browser is an
// explicit, user-initiated action that never touches this app's process.
const SUPPORT_URL = 'https://github.com/sponsors/PLACEHOLDER';

const POINTS = [
  'Every feature is free, no exceptions',
  'Nothing is gated behind this',
  'Source is public — verify the privacy claims yourself',
];

export default function SupportScreen() {
  const { colors } = useTheme();
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.paperRaised }} edges={['top']}>
      <ScrollView contentContainerStyle={styles.body}>
        <Text style={[styles.title, { color: colors.ink, fontFamily: fonts.display }]}>
          Support LedgerLens
        </Text>
        <Text style={[styles.sub, { color: colors.inkFaint }]}>
          Free and open source. Chip in if you'd like.
        </Text>

        <View style={{ gap: 10, marginBottom: spacing.lg }}>
          {POINTS.map(p => (
            <View key={p} style={styles.pointRow}>
              <Text style={{ color: colors.moss, fontFamily: fonts.mono, fontSize: 12 }}>✓</Text>
              <Text style={{ color: colors.ink, fontSize: 12.5, flex: 1 }}>{p}</Text>
            </View>
          ))}
        </View>

        <View style={[styles.priceRow, { borderTopColor: colors.line }]}>
          <Text style={{ color: colors.inkSoft, fontSize: 12 }}>Suggested, entirely optional</Text>
          <Text style={{ fontFamily: fonts.mono, fontSize: 20, color: colors.ink }}>
            $3 <Text style={{ fontSize: 11, color: colors.inkFaint, fontFamily: fonts.body }}>one-time</Text>
          </Text>
        </View>

        <TouchableOpacity
          style={[styles.tipBtn, { backgroundColor: colors.moss }]}
          onPress={() => Linking.openURL(SUPPORT_URL)}>
          <Text style={{ color: colors.paperRaised, fontWeight: '700', fontSize: 14 }}>
            Send a tip
          </Text>
        </TouchableOpacity>
        <Text style={{ color: colors.inkFaint, fontSize: 10, textAlign: 'center', marginTop: 10, lineHeight: 15 }}>
          Opens your browser — handled entirely outside the app. LedgerLens itself never makes a
          network call.
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  body: { paddingHorizontal: spacing.md, paddingTop: spacing.lg, paddingBottom: spacing.xl },
  title: { fontSize: 24, textAlign: 'center' },
  sub: { fontSize: 12, textAlign: 'center', marginBottom: spacing.lg },
  pointRow: { flexDirection: 'row', gap: 8, alignItems: 'flex-start' },
  priceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    paddingTop: 12,
    borderTopWidth: 1,
    marginBottom: 12,
  },
  tipBtn: { borderRadius: radius.md, paddingVertical: 13, alignItems: 'center' },
});
