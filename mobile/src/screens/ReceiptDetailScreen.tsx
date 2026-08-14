import React from 'react';
import { Alert, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import { Card, Money, SectionLabel } from '../components/Primitives';
import { CATEGORY_ICON } from '../constants';
import { deleteReceipt } from '../storage/receipts';
import { useReceiptsStore } from '../storage/useReceiptsStore';
import { fonts, radius, spacing } from '../theme/tokens';
import { useTheme } from '../theme/useTheme';
import type { RootStackParamList } from '../navigation/RootNavigator';

type Props = NativeStackScreenProps<RootStackParamList, 'ReceiptDetail'>;

export default function ReceiptDetailScreen({ route, navigation }: Props) {
  const { colors } = useTheme();
  const { receipts, refresh } = useReceiptsStore();
  useFocusEffect(
    React.useCallback(() => {
      refresh();
    }, [refresh]),
  );

  const receipt = receipts.find(r => r.id === route.params.id);

  async function onDelete() {
    Alert.alert('Delete receipt?', 'This removes it from your ledger. This can\'t be undone.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          await deleteReceipt(route.params.id);
          navigation.goBack();
        },
      },
    ]);
  }

  if (!receipt) {
    return (
      <SafeAreaView style={{ flex: 1, backgroundColor: colors.paperRaised }} edges={['top']}>
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Text style={{ color: colors.moss, fontSize: 14 }}>‹ Back</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.emptyState}>
          <Text style={{ color: colors.inkFaint, fontSize: 13 }}>Receipt not found.</Text>
        </View>
      </SafeAreaView>
    );
  }

  const subtotal = receipt.total - receipt.tax;

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.paperRaised }} edges={['top']}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={{ color: colors.moss, fontSize: 14 }}>‹ Back</Text>
        </TouchableOpacity>
      </View>
      <ScrollView contentContainerStyle={styles.body}>
        <Text style={[styles.vendor, { color: colors.ink, fontFamily: fonts.display }]}>
          {receipt.vendor}
        </Text>
        <Text style={{ color: colors.inkFaint, fontSize: 12, marginBottom: spacing.md }}>
          {new Date(receipt.date).toLocaleDateString('en-US', {
            weekday: 'short',
            month: 'short',
            day: 'numeric',
            year: 'numeric',
          })}{' '}
          · {receipt.currency}
        </Text>

        <View style={[styles.categoryPill, { backgroundColor: colors.mossTint }]}>
          <Text style={{ fontSize: 13 }}>{CATEGORY_ICON[receipt.category] ?? '·'}</Text>
          <Text style={{ color: colors.mossDeep, fontSize: 12, fontWeight: '600' }}>
            {receipt.category}
          </Text>
        </View>

        <SectionLabel>Line items</SectionLabel>
        <Card style={{ marginBottom: spacing.lg }}>
          {receipt.lineItems.length === 0 ? (
            <Text style={{ color: colors.inkFaint, fontSize: 12, fontStyle: 'italic' }}>
              No line items recorded.
            </Text>
          ) : (
            receipt.lineItems.map((li, i) => (
              <View
                key={i}
                style={[
                  styles.liRow,
                  {
                    borderBottomColor: colors.line,
                    borderBottomWidth: i === receipt.lineItems.length - 1 ? 0 : 1,
                  },
                ]}>
                <View style={{ flex: 1 }}>
                  <Text style={{ color: colors.ink, fontSize: 12.5 }} numberOfLines={1}>
                    {li.description}
                  </Text>
                  <Text style={{ color: colors.inkFaint, fontSize: 10.5 }}>
                    {li.qty} × <Money value={li.unitPrice} size={10.5} />
                  </Text>
                </View>
                <Money value={li.total} size={12.5} />
              </View>
            ))
          )}

          <View style={[styles.totals, { borderTopColor: colors.lineStrong }]}>
            <View style={styles.totalsRow}>
              <Text style={{ color: colors.inkSoft, fontSize: 12 }}>Subtotal</Text>
              <Money value={subtotal} size={12} />
            </View>
            <View style={styles.totalsRow}>
              <Text style={{ color: colors.inkSoft, fontSize: 12 }}>Tax</Text>
              <Money value={receipt.tax} size={12} />
            </View>
            <View style={styles.totalsRow}>
              <Text style={{ color: colors.ink, fontSize: 15 }}>Total</Text>
              <Money value={receipt.total} size={15} emphasis />
            </View>
          </View>
        </Card>

        <TouchableOpacity
          onPress={onDelete}
          style={[styles.deleteBtn, { borderColor: colors.oxblood }]}>
          <Text style={{ color: colors.oxblood, fontWeight: '700', fontSize: 13 }}>
            Delete receipt
          </Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  header: { paddingHorizontal: spacing.md, paddingVertical: spacing.sm },
  emptyState: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  body: { paddingHorizontal: spacing.md, paddingBottom: spacing.xl },
  vendor: { fontSize: 22, marginBottom: 2 },
  categoryPill: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    gap: 6,
    borderRadius: radius.pill,
    paddingHorizontal: 12,
    paddingVertical: 6,
    marginBottom: spacing.lg,
  },
  liRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 8, gap: 8 },
  totals: { marginTop: 8, paddingTop: 10, borderTopWidth: 1, gap: 4 },
  totalsRow: { flexDirection: 'row', justifyContent: 'space-between' },
  deleteBtn: {
    borderRadius: radius.md,
    borderWidth: 1.5,
    paddingVertical: 12,
    alignItems: 'center',
  },
});
