import React, { useEffect, useState } from 'react';
import { ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  usePhotoOutput,
} from 'react-native-vision-camera';
import { Card, LocalBadge, Money, SectionLabel } from '../components/Primitives';
import { CATEGORIES } from '../constants';
import { sampleReceipts } from '../data/sampleData';
import { useReceiptsStore } from '../storage/useReceiptsStore';
import { fonts, radius, spacing } from '../theme/tokens';
import { useTheme } from '../theme/useTheme';

// The Trader Joe's sample receipt stands in for a real extraction result
// until llama.rn is wired up (next step) -- capturing a real photo already
// works below and the save-to-ledger path below is real, it's just the
// vendor/line-items/total shown here that are canned, not read from the photo.
const STUB_EXTRACTION = sampleReceipts[1];

export default function ScanScreen() {
  const { colors } = useTheme();
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const photoOutput = usePhotoOutput({ quality: 0.9 });
  const { save } = useReceiptsStore();

  const [capturedPath, setCapturedPath] = useState<string | null>(null);
  const [category, setCategory] = useState<string | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  async function onCapture() {
    setCapturing(true);
    try {
      const photo = await photoOutput.capturePhoto({}, {});
      const path = await photo.saveToTemporaryFileAsync();
      photo.dispose();
      setCapturedPath(path);
      setCategory(null); // never pre-select -- the model doesn't guess a category
    } finally {
      setCapturing(false);
    }
  }

  async function onSave() {
    if (!category) return; // button is disabled in this state, belt-and-braces
    setSaving(true);
    try {
      await save({
        ...STUB_EXTRACTION,
        id: `receipt-${Date.now()}`,
        date: new Date().toISOString().slice(0, 10),
        category,
      });
      setCapturedPath(null);
      setCategory(null);
    } finally {
      setSaving(false);
    }
  }

  const visibleItems = STUB_EXTRACTION.lineItems.slice(0, 4);
  const hiddenCount = STUB_EXTRACTION.lineItems.length - visibleItems.length;
  const subtotal = STUB_EXTRACTION.total - STUB_EXTRACTION.tax;

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.paperRaised }} edges={['top']}>
      <View style={styles.statusStrip}>
        <LocalBadge label={capturedPath ? 'Extracting on-device…' : 'Camera ready'} />
      </View>
      <ScrollView contentContainerStyle={styles.body}>
        <Text style={[styles.title, { color: colors.ink, fontFamily: fonts.display }]}>
          Scan receipt
        </Text>

        <View style={[styles.captureFrame, { borderColor: colors.line, backgroundColor: colors.paperSunken }]}>
          {hasPermission && device ? (
            <Camera
              device={device}
              isActive={!capturedPath}
              outputs={[photoOutput]}
              style={StyleSheet.absoluteFill}
            />
          ) : (
            <View style={styles.permissionFallback}>
              <Text style={{ color: colors.inkSoft, fontSize: 12, textAlign: 'center' }}>
                {hasPermission ? 'No camera found' : 'Camera permission needed to scan receipts'}
              </Text>
            </View>
          )}
        </View>

        {!capturedPath && (
          <TouchableOpacity
            onPress={onCapture}
            disabled={!hasPermission || !device || capturing}
            style={[styles.captureBtn, { backgroundColor: colors.moss }]}>
            <Text style={{ color: colors.paperRaised, fontWeight: '700', fontSize: 14 }}>
              {capturing ? 'Capturing…' : 'Capture receipt'}
            </Text>
          </TouchableOpacity>
        )}

        {capturedPath && (
          <Card>
            <Text style={[styles.vendor, { color: colors.ink, fontFamily: fonts.display }]}>
              {STUB_EXTRACTION.vendor}
            </Text>
            <Text style={{ color: colors.inkFaint, fontSize: 11, marginBottom: 12 }}>
              {new Date().toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric',
              })}{' '}
              · {STUB_EXTRACTION.currency}
            </Text>

            {visibleItems.map((li, i) => (
              <View
                key={i}
                style={[styles.liRow, { borderBottomColor: colors.lineStrong }]}>
                <Text style={{ color: colors.inkSoft, fontSize: 12, flex: 1 }} numberOfLines={1}>
                  {li.description}
                </Text>
                <Money value={li.total} size={12} />
              </View>
            ))}
            {hiddenCount > 0 && (
              <Text style={{ color: colors.inkFaint, fontSize: 11, fontStyle: 'italic', paddingVertical: 6 }}>
                + {hiddenCount} more items
              </Text>
            )}

            <View style={[styles.totals, { borderTopColor: colors.lineStrong }]}>
              <View style={styles.totalsRow}>
                <Text style={{ color: colors.inkSoft, fontSize: 12 }}>Subtotal</Text>
                <Money value={subtotal} size={12} />
              </View>
              <View style={styles.totalsRow}>
                <Text style={{ color: colors.inkSoft, fontSize: 12 }}>Tax</Text>
                <Money value={STUB_EXTRACTION.tax} size={12} />
              </View>
              <View style={styles.totalsRow}>
                <Text style={{ color: colors.ink, fontSize: 15 }}>Total</Text>
                <Money value={STUB_EXTRACTION.total} size={15} emphasis />
              </View>
            </View>

            <View style={styles.categoryPicker}>
              <SectionLabel>Category — your choice</SectionLabel>
              <View style={styles.chipRow}>
                {CATEGORIES.map(c => (
                  <TouchableOpacity key={c} onPress={() => setCategory(c)}>
                    <Text
                      style={[
                        styles.chip,
                        {
                          backgroundColor: category === c ? colors.moss : colors.paperRaised,
                          borderColor: category === c ? colors.moss : colors.lineStrong,
                          color: category === c ? colors.paperRaised : colors.inkSoft,
                          fontWeight: category === c ? '700' : '400',
                        },
                      ]}>
                      {c}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
              <Text style={{ color: colors.inkFaint, fontSize: 10.5, fontStyle: 'italic', marginTop: 8 }}>
                The model reads the receipt. It never guesses a category — that's always your tap.
              </Text>
            </View>

            <TouchableOpacity
              style={[
                styles.saveBtn,
                { backgroundColor: colors.moss, opacity: category && !saving ? 1 : 0.5 },
              ]}
              disabled={!category || saving}
              onPress={onSave}>
              <Text style={{ color: colors.paperRaised, fontWeight: '700', fontSize: 13.5 }}>
                {saving ? 'Saving…' : category ? 'Save to ledger' : 'Pick a category to save'}
              </Text>
            </TouchableOpacity>
          </Card>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  statusStrip: { alignItems: 'center', paddingVertical: spacing.sm },
  body: { paddingHorizontal: spacing.md, paddingBottom: spacing.xl },
  title: { fontSize: 22, marginBottom: spacing.md },
  captureFrame: {
    aspectRatio: 3 / 4,
    borderRadius: radius.md,
    borderWidth: 1,
    overflow: 'hidden',
    marginBottom: spacing.md,
  },
  permissionFallback: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: spacing.lg },
  captureBtn: { borderRadius: radius.md, paddingVertical: 14, alignItems: 'center', marginBottom: spacing.md },
  vendor: { fontSize: 17, marginBottom: 2 },
  liRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 5, borderBottomWidth: 1, gap: 8 },
  totals: { marginTop: 8, paddingTop: 8, borderTopWidth: 1, gap: 4 },
  totalsRow: { flexDirection: 'row', justifyContent: 'space-between' },
  categoryPicker: { marginTop: 14 },
  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 7 },
  chip: { fontSize: 11.5, borderRadius: radius.pill, borderWidth: 1, paddingHorizontal: 12, paddingVertical: 6, overflow: 'hidden' },
  saveBtn: { marginTop: 14, borderRadius: radius.md, paddingVertical: 12, alignItems: 'center' },
});
