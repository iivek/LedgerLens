import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  usePhotoOutput,
} from 'react-native-vision-camera';
import {
  Card,
  LocalBadge,
  Money,
  SectionLabel,
} from '../components/Primitives';
import { CATEGORIES } from '../constants';
import { discard, enqueue, retry, useScanQueue, type QueueItem } from '../model/scanQueue';
import { toReceiptFields } from '../model/schema';
import { useReceiptsStore } from '../storage/useReceiptsStore';
import type { Receipt } from '../types';
import { fonts, radius, spacing } from '../theme/tokens';
import { useTheme } from '../theme/useTheme';

// Dev-only escape hatch: pushed via adb to a fixed path for testing
// extraction against a real receipt photo when the physical camera isn't
// pointed at anything useful (e.g. bench-testing a phone hands-off).
const TEST_IMAGE_PATH = '/data/user/0/com.ledgerlensapp/files/test-receipt.jpg';

// The model's read is a starting point, not the final word -- editable here
// (rather than only after saving) so a receipt with a blank/wrong field
// doesn't force a save-then-fix round trip through the detail screen. Line
// items stay read-only, same reasoning as ReceiptDetailScreen's edit mode:
// they're a secondary detail, not what's actually likely to need correcting.
type Draft = Pick<Receipt, 'vendor' | 'date' | 'total' | 'tax'> & {
  category: string | null;
};

export default function ScanScreen() {
  const { colors } = useTheme();
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const photoOutput = usePhotoOutput({ quality: 0.9 });
  const { save } = useReceiptsStore();
  const items = useScanQueue();

  const [capturing, setCapturing] = useState(false);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  // Which queued item's review card is expanded -- only one at a time, so a
  // draft edit can't leak onto a different item.
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  // Flashes the injected image in place of the live camera for a beat when
  // the dev test-image button is tapped -- there's no real capturePhoto()
  // call to react to, so without this the frame would just stay on whatever
  // (possibly blank/dark) the lens happens to be pointed at.
  const [devPreview, setDevPreview] = useState<string | null>(null);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  async function onCapture() {
    setCapturing(true);
    try {
      const photo = await photoOutput.capturePhoto({}, {});
      const path = await photo.saveToTemporaryFileAsync();
      photo.dispose();
      setCaptureError(null);
      enqueue(path);
    } catch (err) {
      // Real device hardware can throw on the very first capture right after
      // permission is granted (camera session still warming up) -- surface a
      // plain retry prompt instead of letting this bubble into a raw native
      // error banner with no indication that trying again just works.
      console.warn('[LedgerLens] capture failed:', err);
      setCaptureError('Could not capture — try again.');
    } finally {
      setCapturing(false);
    }
  }

  function onUseTestImage() {
    setCaptureError(null);
    setDevPreview(TEST_IMAGE_PATH);
    setTimeout(() => setDevPreview(null), 1400);
    enqueue(TEST_IMAGE_PATH);
  }

  function selectItem(item: QueueItem) {
    if (selectedId === item.id) {
      setSelectedId(null);
      setDraft(null);
      return;
    }
    const fields = item.extraction ? toReceiptFields(item.extraction) : null;
    setSelectedId(item.id);
    setDraft(
      fields
        ? {
            vendor: fields.vendor,
            date: fields.date,
            total: fields.total,
            tax: fields.tax,
            category: null, // never pre-select -- the model doesn't guess a category
          }
        : null,
    );
  }

  async function onSaveItem(item: QueueItem) {
    if (!draft || !draft.category || !item.extraction) return; // button is disabled in this state, belt-and-braces
    setSaving(true);
    try {
      const fields = toReceiptFields(item.extraction);
      await save({
        ...fields,
        vendor: draft.vendor.trim() || fields.vendor,
        date: draft.date.trim() || fields.date,
        total: Number.isFinite(draft.total) ? draft.total : fields.total,
        tax: Number.isFinite(draft.tax) ? draft.tax : fields.tax,
        id: `receipt-${Date.now()}`,
        category: draft.category,
      });
      discard(item.id);
      setSelectedId(null);
      setDraft(null);
    } finally {
      setSaving(false);
    }
  }

  function onDiscardItem(item: QueueItem) {
    discard(item.id);
    if (item.id === selectedId) {
      setSelectedId(null);
      setDraft(null);
    }
  }

  const activeCount = items.filter(
    i => i.status === 'queued' || i.status === 'processing',
  ).length;

  return (
    <SafeAreaView
      style={{ flex: 1, backgroundColor: colors.paperRaised }}
      edges={['top']}
    >
      <View style={styles.statusStrip}>
        <LocalBadge
          label={
            activeCount > 0
              ? `Processing ${activeCount} in queue…`
              : 'Camera ready'
          }
        />
      </View>
      <ScrollView contentContainerStyle={styles.body}>
        <Text
          style={[
            styles.title,
            { color: colors.ink, fontFamily: fonts.display },
          ]}
        >
          Scan receipt
        </Text>

        <View
          style={[
            styles.captureFrame,
            { borderColor: colors.line, backgroundColor: colors.paperSunken },
          ]}
        >
          {devPreview ? (
            <Image
              source={{ uri: `file://${devPreview}` }}
              style={StyleSheet.absoluteFill}
              resizeMode="cover"
            />
          ) : hasPermission && device ? (
            <Camera
              device={device}
              isActive
              outputs={[photoOutput]}
              style={StyleSheet.absoluteFill}
            />
          ) : (
            <View style={styles.permissionFallback}>
              <Text
                style={{
                  color: colors.inkSoft,
                  fontSize: 12,
                  textAlign: 'center',
                }}
              >
                {hasPermission
                  ? 'No camera found'
                  : 'Camera permission needed to scan receipts'}
              </Text>
            </View>
          )}
        </View>

        <TouchableOpacity
          onPress={onCapture}
          disabled={!hasPermission || !device || capturing}
          style={[
            styles.captureBtn,
            {
              backgroundColor: colors.moss,
              opacity: hasPermission && device && !capturing ? 1 : 0.5,
            },
          ]}
        >
          <Text
            style={{ color: colors.paperRaised, fontWeight: '700', fontSize: 14 }}
          >
            {capturing ? 'Capturing…' : 'Capture receipt'}
          </Text>
        </TouchableOpacity>
        <Text
          style={{
            color: colors.inkFaint,
            fontSize: 11,
            fontStyle: 'italic',
            textAlign: 'center',
            marginTop: -spacing.sm,
            marginBottom: spacing.md,
          }}
        >
          Runs entirely on your phone's CPU, one receipt at a time in the
          background — keep scanning or switch tabs, it'll catch up.
        </Text>
        {captureError && (
          <Text
            style={{
              color: colors.oxblood,
              fontSize: 12,
              textAlign: 'center',
              marginTop: -spacing.sm,
              marginBottom: spacing.md,
            }}
          >
            {captureError}
          </Text>
        )}
        {__DEV__ && (
          <TouchableOpacity
            onPress={onUseTestImage}
            style={{ alignItems: 'center', marginBottom: spacing.md }}
          >
            <Text style={{ color: colors.inkFaint, fontSize: 11.5 }}>
              (dev) queue test image instead of camera
            </Text>
          </TouchableOpacity>
        )}

        {items.length > 0 && (
          <View style={styles.queueList}>
            <SectionLabel>Scan queue</SectionLabel>
            {items.map((item, index) => (
              <QueueRow
                key={item.id}
                index={index}
                item={item}
                expanded={item.id === selectedId}
                draft={item.id === selectedId ? draft : null}
                onDraftChange={setDraft}
                saving={saving && item.id === selectedId}
                onToggle={() => selectItem(item)}
                onSave={() => onSaveItem(item)}
                onDiscard={() => onDiscardItem(item)}
                onRetry={() => retry(item.id)}
              />
            ))}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function QueueRow({
  item,
  index,
  expanded,
  draft,
  onDraftChange,
  saving,
  onToggle,
  onSave,
  onDiscard,
  onRetry,
}: {
  item: QueueItem;
  index: number;
  expanded: boolean;
  draft: Draft | null;
  onDraftChange: (draft: Draft) => void;
  saving: boolean;
  onToggle: () => void;
  onSave: () => void;
  onDiscard: () => void;
  onRetry: () => void;
}) {
  const { colors } = useTheme();
  const fields = item.extraction ? toReceiptFields(item.extraction) : null;
  const visibleItems = fields ? fields.lineItems.slice(0, 4) : [];
  const hiddenCount = fields ? fields.lineItems.length - visibleItems.length : 0;

  return (
    <Card>
      <TouchableOpacity
        onPress={item.status === 'done' ? onToggle : undefined}
        style={styles.queueRowHeader}
        disabled={item.status !== 'done'}
      >
        <View style={{ flex: 1 }}>
          {item.status === 'done' && fields ? (
            <>
              <Text
                style={{ color: colors.ink, fontFamily: fonts.display, fontSize: 14 }}
                numberOfLines={1}
              >
                {fields.vendor}
              </Text>
              <Text style={{ color: colors.inkFaint, fontSize: 11 }}>
                {fields.currency} · ${fields.total.toFixed(2)}
              </Text>
            </>
          ) : (
            <Text style={{ color: colors.inkSoft, fontSize: 12.5 }}>
              {item.status === 'processing'
                ? 'Extracting on-device…'
                : item.status === 'error'
                ? item.error ?? 'Could not read this receipt.'
                : 'Queued…'}
            </Text>
          )}
        </View>
        {item.status === 'processing' && (
          <ActivityIndicator color={colors.moss} size="small" />
        )}
        {item.status === 'error' && (
          <TouchableOpacity onPress={onRetry} hitSlop={8}>
            <Text style={{ color: colors.moss, fontSize: 11.5, fontWeight: '600' }}>
              Retry
            </Text>
          </TouchableOpacity>
        )}
        <TouchableOpacity
          onPress={onDiscard}
          hitSlop={8}
          style={{ marginLeft: spacing.sm }}
          testID={`queue-discard-${index}`}
        >
          <Text style={{ color: colors.inkFaint, fontSize: 15 }}>✕</Text>
        </TouchableOpacity>
      </TouchableOpacity>

      {expanded && fields && draft && (
        <View style={{ marginTop: spacing.sm }}>
          <SectionLabel>Vendor</SectionLabel>
          <TextInput
            value={draft.vendor}
            onChangeText={v => onDraftChange({ ...draft, vendor: v })}
            style={[styles.input, { color: colors.ink, borderColor: colors.lineStrong }]}
          />

          <SectionLabel>Date (YYYY-MM-DD)</SectionLabel>
          <TextInput
            value={draft.date}
            onChangeText={v => onDraftChange({ ...draft, date: v })}
            style={[styles.input, { color: colors.ink, borderColor: colors.lineStrong }]}
          />

          {visibleItems.map((li, i) => (
            <View key={i} style={[styles.liRow, { borderBottomColor: colors.lineStrong }]}>
              <Text
                style={{ color: colors.inkSoft, fontSize: 12, flex: 1 }}
                numberOfLines={1}
              >
                {li.description}
              </Text>
              <Money value={li.total} size={12} />
            </View>
          ))}
          {hiddenCount > 0 && (
            <Text
              style={{
                color: colors.inkFaint,
                fontSize: 11,
                fontStyle: 'italic',
                paddingVertical: 6,
              }}
            >
              + {hiddenCount} more items
            </Text>
          )}

          <View style={styles.amountRow}>
            <View style={{ flex: 1 }}>
              <SectionLabel>Total</SectionLabel>
              <TextInput
                value={String(draft.total)}
                onChangeText={v => onDraftChange({ ...draft, total: Number(v) })}
                keyboardType="decimal-pad"
                style={[styles.input, { color: colors.ink, borderColor: colors.lineStrong }]}
              />
            </View>
            <View style={{ flex: 1 }}>
              <SectionLabel>Tax</SectionLabel>
              <TextInput
                value={String(draft.tax)}
                onChangeText={v => onDraftChange({ ...draft, tax: Number(v) })}
                keyboardType="decimal-pad"
                style={[styles.input, { color: colors.ink, borderColor: colors.lineStrong }]}
              />
            </View>
          </View>

          <View style={styles.categoryPicker}>
            <SectionLabel>Category — your choice</SectionLabel>
            <View style={styles.chipRow}>
              {CATEGORIES.map(c => (
                <TouchableOpacity
                  key={c}
                  onPress={() => onDraftChange({ ...draft, category: c })}
                >
                  <Text
                    style={[
                      styles.chip,
                      {
                        backgroundColor: draft.category === c ? colors.moss : colors.paperRaised,
                        borderColor: draft.category === c ? colors.moss : colors.lineStrong,
                        color: draft.category === c ? colors.paperRaised : colors.inkSoft,
                        fontWeight: draft.category === c ? '700' : '400',
                      },
                    ]}
                  >
                    {c}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
            <Text
              style={{
                color: colors.inkFaint,
                fontSize: 10.5,
                fontStyle: 'italic',
                marginTop: 8,
              }}
            >
              The model reads the receipt — vendor, date, and amounts are
              editable above if it got something wrong. Category is always
              your tap, never guessed.
            </Text>
          </View>

          <TouchableOpacity
            style={[
              styles.saveBtn,
              { backgroundColor: colors.moss, opacity: draft.category && !saving ? 1 : 0.5 },
            ]}
            disabled={!draft.category || saving}
            onPress={onSave}
          >
            <Text style={{ color: colors.paperRaised, fontWeight: '700', fontSize: 13.5 }}>
              {saving ? 'Saving…' : draft.category ? 'Save to ledger' : 'Pick a category to save'}
            </Text>
          </TouchableOpacity>
        </View>
      )}
    </Card>
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
  permissionFallback: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.lg,
  },
  captureBtn: {
    borderRadius: radius.md,
    paddingVertical: 14,
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  queueList: { gap: spacing.sm },
  queueRowHeader: { flexDirection: 'row', alignItems: 'center' },
  liRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 5,
    borderBottomWidth: 1,
    gap: 8,
  },
  input: {
    borderWidth: 1,
    borderRadius: radius.sm,
    paddingHorizontal: 10,
    paddingVertical: 8,
    fontSize: 13,
    marginBottom: spacing.sm,
  },
  amountRow: { flexDirection: 'row', gap: spacing.sm, marginTop: spacing.xs },
  categoryPicker: { marginTop: 14 },
  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 7 },
  chip: {
    fontSize: 11.5,
    borderRadius: radius.pill,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 6,
    overflow: 'hidden',
  },
  saveBtn: {
    marginTop: 14,
    borderRadius: radius.md,
    paddingVertical: 12,
    alignItems: 'center',
  },
});
