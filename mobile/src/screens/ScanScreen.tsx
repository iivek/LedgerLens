import React, { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Image,
  ScrollView,
  StyleSheet,
  Text,
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
import { cancelExtraction, extractReceipt } from '../model/extract';
import { toReceiptFields, type RawExtraction } from '../model/schema';
import { useReceiptsStore } from '../storage/useReceiptsStore';
import { fonts, radius, spacing } from '../theme/tokens';
import { useTheme } from '../theme/useTheme';

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
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [extracting, setExtracting] = useState(false);
  const [extraction, setExtraction] = useState<RawExtraction | null>(null);
  const [extractionError, setExtractionError] = useState<string | null>(null);
  // A CPU-only extraction can take many minutes (see CLAUDE.md's on-device
  // latency caveat), long enough that a user can retake and start a second
  // one before the first's promise settles -- this stops the stale run's
  // eventual result/error from clobbering whatever's current by then.
  const extractionIdRef = useRef(0);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  async function onCapture() {
    setCapturing(true);
    let path: string;
    try {
      const photo = await photoOutput.capturePhoto({}, {});
      path = await photo.saveToTemporaryFileAsync();
      photo.dispose();
      setCapturedPath(path);
      setCategory(null); // never pre-select -- the model doesn't guess a category
      setCaptureError(null);
    } catch (err) {
      // Real device hardware can throw on the very first capture right after
      // permission is granted (camera session still warming up) -- surface a
      // plain retry prompt instead of letting this bubble into a raw native
      // error banner with no indication that trying again just works.
      console.warn('[LedgerLens] capture failed:', err);
      setCaptureError('Could not capture — try again.');
      return;
    } finally {
      setCapturing(false);
    }
    runExtraction(path);
  }

  // Dev-only escape hatch: pushed via adb to a fixed path for testing
  // extraction against a real receipt photo when the physical camera isn't
  // pointed at anything useful (e.g. bench-testing a phone hands-off).
  // Goes through the exact same runExtraction() path a real capture does.
  function onUseTestImage() {
    const path = '/data/user/0/com.ledgerlensapp/files/test-receipt.jpg';
    setCapturedPath(path);
    setCategory(null);
    setCaptureError(null);
    runExtraction(path);
  }

  async function runExtraction(path: string) {
    const requestId = ++extractionIdRef.current;
    setExtracting(true);
    setExtraction(null);
    setExtractionError(null);
    try {
      const raw = await extractReceipt(path);
      if (requestId !== extractionIdRef.current) return; // superseded by a retake
      if (raw) {
        // Deliberately always logged, not just on failure -- a JS-context
        // reload (e.g. Metro reconnecting after Doze suspends it during a
        // long background run) can orphan this promise's UI-side update
        // with nothing else showing the result actually arrived.
        console.log('[LedgerLens] extraction succeeded:', JSON.stringify(raw));
        setExtraction(raw);
      } else {
        setExtractionError(
          'Could not read this receipt — try again or retake.',
        );
      }
    } catch (err) {
      if (requestId !== extractionIdRef.current) return;
      console.warn('[LedgerLens] extraction failed:', err);
      setExtractionError('Could not read this receipt — try again or retake.');
    } finally {
      if (requestId === extractionIdRef.current) setExtracting(false);
    }
  }

  function onRetake() {
    if (extracting) {
      // Invalidate the in-flight request first so its eventual settle is a
      // no-op, then actually stop the native inference -- don't let a wall-
      // clock-minutes-long CPU run keep burning battery unseen just because
      // the UI moved on (see CLAUDE.md's on-device latency caveat).
      extractionIdRef.current++;
      cancelExtraction().catch(err =>
        console.warn('[LedgerLens] cancelExtraction failed:', err),
      );
    }
    setCapturedPath(null);
    setCategory(null);
    setExtracting(false);
    setExtraction(null);
    setExtractionError(null);
  }

  async function onSave() {
    if (!category || !extraction) return; // button is disabled in this state, belt-and-braces
    setSaving(true);
    try {
      await save({
        ...toReceiptFields(extraction),
        id: `receipt-${Date.now()}`,
        category,
      });
      setCapturedPath(null);
      setCategory(null);
      setExtraction(null);
    } finally {
      setSaving(false);
    }
  }

  const fields = extraction ? toReceiptFields(extraction) : null;
  const visibleItems = fields ? fields.lineItems.slice(0, 4) : [];
  const hiddenCount = fields
    ? fields.lineItems.length - visibleItems.length
    : 0;
  const subtotal = fields ? fields.total - fields.tax : 0;

  return (
    <SafeAreaView
      style={{ flex: 1, backgroundColor: colors.paperRaised }}
      edges={['top']}
    >
      <View style={styles.statusStrip}>
        <LocalBadge
          label={
            extracting
              ? 'Extracting on-device…'
              : capturedPath
              ? 'Ready'
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
          {capturedPath ? (
            // The real saved photo, not the live preview frozen on its last
            // frame -- that frame is a hardware-composited surface that
            // doesn't participate in ScrollView transforms correctly (it
            // visibly narrows/distorts while scrolling on real devices).
            <Image
              source={{ uri: `file://${capturedPath}` }}
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

        {!capturedPath && (
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
              style={{
                color: colors.paperRaised,
                fontWeight: '700',
                fontSize: 14,
              }}
            >
              {capturing ? 'Capturing…' : 'Capture receipt'}
            </Text>
          </TouchableOpacity>
        )}
        {captureError && !capturedPath && (
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
        {__DEV__ && !capturedPath && (
          <TouchableOpacity
            onPress={onUseTestImage}
            style={{ alignItems: 'center', marginBottom: spacing.md }}
          >
            <Text style={{ color: colors.inkFaint, fontSize: 11.5 }}>
              (dev) use test image instead of camera
            </Text>
          </TouchableOpacity>
        )}

        {capturedPath && (
          <Card>
            <View style={styles.retakeRow}>
              <TouchableOpacity onPress={onRetake}>
                <Text
                  style={{
                    color: colors.moss,
                    fontSize: 12.5,
                    fontWeight: '600',
                  }}
                >
                  ‹ Retake photo
                </Text>
              </TouchableOpacity>
            </View>

            {extracting && (
              <View>
                <View style={styles.extractingRow}>
                  <ActivityIndicator color={colors.moss} />
                  <Text style={{ color: colors.inkSoft, fontSize: 12.5 }}>
                    Reading receipt on-device…
                  </Text>
                </View>
                <Text
                  style={{
                    color: colors.inkFaint,
                    fontSize: 11,
                    fontStyle: 'italic',
                    marginTop: -4,
                    marginBottom: 8,
                  }}
                >
                  This runs entirely on your phone's CPU — on some devices that
                  can take several minutes, not seconds. Feel free to switch
                  tabs; it keeps running in the background.
                </Text>
              </View>
            )}

            {!extracting && extractionError && (
              <Text
                style={{
                  color: colors.oxblood,
                  fontSize: 12.5,
                  paddingVertical: 8,
                }}
              >
                {extractionError}
              </Text>
            )}

            {!extracting && fields && (
              <>
                <Text
                  style={[
                    styles.vendor,
                    { color: colors.ink, fontFamily: fonts.display },
                  ]}
                >
                  {fields.vendor}
                </Text>
                <Text
                  style={{
                    color: colors.inkFaint,
                    fontSize: 11,
                    marginBottom: 12,
                  }}
                >
                  {new Date(fields.date).toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric',
                  })}{' '}
                  · {fields.currency}
                </Text>

                {visibleItems.map((li, i) => (
                  <View
                    key={i}
                    style={[
                      styles.liRow,
                      { borderBottomColor: colors.lineStrong },
                    ]}
                  >
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

                <View
                  style={[styles.totals, { borderTopColor: colors.lineStrong }]}
                >
                  <View style={styles.totalsRow}>
                    <Text style={{ color: colors.inkSoft, fontSize: 12 }}>
                      Subtotal
                    </Text>
                    <Money value={subtotal} size={12} />
                  </View>
                  <View style={styles.totalsRow}>
                    <Text style={{ color: colors.inkSoft, fontSize: 12 }}>
                      Tax
                    </Text>
                    <Money value={fields.tax} size={12} />
                  </View>
                  <View style={styles.totalsRow}>
                    <Text style={{ color: colors.ink, fontSize: 15 }}>
                      Total
                    </Text>
                    <Money value={fields.total} size={15} emphasis />
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
                              backgroundColor:
                                category === c
                                  ? colors.moss
                                  : colors.paperRaised,
                              borderColor:
                                category === c
                                  ? colors.moss
                                  : colors.lineStrong,
                              color:
                                category === c
                                  ? colors.paperRaised
                                  : colors.inkSoft,
                              fontWeight: category === c ? '700' : '400',
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
                    The model reads the receipt. It never guesses a category —
                    that's always your tap.
                  </Text>
                </View>

                <TouchableOpacity
                  style={[
                    styles.saveBtn,
                    {
                      backgroundColor: colors.moss,
                      opacity: category && !saving ? 1 : 0.5,
                    },
                  ]}
                  disabled={!category || saving}
                  onPress={onSave}
                >
                  <Text
                    style={{
                      color: colors.paperRaised,
                      fontWeight: '700',
                      fontSize: 13.5,
                    }}
                  >
                    {saving
                      ? 'Saving…'
                      : category
                      ? 'Save to ledger'
                      : 'Pick a category to save'}
                  </Text>
                </TouchableOpacity>
              </>
            )}
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
  retakeRow: { marginBottom: 10 },
  extractingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 10,
  },
  vendor: { fontSize: 17, marginBottom: 2 },
  liRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 5,
    borderBottomWidth: 1,
    gap: 8,
  },
  totals: { marginTop: 8, paddingTop: 8, borderTopWidth: 1, gap: 4 },
  totalsRow: { flexDirection: 'row', justifyContent: 'space-between' },
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
