import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  Animated,
  ScrollView,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { File, Paths } from 'expo-file-system';
import { useRef, useState, useEffect, useCallback } from 'react';
import { useRouter } from 'expo-router';
import { parseSetAndNumber } from '../../src/scanner/ocr';
import { fetchCardBySetNumber } from '../../src/api/scryfall';
import { upsertCard } from '../../src/db/cards';
import { useStore } from '../../src/store/useStore';
import { useTheme } from '../../src/theme/useTheme';

type ScanPhase =
  | { status: 'idle' }
  | { status: 'scanning' }
  | { status: 'ocr_raw'; text: string }
  | { status: 'parsed'; setCode: string; collectorNumber: string; rawText: string }
  | { status: 'fetching'; setCode: string; collectorNumber: string }
  | { status: 'error'; message: string };

/**
 * Resolves a URI to a local file:// path that react-native-text-recognition
 * can read.  The library's Swift layer does `URL(fileURLWithPath:)` after
 * stripping "file://", so ph:// and content:// URIs must be copied to a
 * temp cache file first.
 */
async function resolveToFileUri(uri: string): Promise<string> {
  if (uri.startsWith('file://') || uri.startsWith('/')) {
    return uri;
  }
  const dest = new File(Paths.cache, `scan_ocr_${Date.now()}.jpg`);
  const source = new File(uri);
  source.copy(dest);
  return dest.uri;
}

async function runOcr(uri: string): Promise<string> {
  const TextRecognition = require('react-native-text-recognition').default;
  if (!TextRecognition || typeof TextRecognition.recognize !== 'function') {
    throw new Error(
      'OCR module is not available. Rebuild the app with `expo run:ios` to link native dependencies.'
    );
  }
  const resolvedUri = await resolveToFileUri(uri);
  const lines: string[] = await TextRecognition.recognize(resolvedUri);
  return lines.join('\n');
}


function OcrDebugPanel({ phase, ocrText }: { phase: ScanPhase; ocrText: string }) {
  // Visible during all active phases (hide only on idle)
  if (phase.status === 'idle') return null;

  const allLines = ocrText ? ocrText.split('\n') : [];

  let parseSummary: string | null = null;
  let parseSuccess = false;

  if (phase.status === 'scanning') {
    parseSummary = allLines.length > 0 ? 'No set/number found yet' : null;
  } else if (phase.status === 'ocr_raw') {
    parseSummary = 'Parsing…';
  } else if (phase.status === 'parsed') {
    parseSummary = `Set: ${phase.setCode.toUpperCase()}  #${phase.collectorNumber}`;
    parseSuccess = true;
  } else if (phase.status === 'fetching') {
    parseSummary = `Set: ${phase.setCode.toUpperCase()}  #${phase.collectorNumber}`;
    parseSuccess = true;
  } else if (phase.status === 'error') {
    parseSummary = phase.message;
    parseSuccess = false;
  }

  const totalLines = allLines.length;
  // Bottom 3 lines are what the parser examined
  const anchorStart = Math.max(0, totalLines - 3);

  return (
    <View style={debugStyles.panel}>
      <Text style={debugStyles.header}>OCR DEBUG</Text>

      {phase.status === 'scanning' && allLines.length === 0 && (
        <Text style={debugStyles.waiting}>waiting for OCR…</Text>
      )}
      {phase.status !== 'scanning' && allLines.length === 0 && (
        <Text style={debugStyles.waiting}>no text captured</Text>
      )}

      {allLines.length > 0 && (
        <ScrollView
          style={debugStyles.scroll}
          contentContainerStyle={debugStyles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {allLines.map((line, i) => {
            const isAnchor = i >= anchorStart;
            return (
              <Text
                key={i}
                style={[debugStyles.line, isAnchor && debugStyles.lineAnchor]}
              >
                {isAnchor ? '▶ ' : '  '}
                {line || ' '}
              </Text>
            );
          })}
        </ScrollView>
      )}

      {parseSummary !== null && (
        <View style={debugStyles.summaryRow}>
          <Text
            style={[
              debugStyles.summary,
              parseSuccess ? debugStyles.summaryOk : debugStyles.summaryFail,
            ]}
          >
            {parseSummary}
          </Text>
        </View>
      )}
    </View>
  );
}

function ErrorToast({ message }: { message: string }) {
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.sequence([
      Animated.timing(fadeAnim, { toValue: 1, duration: 200, useNativeDriver: true }),
    ]).start();
  }, [message, fadeAnim]);

  return (
    <Animated.View style={[styles.errorToast, { opacity: fadeAnim }]}>
      <Text style={styles.errorToastText}>{message}</Text>
    </Animated.View>
  );
}

export default function ScanScreen() {
  const theme = useTheme();
  const router = useRouter();
  const [permission, requestPermission] = useCameraPermissions();
  const [phase, setPhase] = useState<ScanPhase>({ status: 'idle' });
  const [ocrText, setOcrText] = useState<string>('');
  const [isActive, setIsActive] = useState(true);
  const [pickedImageUri, setPickedImageUri] = useState<string | null>(null);
  const [successCard, setSuccessCard] = useState<string | null>(null);
  const cameraRef = useRef<CameraView>(null);
  const { setLastScannedId, addRecentScan, recentScans } = useStore();
  const scanLoopRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const stopScanning = useCallback(() => {
    setIsActive(false);
    if (scanLoopRef.current) {
      clearTimeout(scanLoopRef.current);
      scanLoopRef.current = null;
    }
  }, []);

  const processUri = useCallback(async (uri: string, reschedule: boolean) => {
    try {
      const rawText = await runOcr(uri);
      setOcrText(rawText);
      setPhase({ status: 'ocr_raw', text: rawText });

      const parsed = parseSetAndNumber(rawText);
      if (!parsed) {
        if (reschedule) {
          scanLoopRef.current = setTimeout(runScanCycle, 1200);
        } else {
          setPhase({ status: 'error', message: 'No card info found in image' });
        }
        return;
      }

      setPhase({ status: 'parsed', ...parsed, rawText });
      setPhase({ status: 'fetching', setCode: parsed.setCode, collectorNumber: parsed.collectorNumber });
      const card = await fetchCardBySetNumber(parsed.setCode, parsed.collectorNumber);
      upsertCard(card);
      addRecentScan(card);
      setLastScannedId(card.scryfall_id);
      setPhase({ status: 'idle' });
      setSuccessCard(card.name);
      setTimeout(() => {
        setSuccessCard(null);
        setPickedImageUri(null);
        if (reschedule) runScanCycle();
      }, 1500);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error';
      setPhase({ status: 'error', message: msg });
    }
  }, [setLastScannedId, addRecentScan]); // eslint-disable-line react-hooks/exhaustive-deps

  const runScanCycle = useCallback(async () => {
    if (!cameraRef.current) return;
    setOcrText('');
    setPhase({ status: 'scanning' });

    const photo = await cameraRef.current.takePictureAsync({ quality: 0.6 });
    if (!photo) {
      setPhase({ status: 'error', message: 'Camera failed to capture' });
      return;
    }

    await processUri(photo.uri, true);
  }, [processUri]);

  const pickFromLibrary = useCallback(async () => {
    stopScanning();
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 1,
    });
    if (result.canceled || !result.assets[0]) return;

    const uri = result.assets[0].uri;
    setPickedImageUri(uri);
    setPhase({ status: 'scanning' });
    await processUri(uri, false);
  }, [stopScanning, processUri]);

  useEffect(() => {
    if (isActive) {
      runScanCycle();
    }
    return () => {
      if (scanLoopRef.current) clearTimeout(scanLoopRef.current);
    };
  }, [isActive]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!permission) {
    return <View style={[styles.screen, { backgroundColor: theme.bg }]} />;
  }

  if (!permission.granted) {
    return (
      <View style={[styles.center, { backgroundColor: theme.bg }]}>
        <Text style={[styles.permText, { color: theme.textSecondary }]}>
          Camera access is needed to scan cards.
        </Text>
        <TouchableOpacity
          style={[styles.pill, { backgroundColor: theme.accent }]}
          onPress={requestPermission}
          activeOpacity={0.75}
        >
          <Text style={[styles.pillText, { color: '#000' }]}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const statusLabel = (() => {
    switch (phase.status) {
      case 'idle': return isActive ? 'Point camera at a card' : 'Tap to start scanning';
      case 'scanning': return 'Looking for card…';
      case 'ocr_raw': return 'Reading text…';
      case 'parsed': return `Found: ${phase.setCode.toUpperCase()} #${phase.collectorNumber}`;
      case 'fetching': return `Looking up ${phase.setCode.toUpperCase()} #${phase.collectorNumber}…`;
      case 'error': return null; // shown as toast instead
    }
  })();

  const cameraOverlay = (
    <View style={styles.overlay} pointerEvents="none">
      {/* Status label — bottom center */}
      {statusLabel !== null && (
        <View style={styles.statusBadge}>
          <Text style={styles.statusText}>{statusLabel}</Text>
        </View>
      )}
    </View>
  );

  return (
    <View style={styles.screen}>
      {/* Error toast — absolute, pinned to top */}
      {phase.status === 'error' && (
        <ErrorToast message={phase.message} />
      )}

      {successCard !== null && (
        <View style={scanStyles.successBadge} pointerEvents="none">
          <Text style={scanStyles.successText}>✓ {successCard}</Text>
        </View>
      )}

      {pickedImageUri ? (
        <View style={styles.fullScreenImageContainer}>
          <Image source={{ uri: pickedImageUri }} style={styles.fullScreenImage} resizeMode="contain" />
          {cameraOverlay}
          {/* OCR debug panel inside the fullscreen container so it's unambiguously above the image layer */}
          <OcrDebugPanel phase={phase} ocrText={ocrText} />
        </View>
      ) : (
        <>
          <CameraView ref={cameraRef} style={styles.camera} facing="back">
            {cameraOverlay}
          </CameraView>
          {/* OCR debug panel — absolute, bottom of screen, above footer */}
          <OcrDebugPanel phase={phase} ocrText={ocrText} />
        </>
      )}

      <View
        style={[
          styles.footer,
          {
            backgroundColor: pickedImageUri ? 'transparent' : theme.bg,
            ...(pickedImageUri ? styles.footerAbsolute : {}),
          },
        ]}
      >
        <TouchableOpacity
          style={[styles.pill, styles.pillSecondary, { borderColor: theme.border }]}
          onPress={pickFromLibrary}
          activeOpacity={0.7}
        >
          <Text style={[styles.pillText, { color: theme.text }]}>Photo Library</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.textLink}
          onPress={() => { stopScanning(); router.push('/search'); }}
          activeOpacity={0.6}
        >
          <Text style={[styles.textLinkText, { color: theme.textSecondary }]}>Search instead</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#000' },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
    padding: 32,
  },
  permText: { textAlign: 'center', fontSize: 15, lineHeight: 22 },

  // Pill buttons
  pill: {
    paddingHorizontal: 28,
    paddingVertical: 12,
    borderRadius: 100,
    alignItems: 'center',
    justifyContent: 'center',
  },
  pillSecondary: {
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderWidth: StyleSheet.hairlineWidth,
  },
  pillText: { fontSize: 15, fontWeight: '600', letterSpacing: 0.2 },

  textLink: { padding: 8 },
  textLinkText: { fontSize: 13, letterSpacing: 0.1 },

  // Camera / image
  camera: { flex: 1 },
  fullScreenImageContainer: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 10,
    backgroundColor: '#000',
  },
  fullScreenImage: { ...StyleSheet.absoluteFillObject },

  // Overlay sits on top of camera feed
  overlay: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    paddingBottom: 80,
  },

  // Error toast pinned to top
  errorToast: {
    position: 'absolute',
    top: 56,
    left: 20,
    right: 20,
    zIndex: 100,
    backgroundColor: '#b71c1c',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 10,
  },
  errorToastText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '500',
    lineHeight: 20,
    textAlign: 'center',
  },

  statusBadge: {
    marginTop: 20,
    backgroundColor: 'rgba(0,0,0,0.55)',
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 100,
  },
  statusText: { color: '#fff', fontSize: 13, fontWeight: '500' },

  footer: {
    padding: 24,
    paddingBottom: 32,
    alignItems: 'center',
    gap: 12,
    zIndex: 20,
  },
  footerAbsolute: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
  },
});

// Separate stylesheet for the OCR debug panel so it doesn't pollute the main styles object.
const debugStyles = StyleSheet.create({
  panel: {
    position: 'absolute',
    // Sits above the footer (footer is ~88px: padding 24 + content ~32 + paddingBottom 32)
    bottom: 96,
    left: 12,
    right: 12,
    maxHeight: 220,
    backgroundColor: 'rgba(0,0,0,0.78)',
    borderRadius: 8,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.12)',
    padding: 10,
    zIndex: 50,
  },
  header: {
    color: 'rgba(255,255,255,0.35)',
    fontSize: 9,
    fontWeight: '700',
    letterSpacing: 1.2,
    marginBottom: 6,
    fontFamily: 'monospace',
  },
  waiting: {
    color: 'rgba(255,255,255,0.4)',
    fontSize: 11,
    fontFamily: 'monospace',
    fontStyle: 'italic',
  },
  scroll: {
    maxHeight: 130,
  },
  scrollContent: {
    gap: 1,
  },
  line: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 11,
    fontFamily: 'monospace',
    lineHeight: 17,
  },
  // The bottom 3 lines that the parser actually examined
  lineAnchor: {
    color: '#e8d87a',
  },
  summaryRow: {
    marginTop: 8,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: 'rgba(255,255,255,0.15)',
    paddingTop: 6,
  },
  summary: {
    fontSize: 11,
    fontFamily: 'monospace',
    fontWeight: '600',
  },
  summaryOk: {
    color: '#66bb6a',
  },
  summaryFail: {
    color: '#ef5350',
  },
});

const scanStyles = StyleSheet.create({
  successBadge: {
    position: 'absolute',
    top: 56,
    alignSelf: 'center',
    zIndex: 100,
    backgroundColor: 'rgba(30,180,100,0.92)',
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 100,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    elevation: 8,
  },
  successText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    letterSpacing: 0.2,
  },
  // (more styles added in later tasks)
});
