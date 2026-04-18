import {
  View,
  Text,
  Image,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  Animated,
  LayoutChangeEvent,
  useWindowDimensions,
} from 'react-native';
import { Camera, useCameraDevice, useCameraPermission, useFrameProcessor } from 'react-native-vision-camera';
import type { Frame } from 'react-native-vision-camera';
import { useSharedValue, useRunOnJS } from 'react-native-worklets-core';
import * as ImagePicker from 'expo-image-picker';
import { useRef, useState, useCallback, useEffect } from 'react';
import { useRouter } from 'expo-router';
import { scanCard } from '../../src/scanner/ocr';
import { upsertCard } from '../../src/db/cards';
import { useStore } from '../../src/store/useStore';
import { useTheme } from '../../src/theme/useTheme';
import type { CardCorners } from '../../modules/card-detector/src';
import { detectCardCornersInFrame, initCardDetectorPlugin, CARD_CONFIDENCE_STABLE } from '../../modules/card-detector/src';
import type { FrameProcessorPlugin } from 'react-native-vision-camera';

type ScanPhase =
  | { status: 'idle' }
  | { status: 'scanning' }
  | { status: 'fetching' }
  | { status: 'error'; message: string };

type DetectionInfo = {
  corners: CardCorners;
  imageW:  number;
  imageH:  number;
};

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Map a normalized image point to screen space given a view rect and resize mode. */
function imageToScreen(
  nx: number, ny: number,
  imageW: number, imageH: number,
  viewW: number, viewH: number,
  cover: boolean,
): { x: number; y: number } {
  const scale = cover
    ? Math.max(viewW / imageW, viewH / imageH)
    : Math.min(viewW / imageW, viewH / imageH);
  const dispW = imageW * scale;
  const dispH = imageH * scale;
  const offX  = (viewW - dispW) / 2;
  const offY  = (viewH - dispH) / 2;
  return { x: offX + nx * dispW, y: offY + ny * dispH };
}

/** Thin absolute line between two screen points. */
function OverlayLine({
  from, to, color, thickness = 2,
}: { from: {x:number;y:number}; to: {x:number;y:number}; color: string; thickness?: number }) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.sqrt(dx * dx + dy * dy);
  const angle = Math.atan2(dy, dx) * (180 / Math.PI);
  const cx = (from.x + to.x) / 2;
  const cy = (from.y + to.y) / 2;
  return (
    <View
      pointerEvents="none"
      style={{
        position: 'absolute',
        left: cx - len / 2,
        top: cy - thickness / 2,
        width: len,
        height: thickness,
        backgroundColor: color,
        transform: [{ rotate: `${angle}deg` }],
      }}
    />
  );
}

/**
 * Overlays:
 *  - Cyan quad showing the detected card boundary
 *  - Gold rect showing the bottom-left OCR crop region
 *  - Cyan rect showing the name OCR crop region
 */
function CardDetectionOverlay({
  detection,
  viewW,
  viewH,
  cover,
  ocrText,
  blText,
  activeRegion,
}: {
  detection: DetectionInfo;
  viewW: number;
  viewH: number;
  cover: boolean;
  ocrText: string | null;
  blText: string | null;
  activeRegion: 'bl' | 'name' | null;
}) {
  const { corners, imageW, imageH } = detection;

  const proj = (nx: number, ny: number) =>
    imageToScreen(nx, ny, imageW, imageH, viewW, viewH, cover);

  const tl = proj(corners.topLeft.x,     corners.topLeft.y);
  const tr = proj(corners.topRight.x,    corners.topRight.y);
  const br = proj(corners.bottomRight.x, corners.bottomRight.y);
  const bl = proj(corners.bottomLeft.x,  corners.bottomLeft.y);

  // OCR crop: bottom-left 25% width × 15% height of card (same math as ocr.ts)
  const cardW = Math.sqrt((br.x - bl.x) ** 2 + (br.y - bl.y) ** 2);
  const cardH = Math.sqrt((bl.x - tl.x) ** 2 + (bl.y - tl.y) ** 2);

  // Unit vectors along card bottom edge and up the left edge
  const rightVec = cardW > 0
    ? { x: (br.x - bl.x) / cardW, y: (br.y - bl.y) / cardW }
    : { x: 1, y: 0 };
  const upVec = cardH > 0
    ? { x: (tl.x - bl.x) / cardH, y: (tl.y - bl.y) / cardH }
    : { x: 0, y: -1 };

  const ocrW = 0.45 * cardW;
  const ocrH = 0.075 * cardH;

  // Bottom of OCR region = detected bottom-left corner
  const ocrBL = { x: bl.x, y: bl.y };
  const ocrBR = { x: ocrBL.x + rightVec.x * ocrW, y: ocrBL.y + rightVec.y * ocrW };
  const ocrTL = { x: ocrBL.x + upVec.x * ocrH,    y: ocrBL.y + upVec.y * ocrH };
  const ocrTR = { x: ocrTL.x + rightVec.x * ocrW,  y: ocrTL.y + rightVec.y * ocrW };

  // Name OCR region: 65% wide × 12% tall, anchored at topLeft
  const nameW = 0.65 * cardW;
  const nameH = 0.12 * cardH;
  const nameDownVec = cardH > 0
    ? { x: (bl.x - tl.x) / cardH, y: (bl.y - tl.y) / cardH }
    : { x: 0, y: 1 };

  const nameTL = { x: tl.x, y: tl.y };
  const nameTR = { x: nameTL.x + rightVec.x * nameW, y: nameTL.y + rightVec.y * nameW };
  const nameBL = { x: nameTL.x + nameDownVec.x * nameH, y: nameTL.y + nameDownVec.y * nameH };
  const nameBR = { x: nameTR.x + nameDownVec.x * nameH, y: nameTR.y + nameDownVec.y * nameH };

  const blColor = activeRegion === 'bl'
    ? 'rgba(255,215,0,1.0)'
    : 'rgba(255,215,0,0.5)';
  const nameColor = activeRegion === 'name'
    ? 'rgba(0,220,220,1.0)'
    : 'rgba(0,220,220,0.3)';

  // Axis-aligned bounding rectangles are more robust than rotated lines.
  // The shape/size tells you exactly what the detector picked.
  const quadBB = {
    left:   Math.min(tl.x, tr.x, br.x, bl.x),
    top:    Math.min(tl.y, tr.y, br.y, bl.y),
    width:  Math.max(tl.x, tr.x, br.x, bl.x) - Math.min(tl.x, tr.x, br.x, bl.x),
    height: Math.max(tl.y, tr.y, br.y, bl.y) - Math.min(tl.y, tr.y, br.y, bl.y),
  };
  const blBB = {
    left:   Math.min(ocrTL.x, ocrTR.x, ocrBR.x, ocrBL.x),
    top:    Math.min(ocrTL.y, ocrTR.y, ocrBR.y, ocrBL.y),
    width:  Math.max(ocrTL.x, ocrTR.x, ocrBR.x, ocrBL.x) - Math.min(ocrTL.x, ocrTR.x, ocrBR.x, ocrBL.x),
    height: Math.max(ocrTL.y, ocrTR.y, ocrBR.y, ocrBL.y) - Math.min(ocrTL.y, ocrTR.y, ocrBR.y, ocrBL.y),
  };
  const nameBB = {
    left:   Math.min(nameTL.x, nameTR.x, nameBR.x, nameBL.x),
    top:    Math.min(nameTL.y, nameTR.y, nameBR.y, nameBL.y),
    width:  Math.max(nameTL.x, nameTR.x, nameBR.x, nameBL.x) - Math.min(nameTL.x, nameTR.x, nameBR.x, nameBL.x),
    height: Math.max(nameTL.y, nameTR.y, nameBR.y, nameBL.y) - Math.min(nameTL.y, nameTR.y, nameBR.y, nameBL.y),
  };

  return (
    <>
      {/* Detected card — axis-aligned bounding box of the detected quad */}
      <View
        pointerEvents="none"
        style={{
          position: 'absolute',
          left: quadBB.left,
          top: quadBB.top,
          width: quadBB.width,
          height: quadBB.height,
          borderWidth: 3,
          borderColor: 'rgba(0,220,220,0.95)',
        }}
      />

      {/* Bottom-left OCR region */}
      <View
        pointerEvents="none"
        style={{
          position: 'absolute',
          left: blBB.left,
          top: blBB.top,
          width: blBB.width,
          height: blBB.height,
          borderWidth: 2,
          borderColor: blColor,
        }}
      />

      {/* Name OCR region */}
      <View
        pointerEvents="none"
        style={{
          position: 'absolute',
          left: nameBB.left,
          top: nameBB.top,
          width: nameBB.width,
          height: nameBB.height,
          borderWidth: 2,
          borderColor: nameColor,
        }}
      />

      {/* BL crop text — always shown next to the gold box */}
      {blText != null && (
        <View
          pointerEvents="none"
          style={{
            position: 'absolute',
            left: ocrTR.x + 6,
            top: (ocrTL.y + ocrBL.y) / 2 - 30,
            maxWidth: 170,
            backgroundColor: 'rgba(0,0,0,0.72)',
            paddingHorizontal: 6,
            paddingVertical: 4,
            borderRadius: 4,
            borderLeftWidth: 2,
            borderLeftColor: 'rgba(255,215,0,0.85)',
          }}
        >
          <Text style={{ color: 'rgba(255,215,0,0.9)', fontSize: 9, fontWeight: '700', letterSpacing: 0.8, marginBottom: 2 }}>
            BL CROP
          </Text>
          <Text style={{ color: 'rgba(255,255,255,0.85)', fontSize: 10, fontFamily: 'monospace', lineHeight: 14 }}>
            {blText || '(empty)'}
          </Text>
        </View>
      )}
      {/* Winning-strategy text — shown above card top edge when different from blText */}
      {ocrText != null && ocrText !== blText && (
        <View
          pointerEvents="none"
          style={{
            position: 'absolute',
            left: tl.x,
            top: tl.y - 52,
            maxWidth: 200,
            backgroundColor: 'rgba(0,0,0,0.72)',
            paddingHorizontal: 6,
            paddingVertical: 4,
            borderRadius: 4,
            borderLeftWidth: 2,
            borderLeftColor: 'rgba(0,220,220,0.85)',
          }}
        >
          <Text style={{ color: 'rgba(0,220,220,0.9)', fontSize: 9, fontWeight: '700', letterSpacing: 0.8, marginBottom: 2 }}>
            NAME CROP
          </Text>
          <Text style={{ color: 'rgba(255,255,255,0.85)', fontSize: 10, fontFamily: 'monospace', lineHeight: 14 }}>
            {ocrText}
          </Text>
        </View>
      )}
    </>
  );
}

// ── Frame processor helpers (worklets) ────────────────────────────────────────

const SMOOTH_ALPHA = 0.35;
const STABLE_THRESHOLD = 0.015;
const STABLE_FRAMES = 8;
const MISS_FRAMES = 20;

function emaCorners(prev: CardCorners, next: CardCorners): CardCorners {
  'worklet';
  const lerp = (a: number, b: number) => a + SMOOTH_ALPHA * (b - a);
  return {
    topLeft:     { x: lerp(prev.topLeft.x,     next.topLeft.x),     y: lerp(prev.topLeft.y,     next.topLeft.y) },
    topRight:    { x: lerp(prev.topRight.x,    next.topRight.x),    y: lerp(prev.topRight.y,    next.topRight.y) },
    bottomRight: { x: lerp(prev.bottomRight.x, next.bottomRight.x), y: lerp(prev.bottomRight.y, next.bottomRight.y) },
    bottomLeft:  { x: lerp(prev.bottomLeft.x,  next.bottomLeft.x),  y: lerp(prev.bottomLeft.y,  next.bottomLeft.y) },
    confidence:  next.confidence,
    rectifiedUri: next.rectifiedUri,
  };
}

function cornersAreStable(a: CardCorners, b: CardCorners, threshold: number): boolean {
  'worklet';
  const pts = ['topLeft', 'topRight', 'bottomRight', 'bottomLeft'] as const;
  for (const k of pts) {
    if (Math.abs(a[k].x - b[k].x) > threshold) return false;
    if (Math.abs(a[k].y - b[k].y) > threshold) return false;
  }
  return true;
}

// ── Sub-components ────────────────────────────────────────────────────────────

function parsePrice(pricesJson: string): string {
  try {
    const p = JSON.parse(pricesJson) as Record<string, string | undefined>;
    const val = p.usd ?? p.usd_foil;
    return val ? `$${val}` : '—';
  } catch {
    return '—';
  }
}

function OcrDebugPanel({
  phase,
  strategy,
}: {
  phase:    ScanPhase;
  strategy: 'set_number' | 'name' | null;
}) {
  if (phase.status === 'idle') return null;

  const strategyLabel = strategy === 'set_number'
    ? 'SET + NUMBER'
    : strategy === 'name'
    ? 'NAME FALLBACK'
    : null;

  const statusLabel = (() => {
    switch (phase.status) {
      case 'scanning': return 'Detecting card…';
      case 'fetching': return 'Looking up card…';
      case 'error':    return phase.message;
      default:         return null;
    }
  })();

  return (
    <View style={debugStyles.panel}>
      <Text style={debugStyles.header}>OCR DEBUG</Text>
      {strategyLabel && (
        <Text style={debugStyles.strategy}>{strategyLabel}</Text>
      )}
      {statusLabel && (
        <Text style={[
          debugStyles.status,
          phase.status === 'error' ? debugStyles.statusError : debugStyles.statusOk,
        ]}>
          {statusLabel}
        </Text>
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

// ── Main screen ───────────────────────────────────────────────────────────────

export default function ScanScreen() {
  const theme = useTheme();
  const router = useRouter();
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const { width: winW, height: winH } = useWindowDimensions();

  const [phase, setPhase] = useState<ScanPhase>({ status: 'idle' });
  const [scanStrategy, setScanStrategy] = useState<'set_number' | 'name' | null>(null);
  const [ocrText, setOcrText] = useState<string | null>(null);
  const [blText, setBlText] = useState<string | null>(null);
  const [detection, setDetection] = useState<DetectionInfo | null>(null);
  const [overlayLayout, setOverlayLayout] = useState<{ width: number; height: number }>(() => ({ width: winW, height: winH }));
  const [pickedImageUri, setPickedImageUri] = useState<string | null>(null);
  const [successCard, setSuccessCard] = useState<string | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);
  const [activeRegion, setActiveRegion] = useState<'bl' | 'name' | null>(null);
  const [cardPlugin, setCardPlugin] = useState<FrameProcessorPlugin | null>(null);
  const [debugFrames, setDebugFrames] = useState(0);
  const [debugHits, setDebugHits] = useState(0);
  const [debugLastConf, setDebugLastConf] = useState<number | null>(null);
  const [debugPixFmt, setDebugPixFmt] = useState<string | null>(null);
  const [debugFrameSize, setDebugFrameSize] = useState<{ w: number; h: number } | null>(null);
  const [debugStats, setDebugStats] = useState<{
    medianLuma: number; edgePixels: number;
    contoursTotal: number; passed4Vertex: number; passedMinArea: number;
    passedConvex: number; passedAngles: number; passedAR: number;
  } | null>(null);
  const [debugDetSet, setDebugDetSet] = useState(0);
  const [debugDetCleared, setDebugDetCleared] = useState(0);
  const cameraRef = useRef<Camera>(null);

  useEffect(() => {
    const p = initCardDetectorPlugin();
    console.log('[CardDetector] plugin init:', p == null ? 'NULL' : 'OK');
    setCardPlugin(p);
  }, []);

  const { setLastScannedId, addRecentScan, recentScans } = useStore();

  // Worklet-safe shared values — written on the frame processor thread
  const smoothedCornersWv = useSharedValue<CardCorners | null>(null);
  const stableCount = useSharedValue(0);
  const missCount = useSharedValue(0);
  const isCapturing = useSharedValue(false);

  const PANEL_MAX_HEIGHT = 320;
  const panelHeightAnim = useRef(new Animated.Value(0)).current;

  const openPanel = useCallback(() => {
    setPanelOpen(true);
    Animated.timing(panelHeightAnim, {
      toValue: PANEL_MAX_HEIGHT,
      duration: 250,
      useNativeDriver: false,
    }).start();
  }, [panelHeightAnim]);

  const closePanel = useCallback(() => {
    Animated.timing(panelHeightAnim, {
      toValue: 0,
      duration: 200,
      useNativeDriver: false,
    }).start(() => setPanelOpen(false));
  }, [panelHeightAnim]);

  // Stops any in-flight capture and clears detection state. Call before
  // navigating away or switching to the photo library.
  const stopScanning = useCallback(() => {
    isCapturing.value = false;
    smoothedCornersWv.value = null;
    stableCount.value = 0;
    missCount.value = 0;
    setDetection(null);
  }, [isCapturing, smoothedCornersWv, stableCount, missCount]);

  const triggerOcr = useCallback(async () => {
    if (!cameraRef.current) { isCapturing.value = false; return; }
    try {
      setPhase({ status: 'scanning' });
      setOcrText(null);
      setBlText(null);
      const photo = await cameraRef.current.takePhoto({ qualityPrioritization: 'speed' });
      const uri = `file://${photo.path}`;
      const result = await scanCard(uri, (p) => {
        if (p.step === 'corners_detected') {
          setDetection({ corners: p.corners, imageW: p.imageW, imageH: p.imageH });
          setActiveRegion('bl');
        } else if (p.step === 'bl_ocr_done') {
          setBlText(p.blText);
          setActiveRegion('name');
        } else if (p.step === 'name_ocr_done') {
          setOcrText(p.nameText);
          setActiveRegion(null);
        }
      }, { width: photo.width, height: photo.height });
      upsertCard(result.card);
      addRecentScan(result.card);
      setLastScannedId(result.card.scryfall_id);
      setScanStrategy(result.strategy);
      setOcrText(result.ocrText);
      setSuccessCard(result.card.name);
      setActiveRegion(null);
      setPhase({ status: 'idle' });
      await new Promise<void>(resolve => setTimeout(resolve, 1500));
      setSuccessCard(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error';
      console.warn('[scan]', msg);
      setActiveRegion(null);
      setPhase({ status: 'error', message: msg });
    } finally {
      isCapturing.value = false;
    }
  }, [addRecentScan, setLastScannedId, isCapturing]);

  const jsSetDetection = useRunOnJS(
    (d: DetectionInfo | null) => {
      setDetection(d);
      if (d) setDebugDetSet(n => n + 1);
      else   setDebugDetCleared(n => n + 1);
    },
    [setDetection],
  );
  const jsTriggerOcr = useRunOnJS(triggerOcr, [triggerOcr]);
  const jsDebugTick = useRunOnJS(
    (
      hit: boolean,
      confidence: number | null,
      pixelFormat: string | null,
      frameW: number,
      frameH: number,
      stats: typeof debugStats,
    ) => {
      setDebugFrames(n => n + 1);
      if (hit) setDebugHits(n => n + 1);
      setDebugLastConf(confidence);
      if (pixelFormat) setDebugPixFmt(pixelFormat);
      if (frameW > 0 && frameH > 0) setDebugFrameSize({ w: frameW, h: frameH });
      if (stats) setDebugStats(stats);
    },
    [],
  );

  const frameProcessor = useFrameProcessor((frame: Frame) => {
    'worklet';
    if (cardPlugin == null) return;
    if (isCapturing.value) return;

    const { corners: raw, debug } = detectCardCornersInFrame(frame, cardPlugin);
    jsDebugTick(raw != null, raw?.confidence ?? null, debug.pixelFormat, debug.frameW, debug.frameH, debug.stats);

    if (raw) {
      const prev = smoothedCornersWv.value;
      const smoothed = prev ? emaCorners(prev, raw) : raw;
      const stable = prev ? cornersAreStable(prev, smoothed, STABLE_THRESHOLD) : false;

      smoothedCornersWv.value = smoothed;
      missCount.value = 0;

      // Stability gate: require confidence >= CARD_CONFIDENCE_STABLE to trigger OCR
      const meetsStableConf = raw.confidence >= CARD_CONFIDENCE_STABLE;
      stableCount.value = (stable && meetsStableConf) ? stableCount.value + 1 : 0;

      // Frame buffer comes from the native side in whatever orientation the camera
      // produced. When landscape (w > h), swap dims so imageToScreen maps into
      // the portrait view correctly.
      const landscape = frame.width > frame.height;
      jsSetDetection({
        corners: smoothed,
        imageW: landscape ? frame.height : frame.width,
        imageH: landscape ? frame.width : frame.height,
      });

      if (stableCount.value >= STABLE_FRAMES) {
        stableCount.value = 0;
        isCapturing.value = true;
        jsTriggerOcr();
      }
    } else {
      missCount.value += 1;
      if (missCount.value >= MISS_FRAMES) {
        smoothedCornersWv.value = null;
        missCount.value = 0;
        jsSetDetection(null);
      }
    }
  }, [cardPlugin, isCapturing, smoothedCornersWv, stableCount, missCount, jsTriggerOcr, jsSetDetection, jsDebugTick]);

  const pickFromLibrary = useCallback(async () => {
    isCapturing.value = false;
    smoothedCornersWv.value = null;
    const picked = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 1 });
    if (picked.canceled || !picked.assets[0]) return;

    const asset = picked.assets[0];
    setDetection(null);
    setPickedImageUri(asset.uri);
    setOcrText(null);
    setBlText(null);
    setPhase({ status: 'scanning' });

    try {
      const result = await scanCard(asset.uri, (p) => {
        if (p.step === 'corners_detected') {
          setDetection({ corners: p.corners, imageW: p.imageW, imageH: p.imageH });
          setActiveRegion('bl');
        } else if (p.step === 'bl_ocr_done') {
          setBlText(p.blText);
          setActiveRegion('name');
        } else if (p.step === 'name_ocr_done') {
          setOcrText(p.nameText);
          setActiveRegion(null);
        }
      });
      upsertCard(result.card);
      addRecentScan(result.card);
      setLastScannedId(result.card.scryfall_id);
      setScanStrategy(result.strategy);
      setOcrText(result.ocrText);
      setSuccessCard(result.card.name);
      setActiveRegion(null);
      setPhase({ status: 'idle' });
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error';
      console.warn('[scan] library:', msg);
      setActiveRegion(null);
      setPhase({ status: 'error', message: msg });
    }
  }, [addRecentScan, setLastScannedId, isCapturing, smoothedCornersWv]);

  const handleOverlayLayout = useCallback((e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setOverlayLayout({ width, height });
  }, []);

  if (!hasPermission) {
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

  if (!device) {
    return <View style={[styles.screen, { backgroundColor: theme.bg }]} />;
  }

  const statusLabel = (() => {
    switch (phase.status) {
      case 'idle':     return 'Point camera at a card';
      case 'scanning': return 'Detecting card…';
      case 'fetching': return 'Looking up card…';
      case 'error':    return null; // shown as toast instead
    }
  })();

  const buildOverlay = (cover: boolean) => (
    <View
      style={styles.overlay}
      pointerEvents="none"
      onLayout={handleOverlayLayout}
    >
      {/* RENDER TEST — always-visible yellow box. If you don't see this, child
          rendering is broken inside the Camera's overlay layer entirely. */}
      <View
        pointerEvents="none"
        style={{
          position: 'absolute',
          left: 20,
          top: 300,
          width: 120,
          height: 120,
          borderWidth: 4,
          borderColor: 'yellow',
          backgroundColor: 'rgba(255,255,0,0.2)',
        }}
      />

      {detection && (
        <CardDetectionOverlay
          detection={detection}
          viewW={overlayLayout.width}
          viewH={overlayLayout.height}
          cover={cover}
          ocrText={ocrText}
          blText={blText}
          activeRegion={activeRegion}
        />
      )}
      {statusLabel !== null && (
        <View style={styles.statusBadge}>
          <Text style={styles.statusText}>{statusLabel}</Text>
        </View>
      )}
    </View>
  );
  const cameraOverlay       = buildOverlay(true);
  const pickedImageOverlay  = buildOverlay(false);

  // Projected corner coords (for debug readout so we can see where the quad is drawing)
  const projTL = detection
    ? imageToScreen(detection.corners.topLeft.x, detection.corners.topLeft.y,
                    detection.imageW, detection.imageH,
                    overlayLayout.width, overlayLayout.height, true)
    : null;
  const projBR = detection
    ? imageToScreen(detection.corners.bottomRight.x, detection.corners.bottomRight.y,
                    detection.imageW, detection.imageH,
                    overlayLayout.width, overlayLayout.height, true)
    : null;

  return (
    <View style={styles.screen}>
      {/* Debug overlay — plugin, frames, hits, conf, pixel fmt, per-stage stats */}
      <View style={debugStyles.devBadge} pointerEvents="none">
        <Text style={debugStyles.devText}>
          {'plugin: ' + (cardPlugin == null ? 'NULL' : 'OK') +
           '\nframes: ' + debugFrames +
           '\nhits:   ' + debugHits +
           '\ndSet:   ' + debugDetSet +
           '\ndClr:   ' + debugDetCleared +
           '\ndet?:   ' + (detection ? 'YES' : 'no') +
           '\nconf:   ' + (debugLastConf == null ? '—' : debugLastConf.toFixed(3)) +
           '\nfmt:    ' + (debugPixFmt ?? '—') +
           '\nsize:   ' + (debugFrameSize ? debugFrameSize.w + 'x' + debugFrameSize.h : '—') +
           '\nview:   ' + Math.round(overlayLayout.width) + 'x' + Math.round(overlayLayout.height) +
           (projTL ? '\ntl:     ' + Math.round(projTL.x) + ',' + Math.round(projTL.y) : '') +
           (projBR ? '\nbr:     ' + Math.round(projBR.x) + ',' + Math.round(projBR.y) : '') +
           (debugStats
              ? '\nmedLum: ' + debugStats.medianLuma +
                '\nedges:  ' + debugStats.edgePixels +
                '\ncont:   ' + debugStats.contoursTotal +
                '\n4vert:  ' + debugStats.passed4Vertex +
                '\narea:   ' + debugStats.passedMinArea +
                '\nconv:   ' + debugStats.passedConvex +
                '\nang:    ' + debugStats.passedAngles +
                '\nAR:     ' + debugStats.passedAR
              : '')}
        </Text>
      </View>

      {/* Error toast — absolute, pinned to top */}
      {phase.status === 'error' && (
        <ErrorToast message={phase.message} />
      )}

      {successCard !== null && (
        <View style={scanStyles.successBadge} pointerEvents="none">
          <Text style={scanStyles.successText}>✓ {successCard}</Text>
        </View>
      )}

      {/* Floating recent-scans button */}
      <TouchableOpacity
        style={[
          scanStyles.recentBtn,
          recentScans.length === 0 && scanStyles.recentBtnDisabled,
        ]}
        onPress={panelOpen ? closePanel : openPanel}
        disabled={recentScans.length === 0}
        activeOpacity={0.75}
      >
        <Text style={scanStyles.recentBtnIcon}>⏱</Text>
        {recentScans.length > 0 && (
          <View style={scanStyles.recentBadge}>
            <Text style={scanStyles.recentBadgeText}>{recentScans.length}</Text>
          </View>
        )}
      </TouchableOpacity>

      {/* Expand-down recent scans panel */}
      <Animated.View
        style={[
          scanStyles.panel,
          {
            height: panelHeightAnim,
            opacity: panelHeightAnim.interpolate({
              inputRange: [0, 1],
              outputRange: [0, 1],
              extrapolate: 'clamp',
            }),
          },
        ]}
        pointerEvents={panelOpen ? 'auto' : 'none'}
      >
        <FlatList
          data={recentScans}
          keyExtractor={(item) => item.scryfall_id}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={scanStyles.row}
              onPress={() => { closePanel(); stopScanning(); router.push(`/card/${item.scryfall_id}`); }}
              activeOpacity={0.7}
            >
              <Image
                source={{ uri: item.image_uri }}
                style={scanStyles.thumb}
                resizeMode="cover"
              />
              <View style={scanStyles.rowInfo}>
                <Text style={scanStyles.rowName} numberOfLines={1}>{item.name}</Text>
                <Text style={scanStyles.rowSet}>
                  {item.set_code.toUpperCase()} · #{item.collector_number}
                </Text>
              </View>
              <Text style={scanStyles.rowPrice}>{parsePrice(item.prices)}</Text>
            </TouchableOpacity>
          )}
          ItemSeparatorComponent={() => <View style={scanStyles.separator} />}
        />
      </Animated.View>

      {pickedImageUri ? (
        <View style={styles.fullScreenImageContainer}>
          <Image source={{ uri: pickedImageUri }} style={styles.fullScreenImage} resizeMode="contain" />
          {pickedImageOverlay}
          {/* OCR debug panel inside the fullscreen container so it's unambiguously above the image layer */}
          <OcrDebugPanel phase={phase} strategy={scanStrategy} />
        </View>
      ) : (
        <>
          {/* Camera + overlay share one flex:1 parent so the overlay
              covers exactly the camera's area (not the header/footer). */}
          <View style={{ flex: 1 }}>
            <Camera
              ref={cameraRef}
              style={StyleSheet.absoluteFillObject}
              device={device}
              isActive={!pickedImageUri}
              frameProcessor={frameProcessor}
              photo={true}
              pixelFormat="yuv"
            />
            {cameraOverlay}
          </View>
          {/* OCR debug panel — absolute, bottom of screen, above footer */}
          <OcrDebugPanel phase={phase} strategy={scanStrategy} />
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
          activeOpacity={0.9}
        >
          <Text style={[styles.pillText, { color: theme.text }]}>Photo Library</Text>
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
    ...StyleSheet.absoluteFillObject,
  },

  // Error toast pinned to top
  errorToast: {
    position: 'absolute',
    top: 12,
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

const debugStyles = StyleSheet.create({
  devBadge: {
    position: 'absolute',
    top: 60,
    left: 10,
    zIndex: 999,
    backgroundColor: 'rgba(0,0,0,0.82)',
    padding: 6,
    borderRadius: 6,
  },
  devText: {
    color: '#0f0',
    fontSize: 11,
    fontFamily: 'monospace',
    lineHeight: 14,
  },
  panel: {
    position: 'absolute',
    bottom: 96,
    left: 12,
    right: 12,
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
  strategy: {
    color: '#4ecdc4',
    fontSize: 11,
    fontWeight: '700' as const,
    fontFamily: 'monospace',
    marginBottom: 4,
    letterSpacing: 0.8,
  },
  status: {
    fontSize: 11,
    fontFamily: 'monospace',
  },
  statusOk: {
    color: 'rgba(255,255,255,0.6)',
  },
  statusError: {
    color: '#ef5350',
    fontWeight: '600' as const,
  },
});

const scanStyles = StyleSheet.create({
  successBadge: {
    position: 'absolute',
    top: 12,
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
  recentBtn: {
    position: 'absolute',
    top: 25,
    right: 16,
    zIndex: 50,
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0,0,0,0.65)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  recentBtnDisabled: {
    opacity: 0.35,
  },
  recentBtnIcon: {
    fontSize: 20,
  },
  recentBadge: {
    position: 'absolute',
    top: -4,
    right: -4,
    minWidth: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: '#4ecdc4',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 3,
  },
  recentBadgeText: {
    color: '#000',
    fontSize: 10,
    fontWeight: '700',
  },
  panel: {
    position: 'absolute',
    top: 77,
    left: 12,
    right: 12,
    zIndex: 49,
    backgroundColor: 'rgba(0,0,0,0.77)',
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.12)',
    overflow: 'hidden',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 10,
    gap: 12,
  },
  thumb: {
    width: 48,
    height: 68,
    borderRadius: 4,
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  rowInfo: {
    flex: 1,
    gap: 3,
  },
  rowName: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  rowSet: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 11,
  },
  rowPrice: {
    color: '#4ecdc4',
    fontSize: 13,
    fontWeight: '500',
    minWidth: 44,
    textAlign: 'right',
  },
  separator: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: 'rgba(255,255,255,0.1)',
    marginLeft: 72,
  },
});
