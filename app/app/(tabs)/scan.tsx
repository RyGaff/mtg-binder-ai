import {
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  StyleSheet,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useRef, useState } from 'react';
import { useRouter } from 'expo-router';
import { scanCardImage } from '../../src/scanner/ocr';
import { fetchCardBySetNumber } from '../../src/api/scryfall';
import { upsertCard } from '../../src/db/cards';
import { useStore } from '../../src/store/useStore';

type ScanState = 'idle' | 'scanning' | 'error';

export default function ScanScreen() {
  const router = useRouter();
  const [permission, requestPermission] = useCameraPermissions();
  const [scanState, setScanState] = useState<ScanState>('idle');
  const cameraRef = useRef<CameraView>(null);
  const { setLastScannedId } = useStore();

  if (!permission) {
    return <View style={styles.screen} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.permText}>
          Camera access is needed to scan cards.
        </Text>
        <TouchableOpacity style={styles.btn} onPress={requestPermission}>
          <Text style={styles.btnText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const handleCapture = async () => {
    if (!cameraRef.current || scanState === 'scanning') return;
    setScanState('scanning');
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.8 });
      if (!photo) throw new Error('No photo captured');

      const parsed = await scanCardImage(photo.uri);
      if (!parsed) {
        setScanState('error');
        Alert.alert(
          'Could not read card',
          'Try again, or use Search to find the card manually.',
          [
            { text: 'Try Again', onPress: () => setScanState('idle') },
            { text: 'Search', onPress: () => router.push('/search') },
          ]
        );
        return;
      }

      const card = await fetchCardBySetNumber(parsed.setCode, parsed.collectorNumber);
      upsertCard(card);
      setLastScannedId(card.scryfall_id);
      setScanState('idle');
      router.push(`/card/${card.scryfall_id}`);
    } catch {
      setScanState('error');
      Alert.alert('Error', 'Failed to identify card. Please try again.', [
        { text: 'OK', onPress: () => setScanState('idle') },
      ]);
    }
  };

  return (
    <View style={styles.screen}>
      <CameraView ref={cameraRef} style={styles.camera} facing="back">
        <View style={styles.overlay}>
          <View style={styles.targetBox} />
          <Text style={styles.hint}>
            Align the bottom-left corner of the card
          </Text>
        </View>
      </CameraView>
      <View style={styles.footer}>
        <TouchableOpacity
          style={styles.captureBtn}
          onPress={handleCapture}
          disabled={scanState === 'scanning'}
        >
          {scanState === 'scanning' ? (
            <ActivityIndicator color="#000" />
          ) : (
            <View style={styles.captureInner} />
          )}
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.searchFallback}
          onPress={() => router.push('/search')}
        >
          <Text style={styles.searchFallbackText}>Search instead</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#000' },
  center: {
    flex: 1,
    backgroundColor: '#111318',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
    padding: 32,
  },
  permText: { color: '#ccc', textAlign: 'center', fontSize: 15 },
  btn: {
    backgroundColor: '#4ecdc4',
    borderRadius: 8,
    paddingHorizontal: 24,
    paddingVertical: 12,
  },
  btnText: { color: '#fff', fontWeight: '700' },
  camera: { flex: 1 },
  overlay: {
    flex: 1,
    alignItems: 'flex-start',
    justifyContent: 'flex-end',
    padding: 40,
  },
  targetBox: {
    width: 120,
    height: 50,
    borderWidth: 2,
    borderColor: '#4ecdc4',
    borderRadius: 4,
    marginBottom: 12,
  },
  hint: {
    color: '#fff',
    fontSize: 12,
    backgroundColor: '#0009',
    padding: 6,
    borderRadius: 4,
  },
  footer: {
    backgroundColor: '#000',
    padding: 24,
    alignItems: 'center',
    gap: 16,
  },
  captureBtn: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  captureInner: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: '#fff',
    borderWidth: 2,
    borderColor: '#000',
  },
  searchFallback: { padding: 8 },
  searchFallbackText: { color: '#aaa', fontSize: 13 },
});
