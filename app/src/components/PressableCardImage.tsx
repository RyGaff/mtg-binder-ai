import { memo, useCallback, useState } from 'react';
import { Image, Modal, Pressable, StyleSheet, type StyleProp, type ImageStyle } from 'react-native';

type Props = {
  uri: string;
  uriBack?: string;
  style: StyleProp<ImageStyle>;
  onPress?: () => void;
  resizeMode?: 'cover' | 'contain' | 'stretch' | 'center';
  thumb?: boolean;
  onReady?: () => void;
  // Controlled flip — lets the parent keep the inline image and textbox in sync.
  flipped?: boolean;
  onFlip?: () => void;
};

function PressableCardImageImpl({ uri, uriBack, style, onPress, resizeMode = 'cover', thumb = false, onReady, flipped: flippedProp, onFlip }: Props) {
  const [zoomed, setZoomed] = useState(false);
  const [flippedInternal, setFlippedInternal] = useState(false);
  const [zoomFlipped, setZoomFlipped] = useState(false);
  const controlled = flippedProp !== undefined;
  const flipped = controlled ? flippedProp : flippedInternal;
  const openZoom = useCallback(() => { setZoomFlipped(false); setZoomed(true); }, []);
  const closeZoom = useCallback(() => setZoomed(false), []);
  const toggleFlip = useCallback(() => {
    if (controlled) onFlip?.();
    else setFlippedInternal((f) => !f);
  }, [controlled, onFlip]);

  const hasBack = !!uriBack;
  // Tap-to-flip is also enabled for shared-image multi-face layouts (split / flip / adventure).
  const canFlip = hasBack || controlled;
  const faceUri = flipped && hasBack ? (uriBack as string) : uri;
  const zoomUri = zoomFlipped && hasBack ? (uriBack as string) : uri;
  const displayUri = thumb ? faceUri.replace('/normal.', '/small.') : faceUri;

  // onPress wins; tap-to-flip only when standalone (card detail hero).
  const tapHandler = onPress ?? (canFlip ? toggleFlip : undefined);

  return (
    <>
      <Modal visible={zoomed} transparent animationType="fade" onRequestClose={closeZoom}>
        <Pressable style={styles.overlay} onPress={closeZoom}>
          <Pressable onPress={hasBack ? () => setZoomFlipped((f) => !f) : closeZoom}>
            <Image source={{ uri: zoomUri }} style={styles.zoomImage} resizeMode="contain" />
          </Pressable>
        </Pressable>
      </Modal>
      <Pressable onPress={tapHandler} onLongPress={openZoom}>
        <Image source={{ uri: displayUri }} style={style} resizeMode={resizeMode} onLoadEnd={onReady} />
      </Pressable>
    </>
  );
}

export const PressableCardImage = memo(PressableCardImageImpl);

const styles = StyleSheet.create({
  overlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.85)', alignItems: 'center', justifyContent: 'center' },
  zoomImage: { width: 280, height: 390, borderRadius: 12 },
});
