import { memo, useCallback, useState } from 'react';
import { Image, Modal, Pressable, StyleSheet, type StyleProp, type ImageStyle } from 'react-native';
import { cardImageTransform, type CardLike } from './cardDisplay';

type Props = {
  uri?: string;
  uriBack?: string;
  style: StyleProp<ImageStyle>;
  onPress?: () => void;
  resizeMode?: 'cover' | 'contain' | 'stretch' | 'center';
  thumb?: boolean;
  onReady?: () => void;
  // Controlled flip — lets the parent keep the inline image and textbox in sync.
  flipped?: boolean;
  onFlip?: () => void;
  // When set, tap rotates the single image by this many degrees instead of swapping to uriBack.
  rotateDeg?: number;
  // When provided, uriBack/rotateDeg are derived from the card's `layout` — callers don't need to thread them manually.
  card?: CardLike;
};

function PressableCardImageImpl({ uri, uriBack, style, onPress, resizeMode = 'cover', thumb = false, onReady, flipped: flippedProp, onFlip, rotateDeg, card }: Props) {
  if (card) {
    const t = cardImageTransform(card);
    uri = uri ?? t.uri;
    uriBack = uriBack ?? t.uriBack;
    rotateDeg = rotateDeg ?? t.rotateDeg;
  }
  const resolvedUri = uri ?? '';
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
  const canRotate = !!rotateDeg;
  // Tap-to-flip is also enabled for shared-image multi-face layouts (split / flip / adventure).
  const canFlip = hasBack || canRotate || controlled;
  const faceUri = flipped && hasBack ? (uriBack as string) : resolvedUri;
  const zoomUri = zoomFlipped && hasBack ? (uriBack as string) : resolvedUri;
  const displayUri = thumb ? faceUri.replace('/normal.', '/small.') : faceUri;
  const rotateStyle = canRotate && flipped ? { transform: [{ rotate: `${rotateDeg}deg` }] } : null;
  const zoomRotateStyle = canRotate && zoomFlipped ? { transform: [{ rotate: `${rotateDeg}deg` }] } : null;

  // onPress wins; tap-to-flip only when standalone (card detail hero).
  const tapHandler = onPress ?? (canFlip ? toggleFlip : undefined);

  return (
    <>
      <Modal visible={zoomed} transparent animationType="fade" onRequestClose={closeZoom}>
        <Pressable style={styles.overlay} onPress={closeZoom}>
          <Pressable onPress={hasBack || canRotate ? () => setZoomFlipped((f) => !f) : closeZoom}>
            <Image source={{ uri: zoomUri }} style={[styles.zoomImage, zoomRotateStyle]} resizeMode="contain" />
          </Pressable>
        </Pressable>
      </Modal>
      <Pressable onPress={tapHandler} onLongPress={openZoom}>
        <Image source={{ uri: displayUri }} style={[style, rotateStyle]} resizeMode={resizeMode} onLoadEnd={onReady} />
      </Pressable>
    </>
  );
}

export const PressableCardImage = memo(PressableCardImageImpl);

const styles = StyleSheet.create({
  overlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.85)', alignItems: 'center', justifyContent: 'center' },
  zoomImage: { width: 280, height: 390, borderRadius: 12 },
});
