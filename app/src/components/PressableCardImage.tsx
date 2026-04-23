import { memo, useCallback, useState } from 'react';
import { Image, Modal, Pressable, StyleSheet, type StyleProp, type ImageStyle } from 'react-native';

type Props = {
  uri: string;
  style: StyleProp<ImageStyle>;
  onPress?: () => void;
  resizeMode?: 'cover' | 'contain' | 'stretch' | 'center';
};

function PressableCardImageImpl({ uri, style, onPress, resizeMode = 'cover' }: Props) {
  const [zoomed, setZoomed] = useState(false);
  const openZoom = useCallback(() => setZoomed(true), []);
  const closeZoom = useCallback(() => setZoomed(false), []);

  return (
    <>
      <Modal visible={zoomed} transparent animationType="fade" onRequestClose={closeZoom}>
        <Pressable style={styles.overlay} onPress={closeZoom}>
          <Image source={{ uri }} style={styles.zoomImage} resizeMode="contain" />
        </Pressable>
      </Modal>

      <Pressable onPress={onPress} onLongPress={openZoom}>
        <Image source={{ uri }} style={style} resizeMode={resizeMode} />
      </Pressable>
    </>
  );
}

export const PressableCardImage = memo(PressableCardImageImpl);

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.85)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  zoomImage: { width: 280, height: 390, borderRadius: 12 },
});
