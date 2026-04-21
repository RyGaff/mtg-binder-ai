import { useState } from 'react';
import { Image, Modal, Pressable, View, StyleSheet, type StyleProp, type ImageStyle } from 'react-native';

type Props = {
  uri: string;
  style: StyleProp<ImageStyle>;
  onPress?: () => void;
  resizeMode?: 'cover' | 'contain' | 'stretch' | 'center';
};

export function PressableCardImage({ uri, style, onPress, resizeMode = 'cover' }: Props) {
  const [zoomed, setZoomed] = useState(false);

  return (
    <>
      <Modal visible={zoomed} transparent animationType="fade">
        <Pressable style={styles.overlay} onPress={() => setZoomed(false)}>
          <Image source={{ uri }} style={styles.zoomImage} resizeMode="contain" />
        </Pressable>
      </Modal>

      <Pressable
        onPress={onPress}
        onLongPress={() => setZoomed(true)}
      >
        <Image source={{ uri }} style={style} resizeMode={resizeMode} />
      </Pressable>
    </>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.85)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  zoomImage: { width: 280, height: 390, borderRadius: 12 },
});
