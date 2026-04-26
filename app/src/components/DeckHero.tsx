import { memo } from 'react';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { useTheme } from '../theme/useTheme';

type Props = {
  name: string;
  artCropUri: string;
  onBack: () => void;
  onMore: () => void;
};

function DeckHeroImpl({ name, artCropUri, onBack, onMore }: Props) {
  const t = useTheme();
  return (
    <View style={s.wrap}>
      {artCropUri
        ? <Image source={{ uri: artCropUri }} style={s.fill} resizeMode="cover" />
        : <View style={[s.fill, { backgroundColor: t.surface }]} />}
      <View style={[s.scrim, { backgroundColor: t.bg }]} />
      <Pressable onPress={onBack} hitSlop={8} style={s.back}>
        <Text style={s.iconText}>←</Text>
      </Pressable>
      <Pressable onPress={onMore} hitSlop={8} style={s.more}>
        <Text style={s.iconText}>⋮</Text>
      </Pressable>
      <Text style={s.name} numberOfLines={2}>{name}</Text>
    </View>
  );
}

export const DeckHero = memo(DeckHeroImpl);

const s = StyleSheet.create({
  wrap: { height: 150, position: 'relative', overflow: 'hidden' },
  fill: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 },
  scrim: { position: 'absolute', left: 0, right: 0, bottom: 0, top: 0, opacity: 0.55 },
  back: { position: 'absolute', top: 14, left: 14, padding: 4 },
  more: { position: 'absolute', top: 14, right: 14, padding: 4 },
  iconText: { color: '#fff', fontSize: 22, fontWeight: '600', lineHeight: 24 },
  name: { position: 'absolute', left: 14, right: 14, bottom: 12, color: '#fff', fontSize: 19, fontWeight: '800' },
});
