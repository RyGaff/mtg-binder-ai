import { memo } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { Image } from 'expo-image';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useTheme } from '../theme/useTheme';

type Props = {
  name: string;
  artCropUri: string;
  onBack: () => void;
  onMore: () => void;
};

function DeckHeroImpl({ name, artCropUri, onBack, onMore }: Props) {
  const t = useTheme();
  // Hero extends behind the status bar so art reaches the screen edge; chrome lives
  // in a bottom toolbar with the title, well clear of the battery/time area.
  const insets = useSafeAreaInsets();
  return (
    <View style={[s.wrap, { height: 200 + insets.top }]}>
      {artCropUri
        ? <Image source={artCropUri} style={s.fill} contentFit="cover" cachePolicy="memory-disk" recyclingKey={artCropUri} />
        : <View style={[s.fill, { backgroundColor: t.surface }]} />}
      <View style={s.scrim} />
      {/* Bottom toolbar: ← · deck name · ⋮ — single row anchored to the hero's bottom.
          Buttons are borderless ghost glyphs (no chip background) — the dark scrim plus
          a per-glyph drop shadow gives them all the contrast they need. */}
      <View style={s.bottomBar}>
        <Pressable onPress={onBack} hitSlop={12} style={s.iconBtn}>
          <Text style={s.iconText}>←</Text>
        </Pressable>
        <Text style={s.name} numberOfLines={2}>{name}</Text>
        <Pressable onPress={onMore} hitSlop={12} style={s.iconBtn}>
          <Text style={s.iconText}>⋮</Text>
        </Pressable>
      </View>
    </View>
  );
}

export const DeckHero = memo(DeckHeroImpl);

const s = StyleSheet.create({
  wrap: { position: 'relative', overflow: 'hidden' },
  fill: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 },
  scrim: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.5)' },
  bottomBar: {
    position: 'absolute', left: 12, right: 12, bottom: 10,
    flexDirection: 'row', alignItems: 'center', gap: 10,
  },
  // Ghost button: 36pt tap target, no chip background, glyph-only. Padding via the
  // 36×36 box keeps the touch area ≥ minimum without drawing a visible button shape.
  iconBtn: { width: 36, height: 36, alignItems: 'center', justifyContent: 'center' },
  // Glyph with a soft drop shadow so it stays legible on bright art_crops even though
  // there's no chip behind it. Slightly larger and lighter weight than the old chip
  // glyph (24/500 vs 22/600) — feels more native iOS.
  iconText: {
    color: '#fff', fontSize: 24, fontWeight: '500', lineHeight: 26,
    textShadowColor: 'rgba(0,0,0,0.7)', textShadowRadius: 6, textShadowOffset: { width: 0, height: 1 },
  },
  // Title fills the space between the two glyphs. Truncates at 2 lines.
  name: { flex: 1, color: '#fff', fontSize: 22, fontWeight: '800', textShadowColor: 'rgba(0,0,0,0.6)', textShadowRadius: 4, textShadowOffset: { width: 0, height: 1 } },
});
