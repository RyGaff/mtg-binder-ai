import { memo } from 'react';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';
import type { DeckWithMeta } from '../db/decks';
import { useTheme } from '../theme/useTheme';
import { MANA_TINT, manaGlyph } from '../utils/mana';

type Props = {
  deck: DeckWithMeta;
  mode: 'banner' | 'compact';
  active: boolean;
  onPress: (id: number) => void;
  onLongPress: (deck: DeckWithMeta) => void;
  onMore: (deck: DeckWithMeta) => void;
};

const DAY = 24 * 3600 * 1000;
const rel = (ts: number) => {
  const d = Math.max(0, Date.now() - ts);
  if (d < DAY) return 'today';
  if (d < 7 * DAY) return `${Math.floor(d / DAY)}d ago`;
  if (d < 30 * DAY) return `${Math.floor(d / (7 * DAY))}w ago`;
  return `${Math.floor(d / (30 * DAY))}mo ago`;
};

function DeckRowImpl({ deck, mode, active, onPress, onLongPress, onMore }: Props) {
  const t = useTheme();
  const banner = mode === 'banner';
  const accent = deck.color_identity[0] ? MANA_TINT[deck.color_identity[0]] : t.accent;

  return (
    <Pressable
      onPress={() => onPress(deck.id)}
      onLongPress={() => onLongPress(deck)}
      style={({ pressed }) => [
        banner ? s.banner : s.compact,
        banner
          ? { borderColor: active ? t.accent : t.border, borderWidth: active ? 2 : 1 }
          : { borderBottomColor: t.border, backgroundColor: active ? t.surface : 'transparent' },
        pressed && { opacity: 0.85 },
      ]}
    >
      {banner && (deck.art_crop_uri
        ? <Image source={{ uri: deck.art_crop_uri }} style={s.fill} resizeMode="cover" />
        : <View style={[s.fill, { backgroundColor: accent }]} />)}
      {banner && <View style={[s.fill, { backgroundColor: t.surface, opacity: 0.55 }]} />}
      {!banner && <View style={{ width: 3, height: 32, borderRadius: 2, backgroundColor: accent }} />}

      <View style={s.body}>
        <Text style={[s.name, { color: t.text, fontSize: banner ? 15 : 14 }]} numberOfLines={1}>
          {deck.name}
        </Text>
        <View style={s.meta}>
          {banner && (
            <View style={[s.pill, { backgroundColor: t.surfaceAlt }]}>
              <Text style={[s.pillText, { color: t.textSecondary }]}>{deck.format}</Text>
            </View>
          )}
          {deck.color_identity.map((c) => (
            <Text key={c} style={{ fontFamily: 'Mana', color: MANA_TINT[c], fontSize: banner ? 13 : 12, lineHeight: 15 }}>
              {manaGlyph(c) ?? ''}
            </Text>
          ))}
          <Text style={{ fontSize: 11, color: t.textSecondary }} numberOfLines={1}>
            {banner ? `· ${deck.card_count} cards` : `${deck.format} · ${deck.card_count} cards · ${rel(deck.created_at)}`}
          </Text>
        </View>
      </View>

      <Pressable onPress={() => onMore(deck)} hitSlop={8} style={s.more}>
        <Text style={{ fontSize: 22, fontWeight: '700', color: t.textSecondary }}>⋮</Text>
      </Pressable>
    </Pressable>
  );
}

export const DeckRow = memo(DeckRowImpl);

const s = StyleSheet.create({
  banner: { height: 72, borderRadius: 12, overflow: 'hidden', flexDirection: 'row', alignItems: 'center' },
  compact: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 10, paddingHorizontal: 6, borderBottomWidth: StyleSheet.hairlineWidth },
  fill: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 },
  body: { flex: 1, paddingHorizontal: 14, gap: 4, minWidth: 0 },
  name: { fontWeight: '800' },
  meta: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  pill: { paddingHorizontal: 8, paddingVertical: 2, borderRadius: 999 },
  pillText: { fontSize: 10, fontWeight: '600', textTransform: 'uppercase', letterSpacing: 0.4 },
  more: { paddingHorizontal: 12, alignSelf: 'stretch', justifyContent: 'center' },
});
