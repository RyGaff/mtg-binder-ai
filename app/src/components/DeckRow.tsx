import { memo } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Image } from 'expo-image';
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
    <TouchableOpacity
      onPress={() => onPress(deck.id)}
      onLongPress={() => onLongPress(deck)}
      style={[
        banner ? s.banner : s.compact,
        banner
          ? { borderColor: active ? t.accent : t.border, borderWidth: active ? 2 : 1 }
          : { borderBottomColor: t.border, backgroundColor: active ? t.surface : 'transparent' },
      ]}
    >
      {banner && (deck.art_crop_uri
        ? <Image source={deck.art_crop_uri} style={s.fill} contentFit="cover" cachePolicy="memory-disk" recyclingKey={deck.art_crop_uri} />
        : <View style={[s.fill, { backgroundColor: accent }]} />)}
      {/* Translucent surface darken layer over the banner art. Bumped from
          0.55 → 0.7 so meta text reads against bright art. The text below
          also carries a 1px black shadow as a second-line defense for the
          rare cases where art is light on the bottom edge. */}
      {banner && <View style={[s.fill, { backgroundColor: t.surface, opacity: 0.7 }]} />}
      {!banner && <View style={{ width: 3, height: 32, borderRadius: 2, backgroundColor: accent }} />}

      <View style={s.body}>
        <Text style={[s.name, banner && s.bannerShadow, { color: t.text, fontSize: banner ? 15 : 14 }]} numberOfLines={1}>
          {deck.name}
        </Text>
        <View style={s.meta}>
          {banner && (
            <View style={[s.pill, { backgroundColor: t.surfaceAlt }]}>
              <Text style={[s.pillText, { color: t.text }]}>{deck.format}</Text>
            </View>
          )}
          {deck.color_identity.map((c) => (
            <Text key={c} style={{ fontFamily: 'Mana', color: MANA_TINT[c], fontSize: banner ? 13 : 12, lineHeight: 15 }}>
              {manaGlyph(c) ?? ''}
            </Text>
          ))}
          {(() => {
            // Defensive defaults — main_count/side_count were added recently. If a
            // cached query result from a prior session is hydrated before refetch,
            // those fields can be undefined; fall back to 0 to avoid "main undefined".
            const mainN = deck.main_count ?? 0;
            const sideN = deck.side_count ?? 0;
            // Banner mode: full-contrast text + drop shadow so the count is
            // readable over any deck art. Compact mode sits on a solid
            // surface so the dim secondary color is fine there.
            return (
              <Text
                style={[
                  { fontSize: 11, fontWeight: banner ? '600' : '400', color: banner ? t.text : t.textSecondary },
                  banner && s.bannerShadow,
                ]}
                numberOfLines={1}
              >
                {banner
                  ? `· main ${mainN} · sideboard ${sideN}`
                  : `${deck.format} · main ${mainN} · sideboard ${sideN} · ${rel(deck.created_at)}`}
              </Text>
            );
          })()}
        </View>
      </View>

      <TouchableOpacity onPress={() => onMore(deck)} hitSlop={8} style={s.more}>
        <Text style={{ fontSize: 22, fontWeight: '700', color: t.textSecondary }}>⋮</Text>
      </TouchableOpacity>
    </TouchableOpacity>
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
  // Drop shadow applied to all banner-mode text so it stays legible over
  // bright/busy art crops. Tight 2px blur with high alpha = subtle outline
  // that doesn't muddy the typography.
  bannerShadow: {
    textShadowColor: 'rgba(0,0,0,0.85)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 2,
  },
});
