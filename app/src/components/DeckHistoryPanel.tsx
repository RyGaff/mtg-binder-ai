import { memo, useCallback } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { getDeckHistory, undoDeckEvent, type Board, type DeckHistoryEvent } from '../db/decks';
import { useTheme } from '../theme/useTheme';

type Props = { deckId: number };

const BOARD_LABEL: Record<Board, string> = {
  commander: 'Commander', main: 'Main', side: 'Sideboard', considering: 'Considering',
};

// Short relative-time formatter — "just now / 5m / 2h / 3d / 2w".
// Stops at weeks; anything older shows the date string.
function relativeTime(now: number, ts: number): string {
  const diff = Math.max(0, now - ts);
  const m = Math.floor(diff / 60000);
  if (m < 1) return 'just now';
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h`;
  const d = Math.floor(h / 24);
  if (d < 14) return `${d}d`;
  const w = Math.floor(d / 7);
  if (w < 8) return `${w}w`;
  return new Date(ts).toLocaleDateString();
}

function summary(ev: DeckHistoryEvent): string {
  const from = BOARD_LABEL[ev.board_from];
  if (ev.event_type === 'move' && ev.board_to) {
    return `Moved ${BOARD_LABEL[ev.board_from]} → ${BOARD_LABEL[ev.board_to]}`;
  }
  if (ev.event_type === 'add') return `Added ${ev.qty_delta > 1 ? `×${ev.qty_delta} ` : ''}to ${from}`;
  if (ev.event_type === 'remove') return `Removed ${ev.qty_delta > 1 ? `×${ev.qty_delta} ` : 'all '}from ${from}`;
  return `Removed 1 from ${from}`;
}

function DeckHistoryPanelImpl({ deckId }: Props) {
  const t = useTheme();
  const qc = useQueryClient();
  const historyQ = useQuery({
    queryKey: ['deck-history', deckId],
    queryFn: () => getDeckHistory(deckId),
    enabled: Number.isFinite(deckId),
  });
  const events = historyQ.data ?? [];

  // Tap row → undo the event. Inverse mutation runs in a transaction and the
  // event row itself is removed from history (otherwise undoing would leave
  // a residual entry users would expect to undo again).
  const undo = useCallback((ev: DeckHistoryEvent) => {
    undoDeckEvent(ev.id);
    qc.invalidateQueries({ queryKey: ['deck-cards', deckId] });
    qc.invalidateQueries({ queryKey: ['deck', deckId] });
    qc.invalidateQueries({ queryKey: ['decks'] });
    qc.invalidateQueries({ queryKey: ['deck-history', deckId] });
  }, [deckId, qc]);

  if (events.length === 0) {
    return (
      <View style={[s.wrap, { borderBottomColor: t.border }]}>
        <Text style={[s.empty, { color: t.textSecondary }]}>No history yet.</Text>
      </View>
    );
  }

  const now = Date.now();
  return (
    <View style={[s.wrap, { borderBottomColor: t.border }]}>
      {events.map((ev) => (
        <Pressable
          key={ev.id}
          onPress={() => undo(ev)}
          accessibilityRole="button"
          accessibilityLabel={`Undo ${summary(ev)} for ${ev.card_name || 'card'}`}
          style={({ pressed }) => [s.row, { borderBottomColor: t.border }, pressed && { opacity: 0.5 }]}
        >
          <View style={s.body}>
            <Text style={[s.name, { color: t.text }]} numberOfLines={1}>{ev.card_name || '(unknown card)'}</Text>
            {/* Tint the summary line by event polarity: adds are green, removes
                (incl. decrement) are red, moves stay neutral since they aren't
                net additions or losses to the deck total. */}
            <Text
              style={[
                s.summary,
                {
                  color: ev.event_type === 'add' ? t.success
                    : ev.event_type === 'remove' || ev.event_type === 'decrement' ? t.danger
                    : t.textSecondary,
                },
              ]}
              numberOfLines={1}
            >
              {summary(ev)}
            </Text>
          </View>
          <Text style={[s.time, { color: t.textSecondary }]}>{relativeTime(now, ev.created_at)}</Text>
          {/* Undo glyph mirrors the inverse of the original event: an add gets a
              − (subtract a copy), a remove/decrement gets a + (put it back).
              Move events fall back to a circular-arrow glyph since the inverse
              isn't a quantity change. */}
          <Text style={[s.undoIcon, { color: t.accent }]}>
            {ev.event_type === 'add' ? '−'
              : ev.event_type === 'remove' || ev.event_type === 'decrement' ? '+'
              : '↺'}
          </Text>
        </Pressable>
      ))}
    </View>
  );
}

export const DeckHistoryPanel = memo(DeckHistoryPanelImpl);

const s = StyleSheet.create({
  wrap: { paddingVertical: 4, borderBottomWidth: 1 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 14, paddingVertical: 8, borderBottomWidth: StyleSheet.hairlineWidth },
  body: { flex: 1, gap: 2 },
  name: { fontSize: 13, fontWeight: '600' },
  summary: { fontSize: 11 },
  time: { fontSize: 11, fontVariant: ['tabular-nums'] },
  undoIcon: { fontSize: 22, fontWeight: '300', marginLeft: 8, marginRight: 2, minWidth: 16, textAlign: 'center' },
  empty: { fontSize: 12, paddingVertical: 14, paddingHorizontal: 14, textAlign: 'center' },
});
