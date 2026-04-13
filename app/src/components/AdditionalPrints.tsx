import { View, Text, Pressable, ActivityIndicator, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { usePrintings } from '../api/hooks';
import { useTheme } from '../theme/useTheme';
import { PressableCardImage } from './PressableCardImage';
import type { PrintingSummary } from '../api/scryfall';
import type { CachedCard } from '../db/cards';

type Props = { card: CachedCard };

export function AdditionalPrints({ card }: Props) {
  const router = useRouter();
  const theme = useTheme();
  const { data: printings = [], isLoading, isError } = usePrintings(card);

  if (!isLoading && !isError && printings.length <= 1) return null;

  return (
    <View style={[styles.container, { backgroundColor: theme.surface }]}>
      <Text style={[styles.heading, { color: theme.accent }]}>Printings</Text>

      {isLoading ? (
        <ActivityIndicator color={theme.accent} style={styles.loader} />
      ) : isError ? (
        <Text style={[styles.empty, { color: theme.textSecondary }]}>Could not load printings</Text>
      ) : (
        printings.map(p => (
          <PrintingRow
            key={p.scryfall_id}
            printing={p}
            onPress={() => router.push(`/card/${p.scryfall_id}`)}
          />
        ))
      )}
    </View>
  );
}

type RowProps = { printing: PrintingSummary; onPress: () => void };

function PrintingRow({ printing, onPress }: RowProps) {
  const theme = useTheme();
  const usd = printing.prices.usd ? `$${printing.prices.usd}` : '—';
  const foil = printing.prices.usd_foil ? `✦ $${printing.prices.usd_foil}` : '—';

  return (
    <Pressable style={[styles.row, { borderBottomColor: theme.border }]} onPress={onPress} accessibilityLabel={`${printing.set_name} #${printing.collector_number}`}>
      {printing.image_uri ? (
        <PressableCardImage uri={printing.image_uri} style={styles.thumbnail} onPress={onPress} />
      ) : (
        <View style={[styles.thumbnail, { backgroundColor: theme.surfaceAlt }]} />
      )}
      <View style={styles.rowInfo}>
        <View style={styles.rowTop}>
          <View style={[styles.setSquare, { backgroundColor: theme.accent }]}>
            <Text style={[styles.setCode, { color: theme.text }]}>{printing.set_code.toUpperCase()}</Text>
          </View>
          <Text style={[styles.setName, { color: theme.textSecondary }]} numberOfLines={1}>{printing.set_name}</Text>
          <Text style={[styles.collectorNum, { color: theme.border }]}>#{printing.collector_number}</Text>
        </View>
        <View style={styles.rowPrices}>
          <Text style={[styles.price, { color: theme.text }]}>{usd}</Text>
          <Text style={[styles.price, styles.foilPrice]}>{foil}</Text>
        </View>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  container: { borderRadius: 8, padding: 12, marginTop: 12 },
  heading: { fontWeight: '700', marginBottom: 8, fontSize: 13 },
  loader: { marginVertical: 12 },
  empty: { fontSize: 12 },
  row: { flexDirection: 'row', alignItems: 'center', paddingVertical: 6, gap: 10, borderBottomWidth: StyleSheet.hairlineWidth },
  thumbnail: { width: 44, height: 62, borderRadius: 4 },
  rowInfo: { flex: 1, gap: 4 },
  rowTop: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  rowPrices: { flexDirection: 'row', gap: 10 },
  setSquare: { width: 36, height: 20, borderRadius: 3, alignItems: 'center', justifyContent: 'center', flexShrink: 0 },
  setCode: { fontSize: 9, fontWeight: '700' },
  setName: { flex: 1, fontSize: 11 },
  collectorNum: { fontSize: 11, minWidth: 36, textAlign: 'right' },
  price: { fontSize: 11, minWidth: 48, textAlign: 'right' },
  foilPrice: { color: '#b8a0e8' },
});
