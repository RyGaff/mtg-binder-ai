import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  StyleSheet,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useCard } from '../../src/api/hooks';
import { ConditionPicker, type Condition } from '../../src/components/ConditionPicker';
import { FindSimilar } from '../../src/components/FindSimilar';
import { AdditionalPrints } from '../../src/components/AdditionalPrints';
import { PressableCardImage } from '../../src/components/PressableCardImage';
import { addToCollection } from '../../src/db/collection';
import { addCardToDeck } from '../../src/db/decks';
import { useStore } from '../../src/store/useStore';
import { useTheme } from '../../src/theme/useTheme';

export default function CardDetailModal() {
  const theme = useTheme();
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const qc = useQueryClient();
  const { activeDeckId } = useStore();

  const { data: card, isLoading } = useCard(id);
  const [condition, setCondition] = useState<Condition>('NM');
  const [foil, setFoil] = useState(false);

  if (isLoading) {
    return (
      <View style={[styles.center, { backgroundColor: theme.bg }]}>
        <ActivityIndicator color={theme.accent} />
      </View>
    );
  }

  if (!card) {
    return (
      <View style={[styles.center, { backgroundColor: theme.bg }]}>
        <Text style={{ color: theme.accent }}>Card not found</Text>
      </View>
    );
  }

  const prices = JSON.parse(card.prices || '{}');

  const handleAddToBinder = () => {
    addToCollection({ scryfall_id: card.scryfall_id, quantity: 1, foil, condition });
    qc.invalidateQueries({ queryKey: ['collection'] });
    qc.invalidateQueries({ queryKey: ['collection-value'] });
    Alert.alert('Added', `${card.name} added to your binder.`);
  };

  const handleAddToDeck = () => {
    if (!activeDeckId) {
      Alert.alert('No deck selected', 'Open the Decks tab and select a deck first.');
      return;
    }
    addCardToDeck({ deck_id: activeDeckId, scryfall_id: card.scryfall_id, quantity: 1, board: 'main' });
    Alert.alert('Added', `${card.name} added to deck.`);
  };

  return (
    <ScrollView style={[styles.screen, { backgroundColor: theme.bg }]} contentContainerStyle={styles.content}>
      <TouchableOpacity style={styles.closeBtn} onPress={() => (router.canGoBack() ? router.back() : router.replace('/'))}>
        <Text style={[styles.closeBtnText, { color: theme.textSecondary }]}>✕</Text>
      </TouchableOpacity>
      <View style={styles.row}>
        {card.image_uri ? (
          <PressableCardImage uri={card.image_uri} style={styles.image} resizeMode="contain" />
        ) : (
          <View style={[styles.image, styles.imagePlaceholder, { backgroundColor: theme.surfaceAlt }]} />
        )}
        <View style={styles.meta}>
          <Text style={[styles.name, { color: theme.text }]}>{card.name}</Text>
          <Text style={[styles.subtitle, { color: theme.textSecondary }]}>
            {card.mana_cost}{'  '}{card.type_line}
          </Text>
          <Text style={[styles.oracle, { color: theme.textSecondary }]}>{card.oracle_text}</Text>
          <View style={styles.prices}>
            {prices.usd ? (
              <Text style={[styles.price, { color: theme.accent }]}>${prices.usd}</Text>
            ) : null}
            {prices.usd_foil ? (
              <Text style={[styles.price, { color: theme.accent }]}>✨ ${prices.usd_foil}</Text>
            ) : null}
          </View>
        </View>
      </View>

      <View style={styles.actions}>
        <Text style={[styles.sectionLabel, { color: theme.textSecondary }]}>Condition</Text>
        <ConditionPicker value={condition} onChange={setCondition} />
        <TouchableOpacity style={styles.foilToggle} onPress={() => setFoil((f) => !f)}>
          <Text style={[styles.foilText, { color: foil ? theme.accent : theme.textSecondary }, foil && styles.foilActive]}>
            ✨ Foil {foil ? '(on)' : '(off)'}
          </Text>
        </TouchableOpacity>
        <View style={styles.btnRow}>
          <TouchableOpacity style={[styles.btn, { backgroundColor: theme.accent }]} onPress={handleAddToBinder}>
            <Text style={[styles.btnText, { color: theme.text }]}>+ Binder</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.btn, { backgroundColor: theme.surfaceAlt }]} onPress={handleAddToDeck}>
            <Text style={[styles.btnText, { color: theme.text }]}>+ Deck</Text>
          </TouchableOpacity>
        </View>
      </View>

      <FindSimilar card={card} />
      <AdditionalPrints card={card} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  content: { padding: 16 },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  row: { flexDirection: 'row', gap: 12 },
  image: { width: 130, height: 182, borderRadius: 8 },
  imagePlaceholder: {},
  meta: { flex: 1 },
  name: { fontSize: 18, fontWeight: '700', marginBottom: 4 },
  subtitle: { fontSize: 13, marginBottom: 8 },
  oracle: { fontSize: 12, lineHeight: 18 },
  prices: { flexDirection: 'row', gap: 8, marginTop: 8 },
  price: { fontSize: 13, fontWeight: '600' },
  actions: { marginTop: 16, gap: 10 },
  sectionLabel: {
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  foilToggle: { alignSelf: 'flex-start' },
  foilText: { fontSize: 13 },
  foilActive: { fontWeight: '700' },
  btnRow: { flexDirection: 'row', gap: 8 },
  btn: {
    flex: 1,
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
  },
  btnText: { fontWeight: '700' },
  closeBtn: { alignSelf: 'flex-end', padding: 4, marginBottom: 8 },
  closeBtnText: { fontSize: 18, lineHeight: 20 },
});
