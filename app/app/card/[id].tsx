import {
  View,
  Text,
  Image,
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
import { addToCollection } from '../../src/db/collection';
import { addCardToDeck } from '../../src/db/decks';
import { useStore } from '../../src/store/useStore';

export default function CardDetailModal() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const qc = useQueryClient();
  const { activeDeckId } = useStore();

  const { data: card, isLoading } = useCard(id);
  const [condition, setCondition] = useState<Condition>('NM');
  const [foil, setFoil] = useState(false);

  if (isLoading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color="#4ecdc4" />
      </View>
    );
  }

  if (!card) {
    return (
      <View style={styles.center}>
        <Text style={styles.error}>Card not found</Text>
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
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <TouchableOpacity style={styles.closeBtn} onPress={() => router.back()}>
        <Text style={styles.closeBtnText}>✕</Text>
      </TouchableOpacity>
      <View style={styles.row}>
        {card.image_uri ? (
          <Image
            source={{ uri: card.image_uri }}
            style={styles.image}
            resizeMode="contain"
          />
        ) : (
          <View style={[styles.image, styles.imagePlaceholder]} />
        )}
        <View style={styles.meta}>
          <Text style={styles.name}>{card.name}</Text>
          <Text style={styles.subtitle}>
            {card.mana_cost}{'  '}{card.type_line}
          </Text>
          <Text style={styles.oracle}>{card.oracle_text}</Text>
          <View style={styles.prices}>
            {prices.usd ? (
              <Text style={styles.price}>${prices.usd}</Text>
            ) : null}
            {prices.usd_foil ? (
              <Text style={styles.price}>✨ ${prices.usd_foil}</Text>
            ) : null}
          </View>
        </View>
      </View>

      <View style={styles.actions}>
        <Text style={styles.sectionLabel}>Condition</Text>
        <ConditionPicker value={condition} onChange={setCondition} />
        <TouchableOpacity style={styles.foilToggle} onPress={() => setFoil((f) => !f)}>
          <Text style={[styles.foilText, foil && styles.foilActive]}>
            ✨ Foil {foil ? '(on)' : '(off)'}
          </Text>
        </TouchableOpacity>
        <View style={styles.btnRow}>
          <TouchableOpacity style={styles.btn} onPress={handleAddToBinder}>
            <Text style={styles.btnText}>+ Binder</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.btn, styles.btnSecondary]} onPress={handleAddToDeck}>
            <Text style={styles.btnText}>+ Deck</Text>
          </TouchableOpacity>
        </View>
      </View>

      <FindSimilar card={card} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#111318' },
  content: { padding: 16 },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#111318',
  },
  error: { color: '#4ecdc4' },
  row: { flexDirection: 'row', gap: 12 },
  image: { width: 130, height: 182, borderRadius: 8 },
  imagePlaceholder: { backgroundColor: '#2a1a3e' },
  meta: { flex: 1 },
  name: { color: '#fff', fontSize: 18, fontWeight: '700', marginBottom: 4 },
  subtitle: { color: '#aaa', fontSize: 13, marginBottom: 8 },
  oracle: { color: '#ccc', fontSize: 12, lineHeight: 18 },
  prices: { flexDirection: 'row', gap: 8, marginTop: 8 },
  price: { color: '#4ecdc4', fontSize: 13, fontWeight: '600' },
  actions: { marginTop: 16, gap: 10 },
  sectionLabel: {
    color: '#888',
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  foilToggle: { alignSelf: 'flex-start' },
  foilText: { color: '#888', fontSize: 13 },
  foilActive: { color: '#4ecdc4', fontWeight: '700' },
  btnRow: { flexDirection: 'row', gap: 8 },
  btn: {
    flex: 1,
    backgroundColor: '#4ecdc4',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
  },
  btnSecondary: { backgroundColor: '#252830' },
  btnText: { color: '#fff', fontWeight: '700' },
  closeBtn: { alignSelf: 'flex-end', padding: 4, marginBottom: 8 },
  closeBtnText: { color: '#888', fontSize: 18, lineHeight: 20 },
});
