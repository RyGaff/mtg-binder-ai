import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Animated,
  BackHandler,
  Dimensions,
  Easing,
  Image,
  PanResponder,
  StyleSheet,
} from 'react-native';
import { useLocalSearchParams, useRouter, useNavigation } from 'expo-router';
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQueryClient, type QueryClient } from '@tanstack/react-query';
import { useCard } from '../../src/api/hooks';
import { ConditionPicker, type Condition } from '../../src/components/ConditionPicker';
import { FindSimilar } from '../../src/components/FindSimilar';
import { Synergy } from '../../src/components/Synergy';
import { AdditionalPrints } from '../../src/components/AdditionalPrints';
import { MeldLinks } from '../../src/components/MeldLinks';
import { PressableCardImage } from '../../src/components/PressableCardImage';
import { addToCollection } from '../../src/db/collection';
import { addCardToDeck } from '../../src/db/decks';
import { useStore } from '../../src/store/useStore';
import { useTheme } from '../../src/theme/useTheme';
import { spacing, radius, font, MIN_TOUCH, HIT_SLOP_8 } from '../../src/theme/themes';
import { Icon } from '../../src/components/icons/Icon';

const HERO_ASPECT = 488 / 680;
// Card image is sized as a fraction of the shorter screen edge so it scales
// gracefully across phone portrait, phone landscape, and tablet.
function heroImageSize(width: number, height: number) {
  const w = Math.min(Math.max(Math.min(width, height) * 0.36, 120), 220);
  return { width: w, height: w / HERO_ASPECT };
}

type Theme = ReturnType<typeof useTheme>;
type Card = NonNullable<ReturnType<typeof useCard>['data']>;
type TrailEntry = { id: string; name: string };

const SLIDE_IN_MS = 220;
const SLIDE_OUT_MS = 220;
const GESTURE_OUT_MS = 160;
const CAPTURE_DIST = 8;
// require one axis to exceed the other by this factor before capturing
const DOMINANCE = 1.0;
const COMMIT_DIST_RATIO = 0.25;
const COMMIT_VELOCITY = 0.5;
const EASE = Easing.out(Easing.cubic);

const timing = (value: Animated.Value, toValue: number, duration: number) =>
  Animated.timing(value, { toValue, duration, easing: EASE, useNativeDriver: true });

function initialOffsetFor(dir: 'initial' | 'forward' | 'backward') {
  const width = Dimensions.get('window').width;
  if (dir === 'forward') return width;
  if (dir === 'backward') return -width;
  return 0;
}

export default function CardDetailModal() {
  const theme = useTheme();
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const navigation = useNavigation();
  const qc = useQueryClient();

  const activeDeckId = useStore((s) => s.activeDeckId);
  const trail = useStore((s) => s.cardTrail);
  const pushCardTrail = useStore((s) => s.pushCardTrail);
  const clearCardTrail = useStore((s) => s.clearCardTrail);

  const { data: card, error } = useCard(id);
  const errStatus = (error as (Error & { status?: number }) | null)?.status;
  const prevId = trail.length > 1 ? trail[trail.length - 2].id : '';
  const { data: prevCard } = useCard(prevId);

  const [condition, setCondition] = useState<Condition>('NM');
  const [foil, setFoil] = useState(false);
  const [extrasReady, setExtrasReady] = useState(false);
  const [heroFlipped, setHeroFlipped] = useState(false);
  const toggleHeroFlip = useCallback(() => setHeroFlipped((f) => !f), []);

  // Read direction ONCE before first paint — start off-screen instead of snapping.
  const initialDirRef = useRef(useStore.getState().lastCardNavDir);
  const slideX = useRef(new Animated.Value(initialOffsetFor(initialDirRef.current))).current;
  const slideY = useRef(new Animated.Value(0)).current;

  const trailScrollRef = useRef<ScrollView>(null);
  const scrollAtTopRef = useRef(true);
  const gestureDirRef = useRef<'right' | 'down' | null>(null);
  const navigatingRef = useRef(false);
  const skipInterceptRef = useRef(false);
  const isMountedRef = useRef(true);
  const extrasReadyRef = useRef(false);

  const handleHeroReady = useCallback(() => {
    if (extrasReadyRef.current) return;
    extrasReadyRef.current = true;
    setExtrasReady(true);
  }, []);

  useEffect(() => () => {
    isMountedRef.current = false;
    slideX.stopAnimation();
  }, [slideX]);

  useEffect(() => {
    if (initialDirRef.current === 'initial') return;
    useStore.setState({ lastCardNavDir: 'initial' });
    timing(slideX, 0, SLIDE_IN_MS).start();
  }, [slideX]);

  useEffect(() => {
    if (card) pushCardTrail({ id: card.scryfall_id, name: card.name });
  }, [card?.scryfall_id, card?.name, pushCardTrail]);

  useEffect(() => { setHeroFlipped(false); }, [card?.scryfall_id]);

  useEffect(() => {
    trailScrollRef.current?.scrollToEnd({ animated: false });
  }, [trail.length]);

  const handleClose = useCallback(() => {
    skipInterceptRef.current = true;
    useStore.setState({ suppressTrailReset: false, lastCardNavDir: 'initial' });
    clearCardTrail();
    if (router.canGoBack()) router.back();
    else router.replace('/');
  }, [clearCardTrail, router]);

  // Swap current card for targetId: slide out, setParams, snap, slide in.
  // outMs differs when continuing a gesture (shorter) vs. starting cold.
  const animateSwap = useCallback(
    (targetId: string, outMs: number = SLIDE_OUT_MS, suppressReset = false) => {
      if (navigatingRef.current) return;
      navigatingRef.current = true;
      skipInterceptRef.current = true;
      useStore.setState({
        lastCardNavDir: 'backward',
        ...(suppressReset ? { suppressTrailReset: true } : {}),
      });
      const width = Dimensions.get('window').width;
      timing(slideX, width, outMs).start(() => {
        if (!isMountedRef.current) return;
        router.setParams({ id: targetId });
        slideX.setValue(-width);
        timing(slideX, 0, SLIDE_IN_MS).start(() => {
          if (!isMountedRef.current) return;
          navigatingRef.current = false;
          useStore.setState({ lastCardNavDir: 'initial' });
        });
      });
    },
    [router, slideX],
  );

  const goBackInTrail = useCallback((): boolean => {
    const current = useStore.getState().cardTrail;
    if (current.length <= 1) return false;
    animateSwap(current[current.length - 2].id);
    return true;
  }, [animateSwap]);

  const jumpToTrailCard = useCallback(
    (targetId: string) => animateSwap(targetId, SLIDE_OUT_MS, true),
    [animateSwap],
  );

  useEffect(() => {
    const sub = BackHandler.addEventListener('hardwareBackPress', () => goBackInTrail());
    return () => sub.remove();
  }, [goBackInTrail]);

  useEffect(() => {
    const unsub = navigation.addListener('beforeRemove', (e: { preventDefault: () => void }) => {
      if (skipInterceptRef.current) {
        skipInterceptRef.current = false;
        return;
      }
      // Child nav (synergy/similar/printings) sets this flag — let the replace through, keep trail.
      if (useStore.getState().suppressTrailReset) {
        useStore.setState({ suppressTrailReset: false });
        return;
      }
      const current = useStore.getState().cardTrail;
      if (current.length <= 1) {
        clearCardTrail();
        return;
      }
      e.preventDefault();
      goBackInTrail();
    });
    return unsub;
  }, [navigation, goBackInTrail, clearCardTrail]);

  const animateSwapRef = useRef(animateSwap);
  const handleCloseRef = useRef(handleClose);
  useEffect(() => { animateSwapRef.current = animateSwap; }, [animateSwap]);
  useEffect(() => { handleCloseRef.current = handleClose; }, [handleClose]);

  const panResponder = useMemo(() => {
    const classify = (g: { dx: number; dy: number }): 'right' | 'down' | null => {
      const ax = Math.abs(g.dx);
      const ay = Math.abs(g.dy);
      if (g.dx > CAPTURE_DIST && g.dx > ay * DOMINANCE) {
        return useStore.getState().cardTrail.length > 1 ? 'right' : null;
      }
      if (g.dy > CAPTURE_DIST && g.dy > ax * DOMINANCE) {
        return scrollAtTopRef.current ? 'down' : null;
      }
      return null;
    };

    const settle = (
      dir: 'right' | 'down' | null,
      g: { dx: number; dy: number; vx: number; vy: number },
    ) => {
      const { width, height } = Dimensions.get('window');
      if (dir === 'right' && (g.dx > width * COMMIT_DIST_RATIO || g.vx > COMMIT_VELOCITY)) {
        const current = useStore.getState().cardTrail;
        if (current.length > 1) {
          animateSwapRef.current(current[current.length - 2].id, GESTURE_OUT_MS);
          return;
        }
      } else if (dir === 'down' && (g.dy > height * COMMIT_DIST_RATIO || g.vy > COMMIT_VELOCITY)) {
        timing(slideY, height, GESTURE_OUT_MS).start(() => handleCloseRef.current());
        return;
      }
      Animated.parallel([
        Animated.spring(slideX, { toValue: 0, useNativeDriver: true, bounciness: 0 }),
        Animated.spring(slideY, { toValue: 0, useNativeDriver: true, bounciness: 0 }),
      ]).start();
    };

    return PanResponder.create({
      onStartShouldSetPanResponderCapture: () => false,
      onMoveShouldSetPanResponderCapture: (_e, g) => {
        const dir = classify(g);
        if (dir) gestureDirRef.current = dir;
        return dir !== null;
      },
      onPanResponderMove: (_e, g) => {
        const dir = gestureDirRef.current;
        if (dir === 'right' && g.dx > 0) slideX.setValue(g.dx);
        else if (dir === 'down' && g.dy > 0) slideY.setValue(g.dy);
      },
      onPanResponderRelease: (_e, g) => {
        const dir = gestureDirRef.current;
        gestureDirRef.current = null;
        settle(dir, g);
      },
      onPanResponderTerminate: (_e, g) => {
        const dir = gestureDirRef.current;
        gestureDirRef.current = null;
        settle(dir, g);
      },
      onPanResponderTerminationRequest: () => false,
      onShouldBlockNativeResponder: () => true,
    });
  }, [slideX, slideY]);

  if (!card) {
    return <LoadingOrError theme={theme} errStatus={errStatus} error={error} onExit={handleClose} />;
  }

  const setScrollTop = (y: number) => { scrollAtTopRef.current = y <= 0; };

  return (
    <View style={[styles.screen, { backgroundColor: theme.bg }]}>
      {prevCard && (
        <View style={StyleSheet.absoluteFill} pointerEvents="none">
          <CardPeek card={prevCard} theme={theme} />
        </View>
      )}
      <Animated.View
        style={[
          styles.screen,
          { backgroundColor: theme.bg, transform: [{ translateX: slideX }, { translateY: slideY }] },
        ]}
        {...panResponder.panHandlers}
      >
        <ScrollView
          style={styles.screen}
          contentContainerStyle={styles.content}
          directionalLockEnabled
          onScrollBeginDrag={() => { scrollAtTopRef.current = false; }}
          onScroll={(e) => setScrollTop(e.nativeEvent.contentOffset.y)}
          onScrollEndDrag={(e) => setScrollTop(e.nativeEvent.contentOffset.y)}
          scrollEventThrottle={16}
        >
          <TopBar
            theme={theme}
            trail={trail}
            trailScrollRef={trailScrollRef}
            onJump={jumpToTrailCard}
            onClose={handleClose}
          />
          <CardHero
            card={card}
            theme={theme}
            onImageReady={handleHeroReady}
            flipped={heroFlipped}
            onFlip={toggleHeroFlip}
          />
          <CardActions
            theme={theme}
            condition={condition}
            onConditionChange={setCondition}
            foil={foil}
            onToggleFoil={() => setFoil((f) => !f)}
            onAddToBinder={() => addBinder(card, foil, condition, qc)}
            onAddToDeck={() => addDeck(card, activeDeckId)}
          />
          <MeldLinks card={card} />
          {extrasReady && (
            <>
              <FindSimilar card={card} />
              <Synergy card={card} />
              <AdditionalPrints card={card} />
            </>
          )}
        </ScrollView>
      </Animated.View>
    </View>
  );
}

function addBinder(card: Card, foil: boolean, condition: Condition, qc: QueryClient) {
  addToCollection({ scryfall_id: card.scryfall_id, quantity: 1, foil, condition });
  qc.invalidateQueries({ queryKey: ['collection'] });
  qc.invalidateQueries({ queryKey: ['collection-value'] });
  Alert.alert('Added', `${card.name} added to your binder.`);
}

function addDeck(card: Card, activeDeckId: number | null) {
  if (!activeDeckId) {
    Alert.alert('No deck selected', 'Open the Decks tab and select a deck first.');
    return;
  }
  addCardToDeck({ deck_id: activeDeckId, scryfall_id: card.scryfall_id, quantity: 1, board: 'main' });
  Alert.alert('Added', `${card.name} added to deck.`);
}

function parseJsonSafe<T>(raw: string | null | undefined, fallback: T, label: string): T {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw);
  } catch (e) {
    if (__DEV__) console.warn(`[card] ${label}: malformed JSON`, e);
    return fallback;
  }
}

function parsePrices(raw: string | null | undefined): { usd?: string; usd_foil?: string } {
  return parseJsonSafe(raw, {}, 'parsePrices');
}

type ParsedFace = { name: string; mana_cost: string; type_line: string; oracle_text: string; image_uri: string };
function parseCardFaces(raw: string | null | undefined): ParsedFace[] {
  const arr = parseJsonSafe<unknown>(raw, [], 'parseCardFaces');
  return Array.isArray(arr) ? (arr as ParsedFace[]) : [];
}

function LoadingOrError({
  theme,
  errStatus,
  error,
  onExit,
}: {
  theme: Theme;
  errStatus?: number;
  error: unknown;
  onExit: () => void;
}) {
  let msg: string | null = null;
  if (errStatus === 404) msg = 'Card not found';
  else if (errStatus === 429) msg = 'Scryfall is rate-limiting requests. Try again in a moment.';
  else if (error) msg = 'Could not load this card.';

  if (!msg) {
    return (
      <View style={[styles.center, { backgroundColor: theme.bg }]}>
        <ActivityIndicator color={theme.accent} />
      </View>
    );
  }
  return (
    <View style={[styles.center, { backgroundColor: theme.bg }]}>
      <Text style={[styles.errorText, { color: theme.text }]}>{msg}</Text>
      <TouchableOpacity
        style={[styles.errorExitBtn, { backgroundColor: theme.accent }]}
        onPress={onExit}
        accessibilityRole="button"
        accessibilityLabel="Exit"
      >
        <Text style={[styles.errorExitText, { color: theme.text }]}>Exit</Text>
      </TouchableOpacity>
    </View>
  );
}

function TopBar({
  theme,
  trail,
  trailScrollRef,
  onJump,
  onClose,
}: {
  theme: Theme;
  trail: TrailEntry[];
  trailScrollRef: React.RefObject<ScrollView | null>;
  onJump: (id: string) => void;
  onClose: () => void;
}) {
  return (
    <View style={styles.topBar}>
      <View style={styles.sideSlot} />
      <ScrollView
        ref={trailScrollRef}
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.trail}
        style={styles.trailScroll}
      >
        {trail.map((t, i) => {
          if (i === trail.length - 1) {
            return (
              <Text key={t.id} style={[styles.trailChipCurrent, { color: theme.text }]} numberOfLines={1} ellipsizeMode="tail">
                {t.name}
              </Text>
            );
          }
          return (
            <Fragment key={t.id}>
              <TouchableOpacity
                onPress={() => onJump(t.id)}
                hitSlop={HIT_SLOP_8}
                accessibilityRole="button"
                accessibilityLabel={`Back to ${t.name}`}
              >
                <Text style={[styles.trailChip, { color: theme.textSecondary }]} numberOfLines={1} ellipsizeMode="tail">
                  {t.name}
                </Text>
              </TouchableOpacity>
              <Text style={[styles.trailSep, { color: theme.textSecondary }]}> › </Text>
            </Fragment>
          );
        })}
      </ScrollView>
      <TouchableOpacity
        style={[styles.sideSlot, styles.closeBtn]}
        onPress={onClose}
        hitSlop={HIT_SLOP_8}
        accessibilityRole="button"
        accessibilityLabel="Close"
      >
        <Icon name="close" size={20} color={theme.textSecondary} />
      </TouchableOpacity>
    </View>
  );
}

function HeroMeta({ theme, compact, name, mana, type, oracle, prices }: {
  theme: Theme; compact: boolean; name: string; mana: string; type: string; oracle: string;
  prices: { usd?: string; usd_foil?: string };
}) {
  return (
    <View style={styles.meta}>
      <Text style={[styles.name, { color: theme.text }]} numberOfLines={compact ? 2 : undefined} ellipsizeMode="tail">
        {name}
      </Text>
      <Text style={[styles.subtitle, { color: theme.textSecondary }]} numberOfLines={compact ? 1 : undefined} ellipsizeMode="tail">
        {mana}{'  '}{type}
      </Text>
      <Text style={[styles.oracle, { color: theme.textSecondary }]} numberOfLines={compact ? 5 : undefined} ellipsizeMode="tail">
        {oracle}
      </Text>
      <View style={styles.prices}>
        {prices.usd ? <Text style={[styles.price, { color: theme.accent }]}>${prices.usd}</Text> : null}
        {prices.usd_foil ? (
          <View style={styles.foilPriceRow}>
            <Icon name="sparkle" size={12} color={theme.foilAccent} />
            <Text style={[styles.price, { color: theme.foilAccent }]}>${prices.usd_foil}</Text>
          </View>
        ) : null}
      </View>
    </View>
  );
}

function CardHero({ card, theme, compact = false, onImageReady, flipped = false, onFlip }: { card: Card; theme: Theme; compact?: boolean; onImageReady?: () => void; flipped?: boolean; onFlip?: () => void }) {
  const prices = parsePrices(card.prices);
  const faces = parseCardFaces(card.card_faces);
  const isMultiFace = faces.length >= 2;
  const activeFace = isMultiFace ? faces[flipped ? 1 : 0] : undefined;
  const { width: winW, height: winH } = Dimensions.get('window');
  const imgSize = heroImageSize(winW, winH);
  return (
    <View style={styles.row}>
      {card.image_uri ? (
        <PressableCardImage
          uri={card.image_uri}
          uriBack={card.image_uri_back || undefined}
          flipped={isMultiFace ? flipped : undefined}
          onFlip={isMultiFace ? onFlip : undefined}
          style={[styles.image, imgSize]}
          resizeMode="contain"
          onReady={onImageReady}
        />
      ) : (
        <View style={[styles.image, imgSize, { backgroundColor: theme.surfaceAlt }]} onLayout={onImageReady} />
      )}
      <HeroMeta
        theme={theme}
        compact={compact}
        name={activeFace?.name || card.name}
        mana={activeFace?.mana_cost ?? card.mana_cost}
        type={activeFace?.type_line ?? card.type_line}
        oracle={activeFace?.oracle_text ?? card.oracle_text}
        prices={prices}
      />
    </View>
  );
}

function CardActions({
  theme,
  condition,
  onConditionChange,
  foil,
  onToggleFoil,
  onAddToBinder,
  onAddToDeck,
}: {
  theme: Theme;
  condition: Condition;
  onConditionChange: (c: Condition) => void;
  foil: boolean;
  onToggleFoil: () => void;
  onAddToBinder: () => void;
  onAddToDeck: () => void;
}) {
  const foilColor = foil ? theme.foilAccent : theme.textSecondary;
  return (
    <View style={styles.actions}>
      <Text style={[styles.sectionLabel, { color: theme.textSecondary }]}>Condition</Text>
      <ConditionPicker value={condition} onChange={onConditionChange} />
      <TouchableOpacity
        style={styles.foilToggle}
        onPress={onToggleFoil}
        accessibilityRole="switch"
        accessibilityState={{ checked: foil }}
        accessibilityLabel="Foil"
      >
        <Icon name={foil ? 'sparkle' : 'sparkle-outline'} size={14} color={foilColor} />
        <Text style={[styles.foilText, { color: foilColor }, foil && styles.foilActive]}>
          Foil {foil ? '(on)' : '(off)'}
        </Text>
      </TouchableOpacity>
      <View style={styles.btnRow}>
        <TouchableOpacity style={[styles.btn, styles.btnInner, { backgroundColor: theme.accent }]} onPress={onAddToBinder} accessibilityRole="button" accessibilityLabel="Add to binder">
          <Icon name="plus" size={16} color={theme.text} strokeWidth={2.5} />
          <Text style={[styles.btnText, { color: theme.text }]}>Binder</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.btn, styles.btnInner, { backgroundColor: theme.surfaceAlt }]} onPress={onAddToDeck} accessibilityRole="button" accessibilityLabel="Add to deck">
          <Icon name="plus" size={16} color={theme.text} strokeWidth={2.5} />
          <Text style={[styles.btnText, { color: theme.text }]}>Deck</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

function CardPeek({ card, theme }: { card: Card; theme: Theme }) {
  const prices = parsePrices(card.prices);
  const { width: winW, height: winH } = Dimensions.get('window');
  const imgSize = heroImageSize(winW, winH);
  return (
    <View style={[styles.peek, { backgroundColor: theme.bg }]}>
      <View style={styles.row}>
        {card.image_uri ? (
          <Image source={{ uri: card.image_uri }} style={[styles.image, imgSize]} resizeMode="contain" />
        ) : (
          <View style={[styles.image, imgSize, { backgroundColor: theme.surfaceAlt }]} />
        )}
        <HeroMeta
          theme={theme}
          compact
          name={card.name}
          mana={card.mana_cost}
          type={card.type_line}
          oracle={card.oracle_text}
          prices={prices}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  content: { padding: spacing.lg },
  peek: { flex: 1, padding: spacing.lg, paddingTop: 48 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  errorText: { fontSize: font.body, textAlign: 'center', marginHorizontal: spacing.xl, marginBottom: spacing.lg },
  errorExitBtn: { paddingHorizontal: spacing.xl, paddingVertical: spacing.sm + 2, borderRadius: radius.md, minHeight: MIN_TOUCH, justifyContent: 'center' },
  errorExitText: { fontWeight: '700' },
  row: { flexDirection: 'row', gap: spacing.md },
  image: { borderRadius: radius.md },
  meta: { flex: 1 },
  name: { fontSize: font.title, fontWeight: '700', marginBottom: spacing.xs },
  subtitle: { fontSize: 13, marginBottom: spacing.sm },
  oracle: { fontSize: font.small, lineHeight: 18 },
  prices: { flexDirection: 'row', gap: spacing.sm, marginTop: spacing.sm },
  price: { fontSize: 13, fontWeight: '600' },
  actions: { marginTop: spacing.lg, gap: spacing.sm + 2 },
  sectionLabel: { fontSize: font.caption, textTransform: 'uppercase', letterSpacing: 1 },
  foilToggle: { alignSelf: 'flex-start', flexDirection: 'row', alignItems: 'center', gap: spacing.xs + 2, paddingVertical: spacing.sm, paddingHorizontal: spacing.xs, minHeight: MIN_TOUCH },
  foilPriceRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs },
  foilText: { fontSize: 13 },
  foilActive: { fontWeight: '700' },
  btnRow: { flexDirection: 'row', gap: spacing.sm },
  btn: { flex: 1, borderRadius: radius.md, paddingVertical: spacing.md, alignItems: 'center', minHeight: MIN_TOUCH, justifyContent: 'center' },
  btnInner: { flexDirection: 'row', gap: spacing.xs + 2 },
  btnText: { fontWeight: '700' },
  topBar: { flexDirection: 'row', alignItems: 'center', marginBottom: spacing.sm, minHeight: MIN_TOUCH },
  sideSlot: { width: MIN_TOUCH, height: MIN_TOUCH, alignItems: 'center', justifyContent: 'center' },
  trailScroll: { flex: 1 },
  trail: { flexGrow: 1, justifyContent: 'center', alignItems: 'center', paddingHorizontal: spacing.sm },
  trailChip: { fontSize: font.small, fontWeight: '600', maxWidth: 140 },
  trailChipCurrent: { fontSize: font.small, fontWeight: '700', maxWidth: 180 },
  trailSep: { fontSize: font.small },
  closeBtn: { padding: spacing.xs },
  closeBtnText: { fontSize: font.title, lineHeight: 20 },
});
