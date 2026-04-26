import { useEffect, useMemo, useRef, useState } from 'react';
import { Animated, Easing, Pressable, StyleSheet, Text, View } from 'react-native';
import { useTheme } from '../theme/useTheme';
import { Icon, type IconName } from './icons/Icon';

export type SpeedDialItem = {
  label: string;
  icon: IconName;
  onPress: () => void;
};

type Props = { items: SpeedDialItem[] };

const PRIMARY_SIZE = 56;
const MINI_SIZE = 44;
const STACK_GAP = 12;
const PILL_GAP = 8;
const OPEN_DURATION = 180;
const CLOSE_DURATION = 140;
const STAGGER_MS = 40;

export function SpeedDialFab({ items }: Props) {
  const t = useTheme();
  const [open, setOpen] = useState(false);
  // While closing, we keep the menu mounted until the animation finishes.
  const [mounted, setMounted] = useState(false);

  const openness = useRef(new Animated.Value(0)).current;
  // One per item, indexed parallel to `items`.
  const itemAnims = useMemo(
    () => items.map(() => new Animated.Value(0)),
    // We want stable refs for the lifetime of items; recreate only if length changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [items.length],
  );

  useEffect(() => {
    if (open) {
      setMounted(true);
      Animated.timing(openness, {
        toValue: 1,
        duration: OPEN_DURATION,
        easing: Easing.out(Easing.quad),
        useNativeDriver: true,
      }).start();
      itemAnims.forEach((v, i) => {
        Animated.timing(v, {
          toValue: 1,
          duration: OPEN_DURATION,
          delay: i * STAGGER_MS,
          easing: Easing.out(Easing.quad),
          useNativeDriver: true,
        }).start();
      });
    } else if (mounted) {
      // Close: everything fades out together with no stagger.
      Animated.timing(openness, {
        toValue: 0,
        duration: CLOSE_DURATION,
        easing: Easing.out(Easing.quad),
        useNativeDriver: true,
      }).start();
      Animated.parallel(
        itemAnims.map((v) =>
          Animated.timing(v, {
            toValue: 0,
            duration: CLOSE_DURATION,
            easing: Easing.out(Easing.quad),
            useNativeDriver: true,
          }),
        ),
      ).start(({ finished }) => {
        if (finished) setMounted(false);
      });
    }
  }, [open, itemAnims, openness, mounted]);

  const rotate = openness.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '45deg'] });
  const backdropOpacity = openness;

  const handleItemPress = (item: SpeedDialItem) => {
    // Close first so parent's open state updates before the action fires.
    setOpen(false);
    item.onPress();
  };

  return (
    <>
      {mounted && (
        <Animated.View
          pointerEvents={open ? 'auto' : 'none'}
          style={[styles.backdrop, { opacity: backdropOpacity }]}
        >
          <Pressable style={StyleSheet.absoluteFill} onPress={() => setOpen(false)} />
        </Animated.View>
      )}

      {/* Mini-FAB stack. Rendered above the primary; items[0] sits closest to the primary. */}
      {mounted && (
        <View pointerEvents={open ? 'box-none' : 'none'} style={styles.stack}>
          {items.map((item, i) => {
            // i=0 is closest to the primary (smallest offset above it). i=last is highest.
            const offsetFromPrimary = (i + 1) * (MINI_SIZE + STACK_GAP);
            const anim = itemAnims[i];
            const opacity = anim;
            const translateY = anim.interpolate({ inputRange: [0, 1], outputRange: [8, 0] });
            return (
              <Animated.View
                key={`${item.label}-${i}`}
                style={[
                  styles.row,
                  { bottom: offsetFromPrimary, opacity, transform: [{ translateY }] },
                ]}
              >
                <Pressable
                  accessibilityRole="button"
                  accessibilityLabel={item.label}
                  onPress={() => handleItemPress(item)}
                  style={({ pressed }) => [
                    styles.pill,
                    { backgroundColor: t.surface, borderColor: t.border },
                    pressed && { opacity: 0.75 },
                  ]}
                >
                  <Text style={[styles.pillText, { color: t.text }]}>{item.label}</Text>
                </Pressable>
                <Pressable
                  accessibilityRole="button"
                  accessibilityLabel={item.label}
                  onPress={() => handleItemPress(item)}
                  style={({ pressed }) => [
                    styles.mini,
                    { backgroundColor: t.accent },
                    pressed && { opacity: 0.85 },
                  ]}
                >
                  <Icon name={item.icon} size={22} color="#fff" />
                </Pressable>
              </Animated.View>
            );
          })}
        </View>
      )}

      <Pressable
        accessibilityRole="button"
        accessibilityLabel={open ? 'Close menu' : 'More actions'}
        onPress={() => setOpen((v) => !v)}
        style={({ pressed }) => [
          styles.primary,
          { backgroundColor: t.accent },
          pressed && { opacity: 0.9 },
        ]}
      >
        <Animated.View style={{ transform: [{ rotate }] }}>
          <Icon name="plus" size={28} color="#fff" />
        </Animated.View>
      </Pressable>
    </>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  primary: {
    position: 'absolute',
    right: 24,
    bottom: 32,
    width: PRIMARY_SIZE,
    height: PRIMARY_SIZE,
    borderRadius: PRIMARY_SIZE / 2,
    alignItems: 'center',
    justifyContent: 'center',
    // A subtle elevation so the FAB reads above the dimmed backdrop.
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 6,
  },
  // The stack is anchored to the same right/bottom as the primary; each row
  // is positioned absolutely with its own `bottom` offset.
  stack: {
    position: 'absolute',
    right: 24,
    bottom: 32,
    // Width 0 so children, which are absolutely positioned with right:0, anchor cleanly.
    width: 0,
    height: 0,
  },
  row: {
    position: 'absolute',
    right: (PRIMARY_SIZE - MINI_SIZE) / 2, // center mini under primary horizontally
    flexDirection: 'row',
    alignItems: 'center',
    // Reverse so the mini sits on the right and the pill flows out to its left.
    // We render pill first, then mini, with the row right-anchored.
  },
  pill: {
    height: 36,
    paddingHorizontal: 14,
    borderRadius: 18,
    borderWidth: StyleSheet.hairlineWidth,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: PILL_GAP,
  },
  pillText: {
    fontSize: 14,
    fontWeight: '500',
  },
  mini: {
    width: MINI_SIZE,
    height: MINI_SIZE,
    borderRadius: MINI_SIZE / 2,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 4,
  },
});
