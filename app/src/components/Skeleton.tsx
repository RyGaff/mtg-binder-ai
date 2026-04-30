import { useEffect, useRef } from 'react';
import { Animated, StyleSheet, type StyleProp, type ViewStyle } from 'react-native';
import { useTheme } from '../theme/useTheme';

type Props = {
  width?: number | `${number}%`;
  height?: number;
  radius?: number;
  style?: StyleProp<ViewStyle>;
};

/**
 * Pulsing rectangle used as a loading placeholder. Replaces ActivityIndicator
 * in places where the real content has a known shape — the skeleton occupies
 * the slot the content will eventually fill, so layout doesn't shift when the
 * data lands. Animation runs on the JS thread; we keep the keyframes short
 * (1.2s loop) and only mount when isLoading=true so idle screens pay nothing.
 */
export function Skeleton({ width, height = 16, radius = 6, style }: Props) {
  const t = useTheme();
  const opacity = useRef(new Animated.Value(0.5)).current;
  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(opacity, { toValue: 1, duration: 600, useNativeDriver: true }),
        Animated.timing(opacity, { toValue: 0.5, duration: 600, useNativeDriver: true }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [opacity]);
  return (
    <Animated.View
      // borderRadius / dims set inline so callers can override; theme drives
      // the pulse color, which sits between bg and surface so it reads as
      // "placeholder" in both light and dark themes.
      style={[
        { backgroundColor: t.surfaceAlt, opacity, borderRadius: radius, width: width as number | undefined, height },
        style,
      ]}
    />
  );
}

/** Convenience: a horizontal row of N skeleton cells with optional gap. */
export function SkeletonRow({ count = 1, gap = 6, height = 12, style }: { count?: number; gap?: number; height?: number; style?: StyleProp<ViewStyle> }) {
  return (
    <Animated.View style={[s.row, { gap }, style]}>
      {Array.from({ length: count }).map((_, i) => (
        <Skeleton key={i} height={height} style={{ flex: 1 }} />
      ))}
    </Animated.View>
  );
}

const s = StyleSheet.create({
  row: { flexDirection: 'row', alignItems: 'center' },
});
