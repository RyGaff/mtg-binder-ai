import { useMemo } from 'react';
import { useStore } from '../store/useStore';
import { themes, type Theme } from './themes';

function hydrate(t: Theme | null | undefined): Theme {
  return t ? { ...themes.dark, ...t } : themes.dark;
}

export function useTheme(): Theme {
  const themeKey = useStore((s) => s.theme);
  const customTheme = useStore((s) => {
    if (!s.theme.startsWith('custom-')) return null;
    const index = Number(s.theme.split('-')[1]) as 0 | 1 | 2;
    return s.customThemes[index] ?? null;
  });
  return useMemo(
    () => themeKey.startsWith('custom-')
      ? hydrate(customTheme)
      : themes[themeKey as 'dark' | 'light' | 'amoled'] ?? themes.dark,
    [themeKey, customTheme],
  );
}

/** Rough perceived-luminance of a hex color. 0 = black, 255 = white. Returns -1 on bad input. */
function luminance(hex: string): number {
  if (!/^#?[0-9a-f]{6}$/i.test(hex)) return -1;
  const clean = hex.replace('#', '');
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

export function isDarkTheme(theme: Theme): boolean {
  const l = luminance(theme.bg);
  return l < 0 ? true : l < 128;
}

export function useKeyboardAppearance(): 'dark' | 'light' {
  return isDarkTheme(useTheme()) ? 'dark' : 'light';
}
