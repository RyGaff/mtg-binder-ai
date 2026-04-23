import { useStore } from '../store/useStore';
import { themes, type Theme } from './themes';

// Backfill shared tokens onto persisted custom themes that predate them.
function hydrate(t: Theme | null | undefined): Theme {
  if (!t) return themes.dark;
  return {
    ...themes.dark,
    ...t,
  };
}

export function useTheme(): Theme {
  return useStore((s) => {
    if (s.theme.startsWith('custom-')) {
      const index = Number(s.theme.split('-')[1]) as 0 | 1 | 2;
      return hydrate(s.customThemes[index]);
    }
    return themes[s.theme as 'dark' | 'light' | 'amoled'] ?? themes.dark;
  });
}

/** Rough perceived-luminance of a hex color. 0 = black, 255 = white. Returns -1 on bad input. */
function luminance(hex: string): number {
  if (!/^#?[0-9a-f]{6}$/i.test(hex)) return -1;
  const clean = hex.replace('#', '');
  if (clean.length !== 6) return -1;
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  // Rec. 601 weights — good enough for "is this dark?" decisions.
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

export function isDarkTheme(theme: Theme): boolean {
  const l = luminance(theme.bg);
  // Safe default for MTG Binder AI (dark-first) when bg is malformed.
  if (l < 0) return true;
  return l < 128;
}

/** iOS TextInput keyboardAppearance matching the current theme. */
export function useKeyboardAppearance(): 'dark' | 'light' {
  const theme = useTheme();
  return isDarkTheme(theme) ? 'dark' : 'light';
}
