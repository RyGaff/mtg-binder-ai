export type ThemeName = 'dark' | 'light' | 'amoled' | 'custom-0' | 'custom-1' | 'custom-2';
export type CustomThemeName = Exclude<ThemeName, 'dark' | 'light' | 'amoled'>;

export type Theme = {
  name: ThemeName;
  bg: string;
  surface: string;
  surfaceAlt: string;
  border: string;
  text: string;
  textSecondary: string;
  accent: string;
  foilAccent: string;
  danger: string;
  success: string;
};

export type CustomTheme = Theme & { name: CustomThemeName; label: string };

export const spacing = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32 } as const;
export const radius = { sm: 4, md: 8, lg: 12, xl: 16, pill: 100 } as const;
export const font = { caption: 11, small: 12, body: 14, subhead: 15, title: 18, hero: 22 } as const;
export const MIN_TOUCH = 44;
export const HIT_SLOP_8 = { top: 8, bottom: 8, left: 8, right: 8 };

const SHARED = { foilAccent: '#b8a0e8', danger: '#b71c1c', success: '#1eb464', accent: '#4ecdc4' } as const;

export const themes: Record<'dark' | 'light' | 'amoled', Theme> = {
  dark: {
    name: 'dark',
    bg: '#111318', surface: '#1a1c23', surfaceAlt: '#252830', border: '#2a2d38',
    text: '#ffffff', textSecondary: '#888888',
    ...SHARED,
  },
  light: {
    name: 'light',
    bg: '#f2f2f7', surface: '#ffffff', surfaceAlt: '#e5e5ea', border: '#c7c7cc',
    text: '#000000', textSecondary: '#6d6d72',
    ...SHARED,
  },
  amoled: {
    name: 'amoled',
    bg: '#000000', surface: '#0d0d0d', surfaceAlt: '#1a1a1a', border: '#222222',
    text: '#ffffff', textSecondary: '#888888',
    ...SHARED,
  },
};
