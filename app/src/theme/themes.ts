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
};

export type CustomTheme = Theme & {
  name: CustomThemeName;
  label: string;
};

export const themes: Record<'dark' | 'light' | 'amoled', Theme> = {
  dark: {
    name: 'dark',
    bg: '#111318',
    surface: '#1a1c23',
    surfaceAlt: '#252830',
    border: '#2a2d38',
    text: '#ffffff',
    textSecondary: '#888888',
    accent: '#4ecdc4',
  },
  light: {
    name: 'light',
    bg: '#f2f2f7',
    surface: '#ffffff',
    surfaceAlt: '#e5e5ea',
    border: '#c7c7cc',
    text: '#000000',
    textSecondary: '#6d6d72',
    accent: '#4ecdc4',
  },
  amoled: {
    name: 'amoled',
    bg: '#000000',
    surface: '#0d0d0d',
    surfaceAlt: '#1a1a1a',
    border: '#222222',
    text: '#ffffff',
    textSecondary: '#888888',
    accent: '#4ecdc4',
  },
};