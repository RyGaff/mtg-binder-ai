import { Stack } from 'expo-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEffect } from 'react';
import { useFonts } from 'expo-font';
import * as SplashScreen from 'expo-splash-screen';
import { getDb } from '../src/db/db';
import { checkAndDownload } from '../src/embeddings/downloader';
import { useStore } from '../src/store/useStore';

SplashScreen.preventAutoHideAsync().catch(() => {});

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 5 * 60 * 1000,
      // short gcTime trims memory; per-query overrides extend for card/synergy/printings
      gcTime: 30 * 60 * 1000,
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
    },
    mutations: { retry: 0 },
  },
});

function AppInit() {
  const setEmbeddingStatus = useStore((s) => s.setEmbeddingStatus);
  useEffect(() => {
    getDb();
    checkAndDownload(setEmbeddingStatus);
  }, [setEmbeddingStatus]);
  return null;
}

const MODAL_OPTS = { presentation: 'modal', headerShown: false, gestureEnabled: true } as const;

export default function RootLayout() {
  const [fontsReady] = useFonts({
    Mana: require('../assets/fonts/Mana.ttf'),
  });

  useEffect(() => {
    if (fontsReady) SplashScreen.hideAsync().catch(() => {});
  }, [fontsReady]);

  if (!fontsReady) return null;

  return (
    <QueryClientProvider client={queryClient}>
      <AppInit />
      <Stack>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen name="card/[id]" options={{ ...MODAL_OPTS, gestureEnabled: false }} />
        <Stack.Screen name="deck/[id]" options={{ headerShown: false }} />
        <Stack.Screen name="profile" options={MODAL_OPTS} />
        <Stack.Screen name="theme-editor" options={MODAL_OPTS} />
      </Stack>
    </QueryClientProvider>
  );
}
