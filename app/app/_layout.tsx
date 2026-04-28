import { Stack } from 'expo-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEffect } from 'react';
import { Platform } from 'react-native';
import { useFonts } from 'expo-font';
import * as SplashScreen from 'expo-splash-screen';
import { Image } from 'expo-image';
import { getDb } from '../src/db/db';
import { checkAndDownload } from '../src/embeddings/downloader';
import { useStore } from '../src/store/useStore';
import { useBackgroundMemoryReset } from '../src/utils/memoryHooks';

// SDWebImage's NSCache defaults to unlimited cost. Cap it so a long browse
// session can't accumulate bitmaps past these limits. iOS-only API.
if (Platform.OS === 'ios') {
  Image.configureCache({
    maxMemoryCost: 100 * 1024 * 1024, // 100 MB decoded bitmaps
    maxMemoryCount: 200,              // ≈200 cached images
    maxDiskSize: 250 * 1024 * 1024,   // 250 MB on-disk JPEGs
  });
}

SplashScreen.preventAutoHideAsync().catch(() => {});

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      // Tight defaults to release image-heavy list queries (Scryfall search,
      // collection, decks) shortly after unmount. Per-query overrides extend
      // gcTime for single-card / synergy / printings where refetch is costly.
      staleTime: 30 * 1000,
      gcTime: 60 * 1000,
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
    },
    mutations: { retry: 0 },
  },
});

function AppInit() {
  const setEmbeddingStatus = useStore((s) => s.setEmbeddingStatus);
  useBackgroundMemoryReset();
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
