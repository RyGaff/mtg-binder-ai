import { Stack } from 'expo-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEffect } from 'react';
import { getDb } from '../src/db/db';
import { checkAndDownload } from '../src/embeddings/downloader';
import { useStore } from '../src/store/useStore';

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

export default function RootLayout() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppInit />
      <Stack>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen
          name="card/[id]"
          options={{
            presentation: 'modal',
            headerShown: false,
            gestureEnabled: false,
          }}
        />
        <Stack.Screen
          name="deck/[id]"
          options={{
            title: 'Deck',
            headerStyle: { backgroundColor: '#111318' },
            headerShadowVisible: false,
            headerTintColor: '#4ecdc4',
          }}
        />
        <Stack.Screen
          name="profile"
          options={{
            presentation: 'modal',
            headerShown: false,
            gestureEnabled: true,
          }}
        />
        <Stack.Screen
          name="theme-editor"
          options={{
            presentation: 'modal',
            headerShown: false,
            gestureEnabled: true,
          }}
        />
      </Stack>
    </QueryClientProvider>
  );
}
