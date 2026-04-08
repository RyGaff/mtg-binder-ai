import { Stack } from 'expo-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEffect } from 'react';
import { getDb } from '../src/db/db';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 5 * 60 * 1000 },
  },
});

export default function RootLayout() {
  useEffect(() => {
    // Initialize DB on app start
    getDb();
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <Stack>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen
          name="card/[id]"
          options={{
            presentation: 'modal',
            headerShown: false,
            gestureEnabled: true,
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
      </Stack>
    </QueryClientProvider>
  );
}
