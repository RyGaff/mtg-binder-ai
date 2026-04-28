import { useEffect } from 'react';
import { AppState, type AppStateStatus } from 'react-native';
import { Image } from 'expo-image';
import { clearSessionCardCache } from '../api/cards';
import { clearEmbeddingCache } from '../embeddings/parser';

/** Drops process-wide caches when the app backgrounds. iOS releases the
 *  expo-image NSCache automatically on background; Android does not, and
 *  the embedding Float32Array (~26 MB) lingers on both until next read.
 *  Keeping it short: heavy caches go, SQLite + disk image cache stay. */
export function useBackgroundMemoryReset(): void {
  useEffect(() => {
    const onChange = (state: AppStateStatus) => {
      if (state === 'background' || state === 'inactive') {
        Image.clearMemoryCache().catch(() => {});
        clearSessionCardCache();
        clearEmbeddingCache();
      }
    };
    const sub = AppState.addEventListener('change', onChange);
    return () => sub.remove();
  }, []);
}
