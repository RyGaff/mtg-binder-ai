import { useEffect } from 'react';
import { AppState, type AppStateStatus } from 'react-native';
import { Image } from 'expo-image';
import { clearSessionCardCache } from '../api/cards';
import { clearEmbeddingCache } from '../embeddings/parser';
import { pruneStaleUnreferencedCards } from '../db/cards';
import { pruneStalePrintings } from '../db/printings';

/** Drops process-wide caches when the app backgrounds. iOS releases the
 *  expo-image NSCache automatically on background; Android does not, and
 *  the embedding Float32Array (~26 MB) lingers on both until next read.
 *  Keeping it short: heavy caches go, SQLite + disk image cache stay. */
export function useBackgroundMemoryReset(): void {
  useEffect(() => {
    const onChange = (state: AppStateStatus) => {
      // 'inactive' fires on Control Center swipe, incoming call, etc. — too
      // transient to justify dropping the embedding map (~26 MB re-parse on
      // resume). Only 'background' = user actually left the app.
      if (state === 'background') {
        Image.clearMemoryCache().catch(() => {});
        clearSessionCardCache();
        clearEmbeddingCache();
        // Background is a natural moment to evict stale SQLite rows — user is
        // away, no queries running, no UI to block. Cheap when there's nothing
        // to drop; bounded by the WHERE clauses.
        runPrune();
      }
    };
    const sub = AppState.addEventListener('change', onChange);
    return () => sub.remove();
  }, []);
}

const PRUNE_INTERVAL_MS = 30 * 60 * 1000;
let lastPruneAt = 0;

function runPrune(): void {
  lastPruneAt = Date.now();
  try {
    pruneStaleUnreferencedCards();
    pruneStalePrintings();
  } catch {
    // best-effort; never throw out of a memory hook
  }
}

/** Periodic in-session prune. A user who never backgrounds the app (rare on
 *  mobile, but possible on tablets) would otherwise let stale rows accumulate
 *  for the entire session. Throttled so heavy interaction doesn't trigger
 *  delete storms. */
export function usePeriodicPrune(): void {
  useEffect(() => {
    // Run once after mount if we're past the interval (covers app launch +
    // long-foreground sessions that never backgrounded).
    if (Date.now() - lastPruneAt > PRUNE_INTERVAL_MS) runPrune();
    const id = setInterval(() => {
      // setInterval fires while foreground; AppState backgrounding doesn't
      // pause it on RN, but the prune is cheap and idempotent.
      if (Date.now() - lastPruneAt > PRUNE_INTERVAL_MS) runPrune();
    }, PRUNE_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);
}
