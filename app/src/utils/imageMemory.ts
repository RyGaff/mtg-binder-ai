import { useCallback } from 'react';
import { useFocusEffect } from 'expo-router';
import { Image } from 'expo-image';

/** Clears expo-image's process-wide memory cache when the screen blurs.
 *  Apply to image-heavy list screens (search, binder, decks, deck detail) so
 *  scroll-accumulated bitmaps don't survive after the user moves on. Disk
 *  cache is untouched, so re-entry re-decodes from disk (cheap). */
export function useImageMemoryCleanupOnBlur(): void {
  useFocusEffect(
    useCallback(() => {
      return () => {
        // Fire-and-forget; clearMemoryCache returns Promise<boolean>.
        Image.clearMemoryCache().catch(() => {});
      };
    }, []),
  );
}
