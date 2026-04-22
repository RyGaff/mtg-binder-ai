import { encodeImage } from '../../modules/card-detector/src';

let readinessCached: boolean | null = null;

/**
 * True iff the native encoder is available and has produced a well-shaped
 * embedding at least once this session. Before the first successful scan
 * this returns false; after one good scan it returns true. Designed so
 * the scan pipeline can short-circuit to OCR when the encoder asset is
 * not bundled, without a separate probe call.
 */
export async function isImageSearchReady(): Promise<boolean> {
  return readinessCached === true;
}

/**
 * Run the native encoder on a rectified card crop. Returns null when:
 *   - The encoder asset isn't bundled in this build
 *   - The image can't be decoded
 *   - The native call errors
 *   - The returned vector isn't the expected length (256)
 */
export async function encodeCardImage(rectifiedUri: string): Promise<Float32Array | null> {
  try {
    const vec = await encodeImage(rectifiedUri);
    if (vec && vec.length === 256) {
      readinessCached = true;
      if (__DEV__) {
        let sumSq = 0;
        let nanCount = 0;
        for (let i = 0; i < vec.length; i++) {
          const v = vec[i];
          if (Number.isNaN(v)) nanCount++;
          sumSq += v * v;
        }
        const norm = Math.sqrt(sumSq);
        const head = Array.from(vec.slice(0, 8)).map((n) => n.toFixed(4));
        console.log(`[enc] norm=${norm.toFixed(4)} nan=${nanCount} head=[${head.join(',')}]`);
      }
      return vec;
    }
    return null;
  } catch (err) {
    console.warn('[imageEncoder] native call threw:', err);
    return null;
  }
}

/** Test-only hook. Resets the cached readiness flag. */
export function __resetImageEncoderReadiness(): void {
  readinessCached = null;
}
