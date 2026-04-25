import { encodeImage } from '../../modules/card-detector/src';

let readinessCached: boolean | null = null;

export async function isImageSearchReady(): Promise<boolean> {
  return readinessCached === true;
}

export async function encodeCardImage(rectifiedUri: string): Promise<Float32Array | null> {
  try {
    const vec = await encodeImage(rectifiedUri);
    if (!vec || vec.length !== 256) return null;
    readinessCached = true;
    if (__DEV__) {
      let sumSq = 0, nanCount = 0;
      for (let i = 0; i < vec.length; i++) {
        const v = vec[i];
        if (Number.isNaN(v)) nanCount++;
        sumSq += v * v;
      }
      const head = Array.from(vec.slice(0, 8)).map((n) => n.toFixed(4));
      console.log(`[enc] norm=${Math.sqrt(sumSq).toFixed(4)} nan=${nanCount} head=[${head.join(',')}]`);
    }
    return vec;
  } catch (err) {
    console.warn('[imageEncoder] native call threw:', err);
    return null;
  }
}

/** Test-only hook. Resets the cached readiness flag. */
export function __resetImageEncoderReadiness(): void {
  readinessCached = null;
}
