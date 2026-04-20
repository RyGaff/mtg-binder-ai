import { File } from 'expo-file-system';
import {
  getEmbeddingsFile, getVersionFile,
  getImageEmbeddingsFile, getImageVersionFile,
  clearEmbeddingCache,
} from './parser';

/**
 * URL of the JSON manifest.
 *
 * Hosted via GitHub Releases on the `mtg-binder-ai` repo — bump the release
 * tag when shipping new embeddings and the app will download on next launch.
 *
 * v1 shape (legacy):   { "version": "YYYY-MM-DD", "url": "https://..." }
 * v2 shape (optional): also includes
 *   "image_embeddings":       { url, version, bytes, sha256 }
 *   "image_encoder_ios":      { ... }
 *   "image_encoder_android":  { ... }
 */
const MANIFEST_URL =
  'https://github.com/RyGaff/mtg-binder-ai/releases/download/embeddings-v2/manifest.json';

type ManifestEntry = { url: string; version: string; bytes?: number; sha256?: string };

type Manifest = {
  // v1 legacy top-level fields — still used for text embeddings
  version?: string;
  url?: string;
  // v2 optional fields
  image_embeddings?: ManifestEntry;
  image_encoder_ios?: ManifestEntry;
  image_encoder_android?: ManifestEntry;
};

type EmbeddingStatus = 'idle' | 'downloading' | 'error';

/**
 * Fetch the manifest and download both text embeddings (legacy) and image
 * embeddings (optional v2) if newer versions are available. Status reflects
 * the TEXT download — image download runs in parallel and reports its own
 * status via a separate setter if desired.
 */
export async function checkAndDownload(
  setStatus: (status: EmbeddingStatus) => void
): Promise<void> {
  try {
    const embeddingsFile = getEmbeddingsFile();
    const localVersion = await getLocalVersion();
    console.log('[embeddings] fileExists:', embeddingsFile.exists, 'localVersion:', localVersion);

    if (embeddingsFile.exists && localVersion) {
      console.log('[embeddings] local file present, setting idle');
      setStatus('idle');
      // Run both update checks silently in background
      checkForUpdate(localVersion).catch(() => {});
      checkAndDownloadImage(() => {}).catch(() => {});
      return;
    }

    const response = await fetch(MANIFEST_URL);
    if (!response.ok) {
      setStatus('error');
      return;
    }
    const manifest: Manifest = await response.json();

    if (manifest.version === (await getLocalVersion()) && embeddingsFile.exists) {
      setStatus('idle');
      checkAndDownloadImage(() => {}).catch(() => {});
      return;
    }

    if (!manifest.version || !manifest.url) {
      setStatus('error');
      return;
    }

    setStatus('downloading');
    await File.downloadFileAsync(manifest.url, embeddingsFile, { idempotent: true });
    getVersionFile().write(manifest.version);
    clearEmbeddingCache();
    setStatus('idle');

    // Fire image download in the background — don't block text-embeddings UI.
    checkAndDownloadImage(() => {}).catch(() => {});
  } catch (e) {
    console.log('[embeddings] checkAndDownload error:', e);
    setStatus('error');
  }
}

/**
 * Check the manifest for an `image_embeddings` entry and download if the
 * local copy is stale or missing. Safe to call when the manifest omits the
 * entry — simply sets status to 'idle' and returns.
 */
export async function checkAndDownloadImage(
  setStatus: (status: EmbeddingStatus) => void
): Promise<void> {
  try {
    const response = await fetch(MANIFEST_URL);
    if (!response.ok) {
      setStatus('error');
      return;
    }
    const manifest: Manifest = await response.json();
    const entry = manifest.image_embeddings;
    if (!entry) {
      setStatus('idle');
      return;
    }

    const file = getImageEmbeddingsFile();
    const versionFile = getImageVersionFile();
    const local = await readVersion(versionFile);
    if (file.exists && local === entry.version) {
      setStatus('idle');
      return;
    }

    setStatus('downloading');
    await File.downloadFileAsync(entry.url, file, { idempotent: true });
    versionFile.write(entry.version);
    clearEmbeddingCache();
    setStatus('idle');
  } catch (e) {
    console.log('[embeddings] checkAndDownloadImage error:', e);
    setStatus('error');
  }
}

async function getLocalVersion(): Promise<string | null> {
  const versionFile = getVersionFile();
  if (!versionFile.exists) return null;
  return versionFile.text();
}

async function readVersion(f: File): Promise<string | null> {
  if (!f.exists) return null;
  try { return (await f.text()).trim() || null; } catch { return null; }
}

/** Silently check for a newer text-embeddings version and download in background. */
async function checkForUpdate(localVersion: string): Promise<void> {
  const response = await fetch(MANIFEST_URL);
  if (!response.ok) return;
  const manifest: Manifest = await response.json();
  if (!manifest.version || manifest.version === localVersion) return;
  if (!manifest.url) return;
  await File.downloadFileAsync(manifest.url, getEmbeddingsFile(), { idempotent: true });
  getVersionFile().write(manifest.version);
  clearEmbeddingCache();
}
