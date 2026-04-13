import { File } from 'expo-file-system';
import { getEmbeddingsFile, getVersionFile, clearEmbeddingCache } from './parser';

/**
 * URL of the JSON manifest. Update this when hosting is configured.
 * Manifest shape: { "version": "YYYY-MM-DD", "url": "https://..." }
 */
// TODO: replace with production URL before release
const MANIFEST_URL = 'https://your-cdn.example.com/embeddings/manifest.json';

type Manifest = { version: string; url: string };
type EmbeddingStatus = 'idle' | 'downloading' | 'error';

/**
 * Fetch the manifest, compare to locally stored version, and download the
 * embedding binary if a newer version is available. Calls setStatus to
 * communicate progress to the Zustand store.
 *
 * If embeddings already exist locally, sets idle immediately and checks for
 * updates silently in the background.
 */
export async function checkAndDownload(
  setStatus: (status: EmbeddingStatus) => void
): Promise<void> {
  try {
    const embeddingsFile = getEmbeddingsFile();
    const versionFile = getVersionFile();
    const localVersion = await getLocalVersion();
    console.log('[embeddings] fileExists:', embeddingsFile.exists, 'localVersion:', localVersion);

    if (embeddingsFile.exists && localVersion) {
      console.log('[embeddings] local file present, setting idle');
      setStatus('idle');
      checkForUpdate(localVersion).catch(() => {});
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
      return;
    }

    setStatus('downloading');
    await File.downloadFileAsync(manifest.url, embeddingsFile, { idempotent: true });
    versionFile.write(manifest.version);
    clearEmbeddingCache();
    setStatus('idle');
  } catch (e) {
    console.log('[embeddings] checkAndDownload error:', e);
    setStatus('error');
  }
}

async function getLocalVersion(): Promise<string | null> {
  const versionFile = getVersionFile();
  if (!versionFile.exists) return null;
  return versionFile.text();
}

/** Silently check for a newer version and download in background. */
async function checkForUpdate(localVersion: string): Promise<void> {
  const response = await fetch(MANIFEST_URL);
  if (!response.ok) return;
  const manifest: Manifest = await response.json();
  if (manifest.version === localVersion) return;
  await File.downloadFileAsync(manifest.url, getEmbeddingsFile(), { idempotent: true });
  getVersionFile().write(manifest.version);
  clearEmbeddingCache();
}
