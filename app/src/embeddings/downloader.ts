import { File } from 'expo-file-system';
import {
  getEmbeddingsFile, getVersionFile,
  getImageEmbeddingsFile, getImageVersionFile,
  clearEmbeddingCache,
} from './parser';

/**
 * Hosted via GitHub Releases — bump tag when shipping new embeddings.
 * v1 shape: { version, url }
 * v2 shape also includes: image_embeddings, image_encoder_ios, image_encoder_android
 */
const MANIFEST_URL =
  'https://github.com/RyGaff/mtg-binder-ai/releases/download/embeddings-v2/manifest.json';

type ManifestEntry = { url: string; version: string; bytes?: number; sha256?: string };

type Manifest = {
  version?: string;
  url?: string;
  image_embeddings?: ManifestEntry;
  image_encoder_ios?: ManifestEntry;
  image_encoder_android?: ManifestEntry;
};

type EmbeddingStatus = 'idle' | 'downloading' | 'error';

async function fetchManifest(): Promise<Manifest | null> {
  const response = await fetch(MANIFEST_URL);
  return response.ok ? (await response.json() as Manifest) : null;
}

export async function checkAndDownload(
  setStatus: (status: EmbeddingStatus) => void,
): Promise<void> {
  try {
    const embeddingsFile = getEmbeddingsFile();
    const localVersion = await getLocalVersion();

    if (embeddingsFile.exists && localVersion) {
      setStatus('idle');
      checkForUpdate(localVersion).catch(() => {});
      checkAndDownloadImage(() => {}).catch(() => {});
      return;
    }

    const manifest = await fetchManifest();
    if (!manifest) { setStatus('error'); return; }

    if (manifest.version === (await getLocalVersion()) && embeddingsFile.exists) {
      setStatus('idle');
      checkAndDownloadImage(() => {}).catch(() => {});
      return;
    }

    if (!manifest.version || !manifest.url) { setStatus('error'); return; }

    setStatus('downloading');
    await File.downloadFileAsync(manifest.url, embeddingsFile, { idempotent: true });
    getVersionFile().write(manifest.version);
    clearEmbeddingCache();
    setStatus('idle');

    checkAndDownloadImage(() => {}).catch(() => {});
  } catch (e) {
    console.log('[embeddings] checkAndDownload error:', e);
    setStatus('error');
  }
}

export async function checkAndDownloadImage(
  setStatus: (status: EmbeddingStatus) => void,
): Promise<void> {
  try {
    const manifest = await fetchManifest();
    if (!manifest) { setStatus('error'); return; }
    const entry = manifest.image_embeddings;
    if (!entry) { setStatus('idle'); return; }

    const file = getImageEmbeddingsFile();
    const versionFile = getImageVersionFile();
    const local = await readVersion(versionFile);
    if (file.exists && local === entry.version) { setStatus('idle'); return; }

    console.log(`[embeddings:image] downloading ${entry.version}`);
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
  return versionFile.exists ? versionFile.text() : null;
}

async function readVersion(f: File): Promise<string | null> {
  if (!f.exists) return null;
  try { return (await f.text()).trim() || null; } catch { return null; }
}

async function checkForUpdate(localVersion: string): Promise<void> {
  const manifest = await fetchManifest();
  if (!manifest || !manifest.version || manifest.version === localVersion || !manifest.url) return;
  await File.downloadFileAsync(manifest.url, getEmbeddingsFile(), { idempotent: true });
  getVersionFile().write(manifest.version);
  clearEmbeddingCache();
}
