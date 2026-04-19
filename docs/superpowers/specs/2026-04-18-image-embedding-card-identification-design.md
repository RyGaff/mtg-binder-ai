# Image-Embedding Card Identification

**Date:** 2026-04-18
**Status:** Approved — pre-implementation scaffolding

## Problem

OCR-based card identification is unreliable on real-world photos:
misreads on stylized glyphs, retro-frame layouts without a standard
collector-number region, and fingers occluding the bottom-left info block
all cause misses. Card **artwork** is a stronger, more universal signal —
every printing has a canonical image that a trained encoder can
fingerprint with near-perfect accuracy.

This spec defines the consumer-side contract and scaffolding so a
separate model-training repo can drop in its artifacts without any
code change on the app side.

## Scope

In this spec / this repo (`mtg-binder-ai`):

- Define the artifact contract (model format, embeddings format, manifest).
- Add an on-device image-embedding search module that is **dormant**
  until artifacts exist — shipped behind feature detection.
- Add a local-first card-details cache so successful scans minimize
  Scryfall API calls (collection DB → session cache → Scryfall).
- Wire a native CoreML/TFLite inference bridge sized for when the
  encoder file exists.
- Leave OCR + Scryfall set/collector lookup in place as a fallback.

Explicitly out of scope (moved to a new `mtg-card-encoder` repo):

- Training the image encoder.
- Downloading Scryfall card images in bulk.
- Producing `card_encoder.mlmodel` / `.tflite`.
- Producing `card_embeds_v2.bin`.

A companion prompt for the training repo ships alongside this spec
(see "Spin-off project" at the end).

## Artifact Contract

Any pipeline satisfying these four artifacts can be dropped in.

### A — Encoder model

| Aspect | Spec |
|---|---|
| iOS file | `card_encoder.mlmodel` (CoreML, Float32) |
| Android file | `card_encoder.tflite` (TensorFlow Lite, Float32) |
| Input | `1×224×224×3` RGB tensor, `[0, 1]` range (`uint8 / 255`). Any mean/std normalization baked into Layer 0. |
| Output | `1×256` Float32 tensor, **L2-normalized** (dot-product = cosine similarity). |
| Size target | ≤5 MB per file. |
| Inference target | ≤100 ms on iPhone 12 / Pixel 6. |

### B — Embeddings binary

Extends the existing `embeddings.bin` schema (already parsed by
`app/src/embeddings/parser.ts`) with a version field so image and text
embeddings can coexist.

```
Header:
  4 bytes      magic       = 0x4D 0x54 0x47 0x45   // literal ASCII 'M','T','G','E'
                                                    // (read big-endian = 0x4D544745)
  uint32 LE    version     = 2            // 2 = image embeddings
  uint32 LE    count       N              // number of cards
  uint32 LE    dim         D = 256
  uint32 LE    model_hash                 // SHA-256[:4] of encoder file — identity check

Records (N times):
  36 bytes     scryfall_id (ASCII UUID string)
  D × 4 bytes  float32 LE embedding (L2-normalized)
```

The magic word is the only big-endian field: writers MUST emit bytes in the
order `[0x4D, 0x54, 0x47, 0x45]` regardless of host endianness. All other
multi-byte fields (uint32s, float32s) are little-endian.

File size at `N = 35000, D = 256`: ~35 MB Float32 (unquantized v1).
`model_hash` exists solely so the runtime can refuse to use an
embeddings file that doesn't match the encoder it was built against.

### C — CDN manifest

The existing `embeddings/downloader.ts` already reads a JSON manifest.
Extended schema:

```json
{
  "text_embeddings":        { "url": "...", "version": "2025-04-18-text-v1", "bytes": 55000000, "sha256": "..." },
  "image_embeddings":       { "url": "...", "version": "2025-04-18-img-v1", "bytes": 35000000, "sha256": "..." },
  "image_encoder_ios":      { "url": "...", "version": "2025-04-18-img-v1", "bytes": 3800000,  "sha256": "..." },
  "image_encoder_android":  { "url": "...", "version": "2025-04-18-img-v1", "bytes": 4100000,  "sha256": "..." }
}
```

**Coupling rule:** `image_embeddings.version` must equal
`image_encoder_*.version` — embeddings are only valid for the exact
encoder that produced them. The downloader verifies this on install
and refuses mismatched pairs.

### D — Integration hooks (this repo)

Built now so the surface is frozen before the encoder lands:

```ts
// app/src/embeddings/imageEncoder.ts
/** Loads the bundled CoreML/TFLite model once; runs inference on the rectified
 *  card image URI. Returns a 256-dim L2-normalized embedding, or null when
 *  the encoder artifact is not yet installed on this build. */
export async function encodeCardImage(uri: string): Promise<Float32Array | null>;

/** True once both encoder and image-embeddings file are present and valid. */
export async function isImageSearchReady(): Promise<boolean>;

// app/src/embeddings/imageSearch.ts
export type ImageMatch = {
  scryfallId: string;
  score: number;                           // 0–1 cosine similarity
  topK: Array<{ scryfallId: string; score: number }>;
};

/** Full pipeline: encode → NN search → top-1. Returns null when
 *  isImageSearchReady() is false or no match clears the threshold. */
export async function findCardByImage(uri: string): Promise<ImageMatch | null>;
```

Native bridge extensions (iOS + Android):

```
native module: CardDetector
  detectCardCorners(uri)         // [existing]
  encodeImage(uri) -> float[256] // [new] — bundled .mlmodel/.tflite inference
                                 //         returns null when the file is absent
```

## On-Device Scan Flow

```
rectified 400×560 card
         ↓
  isImageSearchReady()
   ├─ yes → findCardByImage(uri)          // primary path
   │           ├─ match score ≥ 0.75 → commit top-1
   │           ├─ 0.55 – 0.75          → show top-3 chooser
   │           └─ < 0.55               → fall through to OCR
   └─ no  → OCR pipeline [existing]        // legacy fallback
         ↓
  scryfall_id  →  local-first lookup (see next)  → display card
```

## Local-First Card Lookup

Once a `scryfall_id` is resolved (by image or OCR), minimize network
calls with a three-tier lookup in this order:

```
1. Session scan cache (in-memory Map<scryfall_id, CachedCard>)
   └─ cleared when the Scan tab unmounts
2. Collection DB (existing src/db/cards.ts — cards the user already owns)
3. Scryfall API (only on miss — result hydrates tier 1 and 2)
```

New wrapper in `src/api/cards.ts`:

```ts
export async function getCardById(id: string): Promise<CachedCard>;
// Checks session cache → DB → Scryfall, in order. Hydrates upward.
```

Every successful scan calls `getCardById(id)` instead of going directly
to Scryfall. Recently scanned + already-collected cards become free;
only unknown cards hit the network.

## Component Design

### `app/src/embeddings/imageEncoder.ts`

Thin wrapper around the native bridge. Handles:
- Lazy model-file presence check (returns `null` if missing).
- URI decoding and handing the 400×560 rectified image to native.
- Ensures output is Float32Array of length 256.
- Caches the "model available" bool for the session after first check.

### `app/src/embeddings/imageSearch.ts`

- On first call, loads `card_embeds_v2.bin` via the existing
  `parser.ts` (extended to handle `version=2`, `dim=256`, scryfall_id only).
- Verifies `model_hash` in the header matches SHA-256 of the installed
  encoder file; if not, refuses to load and the hook reports
  `isImageSearchReady = false`.
- Linear scan top-K cosine-similarity search (35k × 256 ≈ 9M MACs,
  ~30 ms on phone — no ANN library needed).
- Returns top-3 by default; caller decides threshold.

### Native bridge (iOS + Android)

Extends the existing `CardDetector` module:

- iOS: adds `encodeImage:` method. Loads bundled `card_encoder.mlmodel`
  on first call, keeps a `VNCoreMLModel` handle cached. Input comes
  from a file URI; we decode to UIImage → CVPixelBuffer → 224×224.
- Android: `encodeImageNative` via JNI. TFLite interpreter held as a
  field, shared across calls.
- Both return `null`/empty when the model file is not found in the
  bundle — on current builds (no encoder shipped yet), every call
  safely reports "not ready".

### Parser extension

`app/src/embeddings/parser.ts` gains:
- Magic + version check. If `version == 2`, parse fixed-36-byte
  scryfall_id + `dim` × float32; no name field. If `version == 1`
  (text embeddings), existing code path.
- Returns a typed result distinguishing the two.

### `src/api/cards.ts` (new)

Session cache is a module-scoped `Map`. Cleared by a new
`clearSessionCardCache()` function called from scan-screen unmount.
Falls back through DB and Scryfall in the order above.

## File-Format Parser — Version 2

```ts
interface ImageEmbeddingsV2 {
  byId: Map<string, Float32Array>;  // scryfall_id → 256-dim normalized vector
  dim:  256;
  modelHash: number;                // for validation against installed encoder
}
```

No `byName` index — image embeddings are always scryfall_id-keyed
because the visual identity of a printing is unique.

## Testing

**Unit tests:**
- Parser test: synthetic v1 and v2 buffers round-trip correctly.
- Search test: mock embedding file + known queries; assert top-1 and top-3
  scores match expected.
- Session-cache test: DB hit bypasses Scryfall; Scryfall miss hydrates DB.

**Integration tests (device or simulator):**
- Dormant path: with no encoder file bundled, `isImageSearchReady` = false;
  scan pipeline follows OCR fallback unchanged.
- Ready path: stub encoder model returning known vectors; stub embeddings
  file; assert a scan locates the expected scryfall_id.

## Error Handling

| Condition | Behavior |
|---|---|
| Encoder file missing | `encodeCardImage` returns null; scan pipeline uses OCR. |
| Embeddings file missing | `isImageSearchReady` = false; OCR path only. |
| `model_hash` mismatch | Embeddings file refused; logs a warning; OCR path only. |
| Match below 0.55 | Show the top-3 chooser to the user, or prompt re-scan. |
| Scryfall miss on new card | Request times out → show "Unknown card" with retry. |

## Files Modified / Created

| File | Action |
|---|---|
| `app/src/embeddings/parser.ts` | Modify — support v2 header + scryfall_id-only records |
| `app/src/embeddings/imageEncoder.ts` | Create — native-bridge wrapper |
| `app/src/embeddings/imageSearch.ts` | Create — load + NN search |
| `app/src/api/cards.ts` | Create — local-first `getCardById` |
| `app/app/(tabs)/scan.tsx` | Modify — wire image-first lookup, OCR fallback |
| `app/src/scanner/ocr.ts` | Modify — factor the Scryfall call through `getCardById` |
| `modules/card-detector/ios/CardDetectorBridge.h` | Modify — `encodeImage:` signature |
| `modules/card-detector/ios/CardDetectorBridge.mm` | Modify — CoreML model load + inference |
| `modules/card-detector/android/src/main/java/.../CardDetectorModule.kt` | Modify — TFLite inference |
| `docs/superpowers/specs/2026-04-18-image-embedding-card-identification-design.md` | Create — this file |

All native changes gracefully no-op when model file is absent. App
ships unchanged (behavior-wise) until the training repo publishes
artifacts and the manifest is updated.

## Spin-off Project — Prompt for the Training Repo

```
You are starting a new project — `mtg-card-encoder` — in a fresh repo.

## Goal
Train a compact image-embedding model that identifies any Magic: The Gathering
card from a photograph. The model ships on-device in a mobile app (iOS +
Android) that already detects card corners and produces a rectified 400×560
RGB card image. Your job: given that rectified image, produce a 256-dim
embedding that nearest-neighbor-matches to the correct card's Scryfall ID
against a precomputed database of all ~35,000 unique Magic cards.

Target accuracy: ≥98% top-1 on real-world scanned cards (rotated, glare,
lighting variation, JPEG compression). Fallback to top-3 is acceptable.

## Deliverables (CDN-ready)

1. `card_encoder.mlmodel` — CoreML, Float32, ≤5 MB
2. `card_encoder.tflite` — TensorFlow Lite, Float32, ≤5 MB
3. `card_embeds_v2.bin` — precomputed embeddings for all Scryfall unique
   cards (`unique_artwork` bulk endpoint)
4. `manifest.json` — versioned descriptor with sha256 of each artifact

## Strict I/O Contract (non-negotiable — consumed by the mobile app)

Encoder:
- Input:  1×224×224×3 RGB tensor, values in [0, 1] (uint8 / 255). Any
  ImageNet mean/std normalization must be baked into Layer 0 of the model.
- Output: 1×256 Float32, L2-normalized (dot-product = cosine similarity).
- Inference ≤100 ms on iPhone 12 / Pixel 6.

Embeddings binary:
  Header:
    4 bytes     magic bytes [0x4D, 0x54, 0x47, 0x45] ('MTGE', read big-endian)
    uint32 LE   version     = 2
    uint32 LE   count       N
    uint32 LE   dim         D = 256
    uint32 LE   model_hash  (first 32 bits of SHA-256 of the encoder file)
  Records (N times):
    36 bytes    scryfall_id (ASCII UUID string)
    D × 4 bytes float32 LE embedding (L2-normalized)

manifest.json:
  {
    "image_embeddings":       { "url": "...", "version": "<tag>", "bytes": N, "sha256": "..." },
    "image_encoder_ios":      { "url": "...", "version": "<tag>", "bytes": N, "sha256": "..." },
    "image_encoder_android":  { "url": "...", "version": "<tag>", "bytes": N, "sha256": "..." }
  }

The `version` string on the encoder files and the embeddings file MUST match.

## Recommended approach

1. Architecture: MobileNetV3-Small backbone (ImageNet pretrained), replace
   classifier head with:  GAP → Linear(576 → 256) → L2Normalize.
2. Loss: BatchHard Triplet Loss. Mine hard positives (same-card
   augmentations) and hard negatives (visually similar different cards,
   e.g. different printings of same mana cost / type).
3. Training data: download `unique_artwork` JSON from Scryfall
   (https://scryfall.com/docs/api/bulk-data), download each card image at
   normal resolution. ~35k images.
4. Augmentation pipeline (every training sample):
   - Random perspective warp (angled photos)
   - Random rotation ±10°
   - Random crop 80–100% of the rectified area
   - Photometric: brightness/contrast/gamma jitter, Gaussian blur, JPEG
     compression artifacts (quality 40–95), additive Gaussian noise
   - Specular glare simulation (random bright blob overlay)
   - Color-temperature shift (±500K)
5. Optimizer: AdamW, lr=1e-3, cosine schedule, batch=64, ~30 epochs.
6. Export: CoreML via `coremltools`, TFLite via `tf.lite.TFLiteConverter`.
   Both with Float32 (no quantization for v1; revisit later).
7. Embeddings generation: run the trained model over each Scryfall card
   image (NO augmentation — canonical art only), pack into the binary
   format above.

## Test plan

1. Hold out 10% of cards for eval. Heavy augmentation. Top-1 ≥98%.
2. Real-world test set (~50 phone photos with ground-truth scryfall_ids)
   for final acceptance. Top-1 ≥95%.
3. Measure iOS CoreML inference time on iPhone 12 / simulator.

## Not in scope

- iOS / Android app integration (consumer repo).
- Card detection / rectification (already solved in the consumer app).
- OCR (replaced by this system).
- Similar-card recommendations (different feature).

## Start with

1. `README.md` — project description + the contract above.
2. `src/download_images.py` — fetch Scryfall unique_artwork + images to `data/`.
3. `src/train.py` — training loop with hyperparameters above.
4. `src/export.py` — write `card_encoder.mlmodel`, `card_encoder.tflite`,
   `card_embeds_v2.bin`, `manifest.json`.
5. `src/evaluate.py` — top-1 accuracy on held-out augmented set.

Prefer PyTorch + `timm` for the backbone + `pytorch_metric_learning` for
triplet loss. Python 3.10+, GPU required for training.
```

## TODO — Future Integration (separate plan, after encoder is produced)

- [ ] Update `manifest.json` entries with real CDN URLs for the
      image artifacts.
- [ ] Bundle `card_encoder.mlmodel` + `.tflite` into the app binary
      (under `modules/card-detector/ios/` and `.../android/assets/`).
- [ ] End-to-end test: scan a known card, assert top-1 scryfall_id matches.
- [ ] Telemetry: log match scores in dev builds to tune the 0.55 / 0.75
      thresholds against real-world scans.
- [ ] OCR deprecation path: after 2 weeks of image-search data, decide
      whether to keep OCR as fallback or remove it entirely.
