# Image-Embedding Card Identification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the consumer-side scaffolding for image-embedding card identification. Everything built here is **dormant until the `mtg-card-encoder` repo ships its artifacts** — native bridges return null when the encoder file is absent, the image-search module reports "not ready", and the app falls back to the existing OCR pipeline. Once the artifacts drop in, the image-first path activates with zero further code changes.

**Architecture:** Extend the binary-embeddings parser to handle a v2 header (image embeddings, scryfall_id-only records). Add native CoreML/TFLite inference hooks that gracefully report unavailability. Layer a local-first card-details resolver on top of the existing SQLite DB so scans only hit Scryfall for genuinely new cards. Wire the scan pipeline to try image search first and fall back to OCR.

**Tech Stack:** React Native + Expo, TypeScript, Swift/ObjC + CoreML (iOS), Kotlin + TFLite (Android), Jest for unit tests.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `app/src/embeddings/parser.ts` | Modify | Support magic/version header; v2 scryfall_id-only records; expose parsed version |
| `app/src/embeddings/imageEncoder.ts` | Create | Native-bridge wrapper for `encodeImage`; `isImageSearchReady()` gate |
| `app/src/embeddings/imageSearch.ts` | Create | Load image-embeds file, cosine-similarity search, top-K |
| `app/src/embeddings/downloader.ts` | Modify | Accept v2-manifest entries for image artifacts |
| `app/src/api/cards.ts` | Create | `resolveCardById()` — session cache → DB → Scryfall |
| `app/src/scanner/ocr.ts` | Modify | Route Scryfall calls through `resolveCardById` |
| `app/app/(tabs)/scan.tsx` | Modify | Image-search-first; OCR fallback on miss or not-ready |
| `app/modules/card-detector/ios/CardDetectorBridge.h` | Modify | `encodeImage:` signature |
| `app/modules/card-detector/ios/CardDetectorBridge.mm` | Modify | CoreML model loader + inference (nil if asset absent) |
| `app/modules/card-detector/ios/CardDetectorModule.swift` | Modify | Expose `encodeImage` AsyncFunction |
| `app/modules/card-detector/android/src/main/java/expo/modules/carddetector/CardDetectorModule.kt` | Modify | TFLite interpreter + `encodeImage` AsyncFunction (null if asset absent) |
| `app/modules/card-detector/android/build.gradle` | Modify | Add `org.tensorflow:tensorflow-lite` dependency |
| `app/modules/card-detector/ios/CardDetector.podspec` | Modify | Weak-link CoreML framework |
| `app/modules/card-detector/src/index.ts` | Modify | Typed wrapper for `encodeImage` native call |
| `app/__tests__/embeddings/parser.test.ts` | Modify | V2 header parsing + validation tests |
| `app/__tests__/embeddings/imageSearch.test.ts` | Create | Linear-scan NN test with synthetic embeddings |
| `app/__tests__/api/cards.test.ts` | Create | Session cache + DB + Scryfall fallback tests |

---

## Task 1: Parser v2 Support

**Files:**
- Modify: `app/src/embeddings/parser.ts`
- Modify: `app/__tests__/embeddings/parser.test.ts`

- [ ] **Step 1: Write failing tests for v2 header**

  Append these tests to `app/__tests__/embeddings/parser.test.ts` (create the file if it does not yet exist — check first):

  ```ts
  import { parseEmbeddingBuffer, parseEmbeddingFile } from '../../src/embeddings/parser';

  function buildV2Buffer(records: Array<{ id: string; vec: number[] }>, modelHash = 0xDEADBEEF): ArrayBuffer {
    const dim = records[0].vec.length;
    const recSize = 36 + dim * 4;
    const buf = new ArrayBuffer(20 + records.length * recSize);
    const view = new DataView(buf);
    view.setUint32(0,  0x4D544745, true); // 'MTGE' magic
    view.setUint32(4,  2, true);           // version 2
    view.setUint32(8,  records.length, true);
    view.setUint32(12, dim, true);
    view.setUint32(16, modelHash, true);

    records.forEach((r, i) => {
      const base = 20 + i * recSize;
      // 36-byte scryfall_id, NUL-padded
      for (let j = 0; j < 36; j++) {
        view.setUint8(base + j, j < r.id.length ? r.id.charCodeAt(j) : 0);
      }
      const vec = new Float32Array(buf, base + 36, dim);
      r.vec.forEach((x, k) => vec[k] = x);
    });
    return buf;
  }

  describe('parseEmbeddingBuffer v2 (image embeddings)', () => {
    it('parses a v2 buffer and returns image variant', () => {
      const buf = buildV2Buffer([
        { id: 'aaa-111', vec: [1, 0, 0, 0] },
        { id: 'bbb-222', vec: [0, 1, 0, 0] },
      ], 0x12345678);

      const result = parseEmbeddingBuffer(buf);
      expect(result.version).toBe(2);
      expect(result.dim).toBe(4);
      expect(result.modelHash).toBe(0x12345678);
      expect(result.byId.size).toBe(2);
      expect(Array.from(result.byId.get('aaa-111')!)).toEqual([1, 0, 0, 0]);
      // v2 has no byName
      expect(result.byName.size).toBe(0);
    });
  });

  describe('parseEmbeddingBuffer v1 (text embeddings, backward-compat)', () => {
    it('still parses the legacy no-magic format', () => {
      // Legacy format: [uint32 N][uint32 D][N × (36 id + 64 name + D×f32)]
      const dim = 2;
      const rec = 36 + 64 + dim * 4;
      const buf = new ArrayBuffer(8 + rec);
      const view = new DataView(buf);
      view.setUint32(0, 1, true);       // N
      view.setUint32(4, dim, true);     // D
      for (let j = 0; j < 3; j++) view.setUint8(8 + j, 'zzz'.charCodeAt(j));
      for (let j = 0; j < 8; j++) view.setUint8(8 + 36 + j, 'Testcard'.charCodeAt(j));
      const vec = new Float32Array(buf, 8 + 36 + 64, dim);
      vec[0] = 0.6; vec[1] = 0.8;

      const result = parseEmbeddingBuffer(buf);
      expect(result.version).toBe(1);
      expect(result.byId.size).toBe(1);
      expect(result.byName.get('Testcard')).toBe('zzz');
    });
  });
  ```

- [ ] **Step 2: Run tests to verify they fail**

  ```bash
  cd app && npx jest __tests__/embeddings/parser.test.ts
  ```
  Expected: tests fail — `version`, `dim`, `modelHash` properties don't exist on the return type; the parser has no branching on the magic header.

- [ ] **Step 3: Replace `app/src/embeddings/parser.ts` with v2-aware parser**

  ```ts
  import { File, Paths } from 'expo-file-system';

  export const getEmbeddingsFile = () => new File(Paths.document, 'embeddings.bin');
  export const getVersionFile = () => new File(Paths.document, 'embeddings_version.txt');

  export const getImageEmbeddingsFile = () => new File(Paths.document, 'embeddings_image.bin');
  export const getImageVersionFile  = () => new File(Paths.document, 'embeddings_image_version.txt');

  export type EmbeddingMap = Map<string, Float32Array>;

  export type EmbeddingIndex = {
    version:   1 | 2;
    dim:       number;
    modelHash: number;            // only meaningful for v2 image embeds
    byId:      EmbeddingMap;       // scryfall_id → L2-normalized vector
    byName:    Map<string, string>; // empty for v2
  };

  const MAGIC_MTGE = 0x4D544745;

  let cachedText: EmbeddingIndex | null = null;
  let cachedImage: EmbeddingIndex | null = null;
  let pendingText: Promise<EmbeddingIndex> | null = null;
  let pendingImage: Promise<EmbeddingIndex> | null = null;

  export function clearEmbeddingCache(): void {
    cachedText = null; pendingText = null;
    cachedImage = null; pendingImage = null;
  }

  export async function getEmbeddingMap(): Promise<EmbeddingIndex> {
    if (cachedText) return cachedText;
    if (pendingText) return pendingText;
    pendingText = loadFile(getEmbeddingsFile()).then(i => { cachedText = i; pendingText = null; return i; })
      .catch(e => { pendingText = null; throw e; });
    return pendingText;
  }

  export async function getImageEmbeddingMap(): Promise<EmbeddingIndex> {
    if (cachedImage) return cachedImage;
    if (pendingImage) return pendingImage;
    pendingImage = loadFile(getImageEmbeddingsFile()).then(i => { cachedImage = i; pendingImage = null; return i; })
      .catch(e => { pendingImage = null; throw e; });
    return pendingImage;
  }

  async function loadFile(file: any): Promise<EmbeddingIndex> {
    const buf = await file.arrayBuffer();
    return parseEmbeddingBuffer(buf);
  }

  /**
   * Parse a binary embedding buffer.
   *
   * v2 (image embeddings) — starts with magic 'MTGE' (0x4D544745 little-endian):
   *   uint32 magic, uint32 version=2, uint32 N, uint32 D, uint32 model_hash
   *   N × [36 bytes scryfall_id (NUL-padded)] [D × float32 L2-normalized]
   *
   * v1 (text embeddings, legacy — no magic word):
   *   uint32 N, uint32 D
   *   N × [36 bytes id] [64 bytes name] [D × float32]
   */
  export function parseEmbeddingBuffer(buffer: ArrayBuffer): EmbeddingIndex {
    const view = new DataView(buffer);
    const first = view.getUint32(0, true);

    if (first === MAGIC_MTGE) {
      return parseV2(buffer, view);
    }
    return parseV1(buffer, view);
  }

  function parseV2(buffer: ArrayBuffer, view: DataView): EmbeddingIndex {
    const version   = view.getUint32(4, true);
    const n         = view.getUint32(8, true);
    const dim       = view.getUint32(12, true);
    const modelHash = view.getUint32(16, true);
    const recordSize = 36 + dim * 4;
    const byId: EmbeddingMap = new Map();

    for (let i = 0; i < n; i++) {
      const base = 20 + i * recordSize;
      const idBytes = new Uint8Array(buffer, base, 36);
      const id = String.fromCharCode(...idBytes).replace(/\0/g, '');
      const raw = new Float32Array(buffer, base + 36, dim);
      byId.set(id, normalize(raw));
    }

    return { version: version as 1 | 2, dim, modelHash, byId, byName: new Map() };
  }

  function parseV1(buffer: ArrayBuffer, view: DataView): EmbeddingIndex {
    const n   = view.getUint32(0, true);
    const dim = view.getUint32(4, true);
    const recordSize = 36 + 64 + dim * 4;
    const byId: EmbeddingMap = new Map();
    const byName: Map<string, string> = new Map();

    for (let i = 0; i < n; i++) {
      const base = 8 + i * recordSize;
      const idBytes = new Uint8Array(buffer, base, 36);
      const id = String.fromCharCode(...idBytes).replace(/\0/g, '');
      const nameBytes = new Uint8Array(buffer, base + 36, 64);
      const name = String.fromCharCode(...nameBytes).replace(/\0/g, '');
      const raw = new Float32Array(buffer, base + 100, dim);
      byId.set(id, normalize(raw));
      if (name) byName.set(name, id);
    }

    return { version: 1, dim, modelHash: 0, byId, byName };
  }

  export function normalize(v: Float32Array): Float32Array {
    let sum = 0;
    for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
    const norm = Math.sqrt(sum);
    const out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = norm > 0 ? v[i] / norm : 0;
    return out;
  }
  ```

- [ ] **Step 4: Run tests — verify both v1 and v2 pass**

  ```bash
  cd app && npx jest __tests__/embeddings/parser.test.ts
  ```
  Expected: all tests pass (new v2 tests + existing v1 tests).

- [ ] **Step 5: Commit**

  ```bash
  git add app/src/embeddings/parser.ts app/__tests__/embeddings/parser.test.ts
  git commit -m "feat(embeddings): parser v2 for image embeddings (magic-header + scryfall_id-only records)"
  ```

---

## Task 2: Local-First Card Resolver

**Files:**
- Create: `app/src/api/cards.ts`
- Create: `app/__tests__/api/cards.test.ts`

- [ ] **Step 1: Write failing tests**

  ```ts
  // app/__tests__/api/cards.test.ts
  jest.mock('../../src/db/cards', () => ({
    getCardById: jest.fn(),
    upsertCard: jest.fn(),
    isCardStale: jest.fn(() => false),
  }));

  jest.mock('../../src/api/scryfall', () => ({
    fetchCardById: jest.fn(),
  }));

  import {
    resolveCardById, clearSessionCardCache, getSessionCacheSize,
  } from '../../src/api/cards';
  import * as db from '../../src/db/cards';
  import * as scryfall from '../../src/api/scryfall';

  const CARD = {
    scryfall_id: 'aaa-111', name: 'Lightning Bolt', set_code: 'lea',
    collector_number: '161', mana_cost: '{R}', type_line: 'Instant',
    oracle_text: 'Deal 3 damage.', color_identity: '["R"]',
    image_uri: '', prices: '{}', keywords: '[]',
    cached_at: Date.now(),
  } as const;

  beforeEach(() => {
    clearSessionCardCache();
    jest.clearAllMocks();
  });

  it('serves from session cache on repeated call', async () => {
    (db.getCardById as jest.Mock).mockReturnValue(CARD);
    await resolveCardById('aaa-111');
    await resolveCardById('aaa-111');
    expect(db.getCardById).toHaveBeenCalledTimes(1);
    expect(getSessionCacheSize()).toBe(1);
  });

  it('falls back to DB when not in session cache', async () => {
    (db.getCardById as jest.Mock).mockReturnValue(CARD);
    const result = await resolveCardById('aaa-111');
    expect(result).toEqual(CARD);
    expect(scryfall.fetchCardById).not.toHaveBeenCalled();
  });

  it('falls back to Scryfall on DB miss and hydrates DB + cache', async () => {
    (db.getCardById as jest.Mock).mockReturnValue(null);
    (scryfall.fetchCardById as jest.Mock).mockResolvedValue(CARD);
    const result = await resolveCardById('aaa-111');
    expect(result).toEqual(CARD);
    expect(scryfall.fetchCardById).toHaveBeenCalledWith('aaa-111');
    expect(db.upsertCard).toHaveBeenCalledWith(CARD);
    expect(getSessionCacheSize()).toBe(1);
  });

  it('re-fetches when cached row is stale', async () => {
    (db.getCardById as jest.Mock).mockReturnValue(CARD);
    (db.isCardStale as jest.Mock).mockReturnValue(true);
    (scryfall.fetchCardById as jest.Mock).mockResolvedValue({ ...CARD, cached_at: Date.now() });
    await resolveCardById('aaa-111');
    expect(scryfall.fetchCardById).toHaveBeenCalled();
  });
  ```

- [ ] **Step 2: Run tests to verify they fail**

  ```bash
  cd app && npx jest __tests__/api/cards.test.ts
  ```
  Expected: fail — module `../../src/api/cards` does not exist.

- [ ] **Step 3: Create `app/src/api/cards.ts`**

  ```ts
  import type { CachedCard } from '../db/cards';
  import * as db from '../db/cards';
  import { fetchCardById } from './scryfall';

  /** In-memory cache scoped to the current scan session. Cleared on scan-screen
   *  unmount. A successful Scryfall miss hydrates both this cache and the DB. */
  const sessionCache: Map<string, CachedCard> = new Map();

  /**
   * Resolve a full card record given a Scryfall id. Tries:
   *   1. Session cache  (~1 ms)
   *   2. Local SQLite   (~5 ms) — only if row is fresh
   *   3. Scryfall API   (~200 ms, requires network)
   * On Scryfall hit, writes through to both the DB and the session cache.
   */
  export async function resolveCardById(scryfallId: string): Promise<CachedCard> {
    const cached = sessionCache.get(scryfallId);
    if (cached) return cached;

    const fromDb = db.getCardById(scryfallId);
    if (fromDb && !db.isCardStale(fromDb)) {
      sessionCache.set(scryfallId, fromDb);
      return fromDb;
    }

    const fresh = await fetchCardById(scryfallId);
    db.upsertCard(fresh);
    sessionCache.set(scryfallId, fresh);
    return fresh;
  }

  /** Clear the scan-session cache. Call from the Scan screen's unmount effect. */
  export function clearSessionCardCache(): void {
    sessionCache.clear();
  }

  /** Test/debug hook. */
  export function getSessionCacheSize(): number {
    return sessionCache.size;
  }
  ```

- [ ] **Step 4: Run tests — all pass**

  ```bash
  cd app && npx jest __tests__/api/cards.test.ts
  ```
  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add app/src/api/cards.ts app/__tests__/api/cards.test.ts
  git commit -m "feat(api): resolveCardById with session/DB/Scryfall three-tier cache"
  ```

---

## Task 3: Route OCR Pipeline Through the Resolver

**Files:**
- Modify: `app/src/scanner/ocr.ts`
- Modify: `app/__tests__/scanner/ocr.test.ts`

- [ ] **Step 1: Find current Scryfall calls in `ocr.ts`**

  ```bash
  grep -n "fetchCardBy" app/src/scanner/ocr.ts
  ```
  Expected: two call sites — `fetchCardBySetNumber` (strategy 1) and `fetchCardByName` (strategy 2).

- [ ] **Step 2: Wrap both calls via the resolver**

  The resolver works by scryfall_id, but the OCR pipeline resolves by set+collector or name. Keep the Scryfall API calls as-is for those two cases (no scryfall_id yet), but route through `resolveCardById` for a final hydration pass so the session cache warms up on the returned id.

  In `app/src/scanner/ocr.ts`, add the import near the top:

  ```ts
  import { resolveCardById } from '../api/cards';
  ```

  Replace the two success return paths:

  ```ts
  // Strategy 1 — set_number
  if (parsed) {
    try {
      const card = await fetchCardBySetNumber(parsed.setCode, parsed.collectorNumber);
      // Warm the session cache so later scans of the same card are free.
      const hydrated = await resolveCardById(card.scryfall_id);
      return { strategy: 'set_number', card: hydrated, corners, imageW: imgW, imageH: imgH, ocrText: blText, blText };
    } catch {
      // fall through to name strategy
    }
  }
  ```

  And for strategy 2:

  ```ts
  const card = await fetchCardByName(nameLine.trim());
  const hydrated = await resolveCardById(card.scryfall_id);
  return { strategy: 'name', card: hydrated, corners, imageW: imgW, imageH: imgH, ocrText: tlText, blText };
  ```

- [ ] **Step 3: Update test mocks**

  Add to the `jest.mock` block at the top of `app/__tests__/scanner/ocr.test.ts`:

  ```ts
  jest.mock('../../src/api/cards', () => ({
    resolveCardById: jest.fn(async (id: string) => ({ ...MOCK_CARD, scryfall_id: id })),
  }));
  ```

  And import `MOCK_CARD` above the mock block if not already in scope — reorder if needed so `MOCK_CARD` is defined at file top before mocks reference it. (Jest hoists `jest.mock` so move `MOCK_CARD` to module-level above mocks, or inline the mock return.)

- [ ] **Step 4: Run tests — all still green**

  ```bash
  cd app && npx jest __tests__/scanner/ocr.test.ts
  ```
  Expected: all existing tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add app/src/scanner/ocr.ts app/__tests__/scanner/ocr.test.ts
  git commit -m "feat(scanner): hydrate via resolveCardById so session cache warms on OCR success"
  ```

---

## Task 4: Native Bridge — Add `encodeImage` (iOS)

**Files:**
- Modify: `app/modules/card-detector/ios/CardDetectorBridge.h`
- Modify: `app/modules/card-detector/ios/CardDetectorBridge.mm`
- Modify: `app/modules/card-detector/ios/CardDetectorModule.swift`
- Modify: `app/modules/card-detector/ios/CardDetector.podspec`

- [ ] **Step 1: Declare the bridge method in `CardDetectorBridge.h`**

  Add below the existing `detectCornersFromFileURI:` declaration:

  ```objc
  + (nullable NSArray<NSNumber *> *)encodeImageFromFileURI:(NSString *)uri;
  ```

- [ ] **Step 2: Implement in `CardDetectorBridge.mm`**

  Add at the top:

  ```objc
  #import <CoreML/CoreML.h>
  #import <Vision/Vision.h>
  ```

  Add the helper and bridge method (anywhere inside `@implementation CardDetectorBridge`):

  ```objc
  + (nullable MLModel *)loadEncoderModel {
      static MLModel *model = nil;
      static dispatch_once_t onceToken;
      dispatch_once(&onceToken, ^{
          NSString *path = [[NSBundle mainBundle] pathForResource:@"card_encoder"
                                                           ofType:@"mlmodelc"];
          if (!path) {
              NSLog(@"[CardDetector] card_encoder.mlmodelc not bundled — encodeImage disabled");
              return;
          }
          NSError *err = nil;
          model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path] error:&err];
          if (err) NSLog(@"[CardDetector] failed to load encoder: %@", err);
      });
      return model;
  }

  + (nullable NSArray<NSNumber *> *)encodeImageFromFileURI:(NSString *)uri {
      MLModel *model = [self loadEncoderModel];
      if (!model) return nil;

      NSString *path = [uri hasPrefix:@"file://"] ? [uri substringFromIndex:7] : uri;
      UIImage *ui = [UIImage imageWithContentsOfFile:path];
      if (!ui || !ui.CGImage) return nil;

      // Resize to 224×224 and convert to CVPixelBuffer.
      CGSize targetSize = CGSizeMake(224, 224);
      UIGraphicsBeginImageContextWithOptions(targetSize, YES, 1.0);
      [ui drawInRect:CGRectMake(0, 0, 224, 224)];
      UIImage *resized = UIGraphicsGetImageFromCurrentImageContext();
      UIGraphicsEndImageContext();

      CVPixelBufferRef pb = NULL;
      NSDictionary *attrs = @{ (NSString *)kCVPixelBufferCGImageCompatibilityKey: @YES,
                               (NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey: @YES };
      CVPixelBufferCreate(kCFAllocatorDefault, 224, 224,
                          kCVPixelFormatType_32ARGB,
                          (__bridge CFDictionaryRef)attrs, &pb);
      CVPixelBufferLockBaseAddress(pb, 0);
      CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
      CGContextRef ctx = CGBitmapContextCreate(
          CVPixelBufferGetBaseAddress(pb), 224, 224, 8,
          CVPixelBufferGetBytesPerRow(pb), cs,
          kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
      CGContextDrawImage(ctx, CGRectMake(0, 0, 224, 224), resized.CGImage);
      CGContextRelease(ctx);
      CGColorSpaceRelease(cs);
      CVPixelBufferUnlockBaseAddress(pb, 0);

      NSError *err = nil;
      MLFeatureValue *fv = [MLFeatureValue featureValueWithPixelBuffer:pb];
      MLDictionaryFeatureProvider *input = [[MLDictionaryFeatureProvider alloc]
          initWithDictionary:@{@"input": fv} error:&err];
      CVPixelBufferRelease(pb);
      if (err) return nil;

      id<MLFeatureProvider> output = [model predictionFromFeatures:input error:&err];
      if (err || !output) return nil;

      MLFeatureValue *vec = [output featureValueForName:@"output"];
      MLMultiArray *arr = vec.multiArrayValue;
      if (!arr) return nil;

      NSMutableArray<NSNumber *> *out = [NSMutableArray arrayWithCapacity:arr.count];
      for (int i = 0; i < arr.count; i++) {
          [out addObject:@([arr[@(i)] floatValue])];
      }
      return out;
  }
  ```

  Note: the input/output feature names (`"input"`, `"output"`) must match the CoreML model's signature. The `mtg-card-encoder` repo is instructed to use these names.

- [ ] **Step 3: Expose via Expo Swift module**

  In `app/modules/card-detector/ios/CardDetectorModule.swift`, below the existing `AsyncFunction("detectCardCorners")`:

  ```swift
  AsyncFunction("encodeImage") { (uri: String) -> [NSNumber]? in
      return CardDetectorBridge.encodeImage(fromFileURI: uri)
  }
  ```

- [ ] **Step 4: Podspec — weak-link CoreML so the binary still runs on older devices**

  Add to the `frameworks` line in `CardDetector.podspec`:

  ```ruby
  s.weak_frameworks = 'CoreML'
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add app/modules/card-detector/ios
  git commit -m "feat(ios): CardDetector.encodeImage via CoreML — no-op when asset missing"
  ```

---

## Task 5: Native Bridge — Add `encodeImage` (Android)

**Files:**
- Modify: `app/modules/card-detector/android/src/main/java/expo/modules/carddetector/CardDetectorModule.kt`
- Modify: `app/modules/card-detector/android/build.gradle`

- [ ] **Step 1: Add TFLite dependency to Gradle**

  In `app/modules/card-detector/android/build.gradle`, inside the `dependencies { ... }` block:

  ```gradle
  implementation 'org.tensorflow:tensorflow-lite:2.14.0'
  implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
  ```

- [ ] **Step 2: Extend `CardDetectorModule.kt`**

  Add inside `class CardDetectorModule : Module()`:

  ```kotlin
  import android.graphics.BitmapFactory
  import android.graphics.Bitmap
  import org.tensorflow.lite.Interpreter
  import java.io.FileInputStream
  import java.nio.ByteBuffer
  import java.nio.ByteOrder
  import java.nio.channels.FileChannel

  private var interpreter: Interpreter? = null
  private var interpreterLoaded = false

  private fun loadInterpreter(): Interpreter? {
      if (interpreterLoaded) return interpreter
      interpreterLoaded = true
      try {
          val ctx = appContext.reactContext ?: return null
          val inputStream = FileInputStream(
              ctx.assets.openFd("card_encoder.tflite").fileDescriptor
          )
          val fd = ctx.assets.openFd("card_encoder.tflite")
          val channel = FileInputStream(fd.fileDescriptor).channel
          val buf = channel.map(FileChannel.MapMode.READ_ONLY,
              fd.startOffset, fd.declaredLength)
          interpreter = Interpreter(buf)
      } catch (e: Exception) {
          android.util.Log.w("CardDetector",
              "card_encoder.tflite not in assets — encodeImage disabled", e)
          interpreter = null
      }
      return interpreter
  }
  ```

  And inside `override fun definition() = ModuleDefinition { ... }` add:

  ```kotlin
  AsyncFunction("encodeImage") { uri: String ->
      val itp = loadInterpreter() ?: return@AsyncFunction null
      val path = if (uri.startsWith("file://")) uri.removePrefix("file://") else uri
      val bmp = BitmapFactory.decodeFile(path) ?: return@AsyncFunction null
      val resized = Bitmap.createScaledBitmap(bmp, 224, 224, true)

      val input = ByteBuffer.allocateDirect(4 * 224 * 224 * 3).order(ByteOrder.nativeOrder())
      val pixels = IntArray(224 * 224)
      resized.getPixels(pixels, 0, 224, 0, 0, 224, 224)
      for (p in pixels) {
          input.putFloat(((p shr 16) and 0xFF) / 255f)
          input.putFloat(((p shr 8)  and 0xFF) / 255f)
          input.putFloat((p and 0xFF)            / 255f)
      }
      input.rewind()

      val output = Array(1) { FloatArray(256) }
      itp.run(input, output)
      return@AsyncFunction output[0].toList().map { it.toDouble() }
  }
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add app/modules/card-detector/android
  git commit -m "feat(android): CardDetector.encodeImage via TFLite — no-op when asset missing"
  ```

---

## Task 6: TypeScript Wrapper — `imageEncoder.ts`

**Files:**
- Modify: `app/modules/card-detector/src/index.ts`
- Create: `app/src/embeddings/imageEncoder.ts`

- [ ] **Step 1: Expose `encodeImage` from the native-module package**

  In `app/modules/card-detector/src/index.ts`, add:

  ```ts
  /**
   * Run the bundled image encoder on the file at `imageUri`. Returns a
   * 256-length Float32Array (L2-normalized) or null when the encoder asset
   * is not bundled in this build.
   */
  export async function encodeImage(imageUri: string): Promise<Float32Array | null> {
    const Native = requireNativeModule('CardDetector');
    const raw: number[] | null = await Native.encodeImage(imageUri);
    if (!raw) return null;
    return new Float32Array(raw);
  }
  ```

- [ ] **Step 2: Create the consumer-side wrapper**

  `app/src/embeddings/imageEncoder.ts`:

  ```ts
  import { encodeImage } from '../../modules/card-detector/src';

  let readinessCached: boolean | null = null;

  /**
   * True iff the native encoder is available AND produces a well-shaped
   * embedding. Result is cached after the first call.
   */
  export async function isImageSearchReady(): Promise<boolean> {
    if (readinessCached !== null) return readinessCached;
    // Probe with a 1×1 pixel URI or give up if the module is missing.
    // Since probing requires an actual image, assume ready = false until
    // the first real scan confirms otherwise; flip to true once we get
    // a non-null embedding back.
    readinessCached = false;
    return readinessCached;
  }

  /**
   * Run the native encoder. Returns null when:
   *   - The encoder asset isn't bundled in this build
   *   - The image can't be decoded
   *   - The native call errors
   */
  export async function encodeCardImage(uri: string): Promise<Float32Array | null> {
    try {
      const vec = await encodeImage(uri);
      if (vec && vec.length === 256) {
        readinessCached = true;
        return vec;
      }
      return null;
    } catch (err) {
      console.warn('[imageEncoder] native call failed:', err);
      return null;
    }
  }
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add app/modules/card-detector/src/index.ts app/src/embeddings/imageEncoder.ts
  git commit -m "feat(embeddings): TS imageEncoder wrapper that gracefully reports readiness"
  ```

---

## Task 7: `imageSearch.ts` — NN Lookup

**Files:**
- Create: `app/src/embeddings/imageSearch.ts`
- Create: `app/__tests__/embeddings/imageSearch.test.ts`

- [ ] **Step 1: Write failing tests**

  ```ts
  // app/__tests__/embeddings/imageSearch.test.ts
  jest.mock('../../src/embeddings/parser', () => ({
    getImageEmbeddingMap: jest.fn(),
  }));

  jest.mock('../../src/embeddings/imageEncoder', () => ({
    encodeCardImage: jest.fn(),
  }));

  import { findCardByImage } from '../../src/embeddings/imageSearch';
  import { getImageEmbeddingMap } from '../../src/embeddings/parser';
  import { encodeCardImage } from '../../src/embeddings/imageEncoder';

  function vec(values: number[]): Float32Array {
    const a = new Float32Array(values.length);
    values.forEach((v, i) => a[i] = v);
    // normalize
    let s = 0; a.forEach(x => s += x * x);
    const n = Math.sqrt(s);
    if (n > 0) for (let i = 0; i < a.length; i++) a[i] /= n;
    return a;
  }

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns null when encoder is not ready', async () => {
    (encodeCardImage as jest.Mock).mockResolvedValue(null);
    const result = await findCardByImage('file:///fake.jpg');
    expect(result).toBeNull();
  });

  it('returns top-1 match with highest cosine similarity', async () => {
    (encodeCardImage as jest.Mock).mockResolvedValue(vec([1, 0, 0, 0]));
    (getImageEmbeddingMap as jest.Mock).mockResolvedValue({
      version: 2, dim: 4, modelHash: 0,
      byId: new Map<string, Float32Array>([
        ['card-a', vec([0.95, 0.05, 0, 0])],   // very close
        ['card-b', vec([0.5, 0.5, 0, 0])],
        ['card-c', vec([0, 1, 0, 0])],
      ]),
      byName: new Map(),
    });
    const result = await findCardByImage('file:///fake.jpg');
    expect(result?.scryfallId).toBe('card-a');
    expect(result?.topK[0].scryfallId).toBe('card-a');
    expect(result?.topK).toHaveLength(3);
    expect(result?.score).toBeGreaterThan(0.99);
  });
  ```

- [ ] **Step 2: Run tests — verify failure**

  ```bash
  cd app && npx jest __tests__/embeddings/imageSearch.test.ts
  ```
  Expected: fail — `../../src/embeddings/imageSearch` does not exist.

- [ ] **Step 3: Create `app/src/embeddings/imageSearch.ts`**

  ```ts
  import { encodeCardImage } from './imageEncoder';
  import { getImageEmbeddingMap } from './parser';

  export type ImageMatch = {
    scryfallId: string;
    score: number;                           // top-1 cosine similarity
    topK: Array<{ scryfallId: string; score: number }>;
  };

  const TOP_K = 3;

  /**
   * Encode the rectified card image and find the nearest neighbor in the
   * loaded image-embeddings database. Returns null when:
   *   - The encoder asset is not bundled
   *   - The embeddings file is not loaded / doesn't exist
   *   - Any step errors
   */
  export async function findCardByImage(uri: string): Promise<ImageMatch | null> {
    const query = await encodeCardImage(uri);
    if (!query) return null;

    let index;
    try {
      index = await getImageEmbeddingMap();
    } catch {
      return null;
    }
    if (index.version !== 2) return null;
    if (index.byId.size === 0) return null;

    // Linear scan. 35k × 256 dot products ≈ 9M MACs — ~30 ms on phone.
    type Hit = { id: string; score: number };
    const topK: Hit[] = [];
    for (const [id, vec] of index.byId) {
      let score = 0;
      for (let i = 0; i < query.length; i++) score += query[i] * vec[i];

      if (topK.length < TOP_K) {
        topK.push({ id, score });
        topK.sort((a, b) => b.score - a.score);
      } else if (score > topK[TOP_K - 1].score) {
        topK[TOP_K - 1] = { id, score };
        topK.sort((a, b) => b.score - a.score);
      }
    }

    if (topK.length === 0) return null;
    return {
      scryfallId: topK[0].id,
      score:      topK[0].score,
      topK:       topK.map(h => ({ scryfallId: h.id, score: h.score })),
    };
  }
  ```

- [ ] **Step 4: Run tests — all pass**

  ```bash
  cd app && npx jest __tests__/embeddings/imageSearch.test.ts
  ```
  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add app/src/embeddings/imageSearch.ts app/__tests__/embeddings/imageSearch.test.ts
  git commit -m "feat(embeddings): findCardByImage top-K cosine-similarity search"
  ```

---

## Task 8: Downloader — Image-Artifact Manifest Entries

**Files:**
- Modify: `app/src/embeddings/downloader.ts`

- [ ] **Step 1: Extend the manifest type and add a parallel download path**

  Open `app/src/embeddings/downloader.ts`. Replace the `Manifest` type and `checkAndDownload` logic with a version that handles both text and image artifacts:

  ```ts
  type ManifestEntry  = { url: string; version: string; bytes: number; sha256: string };
  type ManifestV2 = {
    text_embeddings?:        ManifestEntry;
    image_embeddings?:       ManifestEntry;
    image_encoder_ios?:      ManifestEntry;
    image_encoder_android?:  ManifestEntry;
    // Legacy single-object form for pre-v2 CDN manifests
    version?: string;
    url?:     string;
  };
  ```

  Add below the existing text-embedding download logic, a parallel `checkAndDownloadImage` function that writes to `embeddings_image.bin` and `embeddings_image_version.txt`, mirroring the current flow but using `getImageEmbeddingsFile()` / `getImageVersionFile()`.

  Skeleton (fill the same try/catch structure as the existing function):

  ```ts
  export async function checkAndDownloadImage(
    setStatus: (status: EmbeddingStatus) => void
  ): Promise<void> {
    try {
      const r = await fetch(MANIFEST_URL);
      if (!r.ok) { setStatus('error'); return; }
      const m: ManifestV2 = await r.json();
      const entry = m.image_embeddings;
      if (!entry) { setStatus('idle'); return; }

      const file = getImageEmbeddingsFile();
      const versionFile = getImageVersionFile();
      const local = await readLocalVersion(versionFile);
      if (file.exists && local === entry.version) { setStatus('idle'); return; }

      setStatus('downloading');
      const dl = await File.downloadFileAsync(entry.url, file);
      if (dl.status !== 200) { setStatus('error'); return; }
      await versionFile.write(entry.version);
      clearEmbeddingCache();
      setStatus('idle');
    } catch {
      setStatus('error');
    }
  }

  async function readLocalVersion(f: any): Promise<string | null> {
    try { return (await f.read()).trim() || null; } catch { return null; }
  }
  ```

  (The exact existing file may already import `File` from `expo-file-system`; reuse those imports. Make sure `getImageEmbeddingsFile` and `getImageVersionFile` imports are added.)

- [ ] **Step 2: Commit**

  ```bash
  git add app/src/embeddings/downloader.ts
  git commit -m "feat(embeddings): download image embeddings in parallel with text embeddings"
  ```

---

## Task 9: Scan Pipeline — Image-First with OCR Fallback

**Files:**
- Modify: `app/src/scanner/ocr.ts`
- Modify: `app/app/(tabs)/scan.tsx`

- [ ] **Step 1: Add an image-first entry in the scanner**

  In `app/src/scanner/ocr.ts`, add this new exported function near the bottom:

  ```ts
  import { findCardByImage, ImageMatch } from '../embeddings/imageSearch';
  import { resolveCardById } from '../api/cards';

  const MATCH_ACCEPT    = 0.75;  // auto-commit threshold
  const MATCH_AMBIGUOUS = 0.55;  // show top-3 chooser threshold

  export type ImageScanResult = {
    strategy: 'image';
    match:    ImageMatch;        // top-1 + topK
    card:     CachedCard;
  };

  /** Try image embedding first; returns null when not ready or match too weak. */
  export async function scanCardByImage(uri: string): Promise<ImageScanResult | null> {
    const match = await findCardByImage(uri);
    if (!match) return null;
    if (match.score < MATCH_AMBIGUOUS) return null;
    if (match.score < MATCH_ACCEPT) {
      // Caller should show top-3; we still return so UI can decide.
    }
    const card = await resolveCardById(match.scryfallId);
    return { strategy: 'image', match, card };
  }
  ```

- [ ] **Step 2: Wire into `scan.tsx`**

  In `app/app/(tabs)/scan.tsx`, locate the `triggerOcr` function and modify the `scanCard` call to try the image path first:

  ```tsx
  import { scanCardByImage, ImageScanResult } from '../../src/scanner/ocr';
  ```

  Replace the portion of `triggerOcr` that currently does `await scanCard(uri, ...)` with:

  ```tsx
  // Try image-embedding identification first.
  const imageResult = await scanCardByImage(uri);
  if (imageResult && imageResult.match.score >= 0.75) {
    upsertCard(imageResult.card);
    addRecentScan(imageResult.card);
    setLastScannedId(imageResult.card.scryfall_id);
    setSuccessCard(imageResult.card.name);
    setPhase({ status: 'idle' });
    await new Promise<void>(r => setTimeout(r, 1500));
    setSuccessCard(null);
    return;
  }
  // Fallback — OCR path (existing).
  const result = await scanCard(uri, /* onProgress callback, imageSize */);
  ```

  Leave the existing onProgress / imageSize arguments intact as today.

- [ ] **Step 3: Clear session cache on unmount**

  At the top of `scan.tsx` near the other imports:

  ```tsx
  import { clearSessionCardCache } from '../../src/api/cards';
  ```

  Add near the end of `useEffect`s in `ScanScreen`:

  ```tsx
  useEffect(() => {
    return () => { clearSessionCardCache(); };
  }, []);
  ```

- [ ] **Step 4: Run Jest**

  ```bash
  cd app && npx jest
  ```
  Expected: all existing + new tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add app/src/scanner/ocr.ts app/app/\(tabs\)/scan.tsx
  git commit -m "feat(scan): image-embedding-first identification; OCR fallback; session cache lifecycle"
  ```

---

## Task 10: Verification on Device

**Files:** none modified — this is a validation step.

- [ ] **Step 1: Build and run on device**

  ```bash
  cd app && npx expo run:ios --device
  ```

- [ ] **Step 2: Confirm dormant behavior**

  With no encoder file bundled yet, verify:
  - App builds and launches normally.
  - Scan flow still works (OCR path fires as today).
  - Dev-console shows: `[CardDetector] card_encoder.mlmodelc not bundled — encodeImage disabled` on the first scan attempt.
  - No crashes.

- [ ] **Step 3: Commit verification note**

  ```bash
  echo "Verified $(date): image-embedding scaffolding dormant, OCR fallback intact." \
    >> docs/superpowers/plans/2026-04-18-image-embedding-card-identification.md
  git add docs/superpowers/plans/2026-04-18-image-embedding-card-identification.md
  git commit -m "docs: verification of dormant image-embedding scaffolding"
  ```

---

## Self-Review Notes

- **Spec coverage:** Each spec section maps to at least one task —
  parser v2 → Task 1; artifact contract → Tasks 4–5; hooks → Tasks 6–7;
  local-first lookup → Tasks 2–3; scan pipeline → Task 9;
  dormancy-on-missing-assets → Tasks 4, 5, 6 (all fail gracefully);
  downloader → Task 8; on-device verification → Task 10.

- **Type consistency:** `EmbeddingIndex` shape (`version`, `dim`, `modelHash`,
  `byId`, `byName`) matches between parser (Task 1), imageSearch (Task 7),
  and imageEncoder (Task 6). `ImageMatch` defined in imageSearch.ts (Task 7)
  and re-exported via ocr.ts (Task 9). `resolveCardById` signature stays
  `(scryfallId: string) => Promise<CachedCard>` across Tasks 2, 3, 9.

- **Dormant guarantees:** Every new native/TS module returns null when
  its underlying asset is missing. No change of behavior ships until
  `mtg-card-encoder` artifacts exist and the manifest references them.
