// Mocks must be declared before imports
jest.mock('../../modules/card-detector/src', () => ({
  detectCardCorners: jest.fn(),
}));

jest.mock('expo-image-manipulator', () => ({
  manipulateAsync: jest.fn(),
}));

jest.mock('react-native-text-recognition', () => ({
  __esModule: true,
  default: { recognize: jest.fn() },
}));

jest.mock('expo-file-system', () => ({
  File: jest.fn().mockImplementation((p: string) => ({ uri: `file://${p}`, copy: jest.fn() })),
  Paths: { cache: '/cache' },
}));

jest.mock('../../src/api/scryfall', () => ({
  fetchCardBySetNumber: jest.fn(),
  fetchCardByName: jest.fn(),
}));

jest.mock('../../src/api/cards', () => ({
  resolveCardById: jest.fn(async (id: string) => ({
    scryfall_id: id, name: 'Lightning Bolt', set_code: 'lea',
    collector_number: '161', mana_cost: '{R}', type_line: 'Instant',
    oracle_text: '', color_identity: '[]', image_uri: '',
    prices: '{}', keywords: '[]', cached_at: 0,
  })),
}));

import { parseSetAndNumber, scanCard } from '../../src/scanner/ocr';
import { detectCardCorners } from '../../modules/card-detector/src';
import * as ImageManipulator from 'expo-image-manipulator';
import { fetchCardBySetNumber, fetchCardByName } from '../../src/api/scryfall';
import { resolveCardById } from '../../src/api/cards';
import TextRecognition from 'react-native-text-recognition';

const mockDetect = detectCardCorners as jest.Mock;
const mockManipulate = ImageManipulator.manipulateAsync as jest.Mock;
const mockRecognize = (TextRecognition as any).recognize as jest.Mock;
const mockFetchBySet = fetchCardBySetNumber as jest.Mock;
const mockFetchByName = fetchCardByName as jest.Mock;
const mockResolveById = resolveCardById as jest.Mock;

const CORNERS: import('../../modules/card-detector/src').CardCorners = {
  topLeft:     { x: 0.1, y: 0.1 },
  topRight:    { x: 0.9, y: 0.1 },
  bottomRight: { x: 0.9, y: 0.9 },
  bottomLeft:  { x: 0.1, y: 0.9 },
  confidence:  0.85,
  rectifiedUri: 'file:///tmp/test.rect.jpg',
};

const MOCK_CARD = {
  scryfall_id: 'abc-123',
  name: 'Lightning Bolt',
  set_code: 'lea',
  collector_number: '161',
  mana_cost: '{R}',
  type_line: 'Instant',
  oracle_text: 'Deal 3 damage.',
  color_identity: '["R"]',
  image_uri: 'https://example.com/bolt.jpg',
  prices: '{"usd":"1.00"}',
  keywords: '[]',
  cached_at: Date.now(),
};

const stubManipulate = (width = 1000, height = 1400) => {
  mockManipulate.mockResolvedValue({ uri: 'file:///crop.jpg', width, height });
};

beforeEach(() => {
  jest.clearAllMocks();
  stubManipulate();
});

// ── parseSetAndNumber ────────────────────────────────────────────────────────

describe('parseSetAndNumber', () => {
  it('parses set-code-first format', () => {
    expect(parseSetAndNumber('lea 161/302')).toEqual({ setCode: 'lea', collectorNumber: '161' });
  });
  it('parses set code without slash', () => {
    expect(parseSetAndNumber('m21 420')).toEqual({ setCode: 'm21', collectorNumber: '420' });
  });
  it('parses 3-char set codes', () => {
    expect(parseSetAndNumber('cmr 085/361')).toEqual({ setCode: 'cmr', collectorNumber: '085' });
  });
  it('handles mixed case', () => {
    expect(parseSetAndNumber('LEA 161')).toEqual({ setCode: 'lea', collectorNumber: '161' });
  });
  it('parses modern card format: number rarity set lang', () => {
    expect(parseSetAndNumber('042/350 R IKO EN')).toEqual({ setCode: 'iko', collectorNumber: '042' });
  });
  it('parses modern format without total', () => {
    expect(parseSetAndNumber('161 R LEA EN')).toEqual({ setCode: 'lea', collectorNumber: '161' });
  });
  it('ignores language and rarity tokens', () => {
    expect(parseSetAndNumber('085/361 C CMR EN')).toEqual({ setCode: 'cmr', collectorNumber: '085' });
  });
  it('handles OCR noise around the real data', () => {
    expect(parseSetAndNumber('Illustrated by John Avon\n085/361 C CMR EN')).toEqual({
      setCode: 'cmr', collectorNumber: '085',
    });
  });
  it('returns null for unrecognizable text', () => {
    expect(parseSetAndNumber('not a card')).toBeNull();
  });
  it('returns null when no collector number present', () => {
    expect(parseSetAndNumber('IKO EN R')).toBeNull();
  });
});

// ── scanCard ─────────────────────────────────────────────────────────────────

describe('scanCard', () => {
  it('throws when no card detected', async () => {
    mockDetect.mockResolvedValue(null);
    await expect(scanCard('file:///photo.jpg')).rejects.toThrow('No card detected in image');
  });

  it('returns set_number result when bottom-left OCR succeeds', async () => {
    mockDetect.mockResolvedValue(CORNERS);
    mockRecognize.mockResolvedValue(['161 R LEA EN']);
    mockFetchBySet.mockResolvedValue(MOCK_CARD);

    const result = await scanCard('file:///photo.jpg');

    expect(result.strategy).toBe('set_number');
    expect(result.card.name).toBe('Lightning Bolt');
    expect(mockFetchBySet).toHaveBeenCalledWith('lea', '161');
    expect(mockResolveById).toHaveBeenCalledWith(MOCK_CARD.scryfall_id);
    expect(mockFetchByName).not.toHaveBeenCalled();
  });

  it('falls through to name strategy when set/number parse fails', async () => {
    mockDetect.mockResolvedValue(CORNERS);
    mockRecognize
      .mockResolvedValueOnce(['not a card'])
      .mockResolvedValueOnce(['Lightning Bolt']);
    mockFetchByName.mockResolvedValue(MOCK_CARD);

    const result = await scanCard('file:///photo.jpg');

    expect(result.strategy).toBe('name');
    expect(result.card.name).toBe('Lightning Bolt');
    expect(mockFetchByName).toHaveBeenCalledWith('Lightning Bolt');
    expect(mockResolveById).toHaveBeenCalledWith(MOCK_CARD.scryfall_id);
  });

  it('falls through to name strategy when Scryfall 404 on set/number', async () => {
    mockDetect.mockResolvedValue(CORNERS);
    mockRecognize
      .mockResolvedValueOnce(['161 R LEA EN'])
      .mockResolvedValueOnce(['Lightning Bolt']);
    mockFetchBySet.mockRejectedValue(new Error('Scryfall 404'));
    mockFetchByName.mockResolvedValue(MOCK_CARD);

    const result = await scanCard('file:///photo.jpg');

    expect(result.strategy).toBe('name');
    expect(mockFetchByName).toHaveBeenCalledWith('Lightning Bolt');
    expect(mockResolveById).toHaveBeenCalledWith(MOCK_CARD.scryfall_id);
  });

  it('throws when name region OCR returns no text', async () => {
    mockDetect.mockResolvedValue(CORNERS);
    mockRecognize.mockResolvedValue([]);

    await expect(scanCard('file:///photo.jpg')).rejects.toThrow('No text found in name region');
  });
});

describe('scanCard — rectified image path', () => {
  const RECT_CORNERS: import('../../modules/card-detector/src').CardCorners = {
    ...CORNERS,
    rectifiedUri: 'file:///tmp/test.rect.jpg',
  };

  it('crops from rectifiedUri when available (BL path)', async () => {
    mockDetect.mockResolvedValue(RECT_CORNERS);
    mockManipulate.mockResolvedValue({ uri: 'file:///crop.jpg', width: 400, height: 560 });
    mockRecognize.mockResolvedValue(['161 R LEA EN']);
    mockFetchBySet.mockResolvedValue(MOCK_CARD);

    await scanCard('file:///photo.jpg');

    expect(mockManipulate.mock.calls[0][0]).toBe('file:///tmp/test.rect.jpg');
  });

  it('crops from rectifiedUri for name fallback', async () => {
    mockDetect.mockResolvedValue(RECT_CORNERS);
    mockManipulate.mockResolvedValue({ uri: 'file:///crop.jpg', width: 400, height: 560 });
    mockRecognize
      .mockResolvedValueOnce(['not a card'])
      .mockResolvedValueOnce(['Lightning Bolt']);
    mockFetchByName.mockResolvedValue(MOCK_CARD);

    await scanCard('file:///photo.jpg');

    expect(mockManipulate.mock.calls[0][0]).toBe('file:///tmp/test.rect.jpg');
    expect(mockManipulate.mock.calls[1][0]).toBe('file:///tmp/test.rect.jpg');
  });

  it('falls back to raw URI crop when no rectifiedUri', async () => {
    const noRectCorners = { ...CORNERS, rectifiedUri: undefined };
    mockDetect.mockResolvedValue(noRectCorners);
    mockManipulate
      .mockResolvedValueOnce({ uri: 'file:///info.jpg', width: 1000, height: 1400 })
      .mockResolvedValueOnce({ uri: 'file:///crop.jpg', width: 1000, height: 1400 });
    mockRecognize.mockResolvedValue(['161 R LEA EN']);
    mockFetchBySet.mockResolvedValue(MOCK_CARD);

    await scanCard('file:///photo.jpg');

    expect(mockManipulate.mock.calls[1][0]).toBe('file:///photo.jpg');
  });
});
