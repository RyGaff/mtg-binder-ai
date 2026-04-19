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
  (db.isCardStale as jest.Mock).mockReturnValue(false);
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

it('clears the session cache', async () => {
  (db.getCardById as jest.Mock).mockReturnValue(CARD);
  await resolveCardById('aaa-111');
  expect(getSessionCacheSize()).toBe(1);
  clearSessionCardCache();
  expect(getSessionCacheSize()).toBe(0);
});
