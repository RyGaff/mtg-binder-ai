import { serializeToJson, serializeToCsv, parseImportFile } from '../../src/export/collection';
import type { CollectionEntryWithCard } from '../../src/db/collection';

const mockEntries: CollectionEntryWithCard[] = [
  {
    id: 1,
    scryfall_id: 'abc',
    quantity: 2,
    foil: false,
    condition: 'NM',
    added_at: 1000,
    name: 'Lightning Bolt',
    set_code: 'lea',
    collector_number: '161',
    mana_cost: '{R}',
    type_line: 'Instant',
    oracle_text: '',
    color_identity: '[]',
    image_uri: '',
    image_uri_back: '',
    card_faces: '[]',
    all_parts: '[]',
    prices: '{"usd":"1.20"}',
    keywords: '[]',
    cached_at: 1000,
  },
];

describe('collection export', () => {
  it('serializeToJson produces valid JSON with all entries', () => {
    const json = serializeToJson(mockEntries);
    const parsed = JSON.parse(json);
    expect(parsed).toHaveLength(1);
    expect(parsed[0].name).toBe('Lightning Bolt');
  });

  it('serializeToCsv has header row and data row', () => {
    const csv = serializeToCsv(mockEntries);
    const lines = csv.split('\n');
    expect(lines[0]).toContain('name');
    expect(lines[1]).toContain('Lightning Bolt');
  });

  it('parseImportFile parses JSON', () => {
    const json = serializeToJson(mockEntries);
    const result = parseImportFile(json, 'json');
    expect(result).toHaveLength(1);
    expect(result[0].scryfall_id).toBe('abc');
  });

  it('parseImportFile parses CSV and preserves scryfall_id', () => {
    const csv = serializeToCsv(mockEntries);
    const result = parseImportFile(csv, 'csv');
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe('Lightning Bolt');
    expect(result[0].scryfall_id).toBe('abc');
  });

  it('parseImportFile parses Moxfield CSV format', () => {
    const moxfield = [
      '"Count","Tradelist Count","Name","Edition","Condition","Language","Foil","Tags","Last Modified","Collector Number","Alter","Proxy","Purchase Price"',
      '"2","2","Lightning Bolt","lea","Near Mint","English","","","2024-01-01 00:00:00.000000","161","False","False",""',
      '"1","1","Black Lotus","lea","Near Mint","English","foil","","2024-01-01 00:00:00.000000","232","False","False",""',
    ].join('\n');
    const result = parseImportFile(moxfield, 'csv');
    expect(result).toHaveLength(2);
    expect(result[0].name).toBe('Lightning Bolt');
    expect(result[0].quantity).toBe(2);
    expect(result[0].set_code).toBe('lea');
    expect(result[0].collector_number).toBe('161');
    expect(result[0].foil).toBe(false);
    expect(result[0].condition).toBe('NM');
    expect(result[1].foil).toBe(true);
  });

  it('parseImportFile handles card names with commas', () => {
    const moxfield = [
      '"Count","Tradelist Count","Name","Edition","Condition","Language","Foil","Tags","Last Modified","Collector Number","Alter","Proxy","Purchase Price"',
      '"1","1","Birgi, God of Storytelling // Harnfel, Horn of Bounty","khm","Near Mint","English","","","2024-01-01 00:00:00.000000","123","False","False",""',
    ].join('\n');
    const result = parseImportFile(moxfield, 'csv');
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe('Birgi, God of Storytelling // Harnfel, Horn of Bounty');
    expect(result[0].set_code).toBe('khm');
  });
});
