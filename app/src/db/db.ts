import * as SQLite from 'expo-sqlite';

let _db: SQLite.SQLiteDatabase | null = null;

export function getDb(): SQLite.SQLiteDatabase {
  if (!_db) {
    _db = SQLite.openDatabaseSync('mtg-binder.db');
    initSchema(_db);
  }
  return _db;
}

function addColumnIfMissing(db: SQLite.SQLiteDatabase, table: string, column: string, columnDef: string): void {
  const cols = db.getAllSync<{ name: string }>(`PRAGMA table_info(${table});`);
  if (cols.some((c) => c.name === column)) return;
  db.execSync(`ALTER TABLE ${table} ADD COLUMN ${column} ${columnDef};`);
}

function initSchema(db: SQLite.SQLiteDatabase): void {
  db.execSync('PRAGMA journal_mode = WAL;');
  db.execSync('PRAGMA foreign_keys = ON;');
  db.execSync('PRAGMA synchronous = NORMAL;'); // safe for app-local cache DB
  db.execSync('PRAGMA temp_store = MEMORY;');
  db.execSync('PRAGMA mmap_size = 134217728;'); // 128 MB mmap
  addColumnIfMissing(db, 'cards', 'image_uri_back', "TEXT DEFAULT ''");
  addColumnIfMissing(db, 'cards', 'card_faces', "TEXT DEFAULT '[]'");
  addColumnIfMissing(db, 'cards', 'all_parts', "TEXT DEFAULT '[]'");
  addColumnIfMissing(db, 'cards', 'layout', "TEXT DEFAULT 'normal'");
  db.withTransactionSync(() => {
    db.execSync(`
    -- NOTE: do NOT add WITHOUT ROWID — cards_fts joins on cards.rowid
    CREATE TABLE IF NOT EXISTS cards (
      scryfall_id     TEXT PRIMARY KEY,
      name            TEXT NOT NULL,
      set_code        TEXT NOT NULL,
      collector_number TEXT NOT NULL,
      mana_cost       TEXT DEFAULT '',
      type_line       TEXT DEFAULT '',
      oracle_text     TEXT DEFAULT '',
      color_identity  TEXT DEFAULT '[]',
      image_uri       TEXT DEFAULT '',
      image_uri_back  TEXT DEFAULT '',
      card_faces      TEXT DEFAULT '[]',
      all_parts       TEXT DEFAULT '[]',
      prices          TEXT DEFAULT '{}',
      keywords        TEXT DEFAULT '[]',
      layout          TEXT DEFAULT 'normal',
      cached_at       INTEGER NOT NULL
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts
      USING fts5(scryfall_id UNINDEXED, name, oracle_text, keywords, content=cards, content_rowid=rowid);

    CREATE TRIGGER IF NOT EXISTS cards_ai AFTER INSERT ON cards BEGIN
      INSERT INTO cards_fts(rowid, scryfall_id, name, oracle_text, keywords)
        VALUES (new.rowid, new.scryfall_id, new.name, new.oracle_text, new.keywords);
    END;

    CREATE TRIGGER IF NOT EXISTS cards_au AFTER UPDATE ON cards BEGIN
      INSERT INTO cards_fts(cards_fts, rowid, scryfall_id, name, oracle_text, keywords)
        VALUES ('delete', old.rowid, old.scryfall_id, old.name, old.oracle_text, old.keywords);
      INSERT INTO cards_fts(rowid, scryfall_id, name, oracle_text, keywords)
        VALUES (new.rowid, new.scryfall_id, new.name, new.oracle_text, new.keywords);
    END;

    CREATE TRIGGER IF NOT EXISTS cards_ad AFTER DELETE ON cards BEGIN
      INSERT INTO cards_fts(cards_fts, rowid, scryfall_id, name, oracle_text, keywords)
        VALUES ('delete', old.rowid, old.scryfall_id, old.name, old.oracle_text, old.keywords);
    END;

    CREATE TABLE IF NOT EXISTS collection_entries (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      scryfall_id TEXT NOT NULL REFERENCES cards(scryfall_id),
      quantity    INTEGER NOT NULL DEFAULT 1,
      foil        INTEGER NOT NULL DEFAULT 0,
      condition   TEXT NOT NULL DEFAULT 'NM',
      added_at    INTEGER NOT NULL
    );

    -- Dedupe pre-existing (scryfall_id, foil, condition) dupes by summing qty into the oldest row.
    UPDATE collection_entries AS ce
       SET quantity = (
         SELECT SUM(quantity) FROM collection_entries
          WHERE scryfall_id = ce.scryfall_id AND foil = ce.foil AND condition = ce.condition
       )
     WHERE id = (
       SELECT MIN(id) FROM collection_entries
        WHERE scryfall_id = ce.scryfall_id AND foil = ce.foil AND condition = ce.condition
     );
    DELETE FROM collection_entries
     WHERE id NOT IN (
       SELECT MIN(id) FROM collection_entries
        GROUP BY scryfall_id, foil, condition
     );

    CREATE UNIQUE INDEX IF NOT EXISTS collection_entries_unique
      ON collection_entries (scryfall_id, foil, condition);

    CREATE TABLE IF NOT EXISTS decks (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      name          TEXT NOT NULL,
      format        TEXT NOT NULL DEFAULT '',
      created_at    INTEGER NOT NULL,
      art_crop_uri  TEXT NOT NULL DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS deck_cards (
      deck_id     INTEGER NOT NULL REFERENCES decks(id) ON DELETE CASCADE,
      scryfall_id TEXT NOT NULL REFERENCES cards(scryfall_id),
      quantity    INTEGER NOT NULL DEFAULT 1,
      board       TEXT NOT NULL DEFAULT 'main',
      PRIMARY KEY (deck_id, scryfall_id, board)
    );

    -- Covers getCardBySetNumber (hot path on import + scan confirmation).
    CREATE INDEX IF NOT EXISTS cards_set_number_idx ON cards (set_code, collector_number);

    -- Covers ORDER BY c.name on the binder grid without a sort step.
    CREATE INDEX IF NOT EXISTS cards_name_idx ON cards (name);

    -- Speeds up FK-driven joins from deck_cards -> cards on scryfall_id.
    CREATE INDEX IF NOT EXISTS deck_cards_scryfall_idx ON deck_cards (scryfall_id);
  `);
  });
  // Migrate existing decks tables (added after initial release).
  addColumnIfMissing(db, 'decks', 'art_crop_uri', "TEXT NOT NULL DEFAULT ''");
}
