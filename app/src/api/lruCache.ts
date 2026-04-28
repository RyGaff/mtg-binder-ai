/** Bounded LRU map with optional time-to-idle eviction. Insertion-order Map:
 *  re-inserts on hit move keys to the most-recently-used end; oldest key
 *  evicts when size exceeds capacity. When `ttlMs` is set, entries also
 *  evict lazily on read if `ttlMs` has elapsed since last access — bounds
 *  memory by both count and age without needing a periodic sweep. */
type Entry<V> = { value: V; lastAccess: number };

export class LruCache<K, V> {
  private readonly map = new Map<K, Entry<V>>();
  private readonly ttlMs: number;

  constructor(private readonly capacity: number, ttlMs: number = Number.POSITIVE_INFINITY) {
    if (capacity <= 0) throw new Error('LruCache capacity must be > 0');
    if (ttlMs <= 0) throw new Error('LruCache ttlMs must be > 0');
    this.ttlMs = ttlMs;
  }

  private isExpired(entry: Entry<V>): boolean {
    return Date.now() - entry.lastAccess > this.ttlMs;
  }

  /** Read + promote to MRU. Lazily evicts and returns undefined if expired. */
  get(key: K): V | undefined {
    const entry = this.map.get(key);
    if (!entry) return undefined;
    if (this.isExpired(entry)) {
      this.map.delete(key);
      return undefined;
    }
    entry.lastAccess = Date.now();
    this.map.delete(key);
    this.map.set(key, entry);
    return entry.value;
  }

  /** Read without promoting. Still evicts expired entries lazily. */
  peek(key: K): V | undefined {
    const entry = this.map.get(key);
    if (!entry) return undefined;
    if (this.isExpired(entry)) {
      this.map.delete(key);
      return undefined;
    }
    return entry.value;
  }

  set(key: K, value: V): void {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, { value, lastAccess: Date.now() });
    if (this.map.size > this.capacity) {
      const oldest = this.map.keys().next().value as K | undefined;
      if (oldest !== undefined) this.map.delete(oldest);
    }
  }

  delete(key: K): boolean {
    return this.map.delete(key);
  }

  clear(): void {
    this.map.clear();
  }

  get size(): number {
    return this.map.size;
  }
}
