/** Bounded LRU map. Insertion-order Map: re-inserts on hit move keys to the
 *  most-recently-used end; oldest key evicts when size exceeds capacity. */
export class LruCache<K, V> {
  private readonly map = new Map<K, V>();

  constructor(private readonly capacity: number) {
    if (capacity <= 0) throw new Error('LruCache capacity must be > 0');
  }

  /** Read + promote to MRU. */
  get(key: K): V | undefined {
    if (!this.map.has(key)) return undefined;
    const value = this.map.get(key) as V;
    this.map.delete(key);
    this.map.set(key, value);
    return value;
  }

  /** Read without touching LRU order. Use when the caller may discard the
   *  entry (e.g. stale-check followed by delete) — avoids a wasted promote. */
  peek(key: K): V | undefined {
    return this.map.get(key);
  }

  /** Membership check without promotion. */
  has(key: K): boolean {
    return this.map.has(key);
  }

  set(key: K, value: V): void {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, value);
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
