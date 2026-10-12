"""Hand-built data structures used by the attribution engine.

Nothing here imports a third-party library on purpose: the point of the
project is to show the structures, not to call someone else's.

    HashMap      separate-chaining hash table with load-factor resizing
    MinHeap      binary heap used for top-K selection
    RollingHash  Rabin-Karp polynomial hashing over token sequences
"""

# ----------------------------------------------------------------------
# String hashing (FNV-1a, 64 bit)
# ----------------------------------------------------------------------

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF


def fnv1a(text):
    """Deterministic 64-bit hash of a string.

    Python's built-in hash() is randomised per process, which would make
    fingerprints non-reproducible between runs. FNV-1a is stable.
    """
    h = _FNV_OFFSET
    for ch in text:
        h ^= ord(ch) & 0xFF
        h = (h * _FNV_PRIME) & _MASK64
    return h


# ----------------------------------------------------------------------
# Hash map
# ----------------------------------------------------------------------

class HashMap:
    """Hash table with separate chaining.

    Average case: get/put/contains O(1).  Worst case O(n) if every key
    collides.  Resizes (doubling) once load factor passes 0.75, which keeps
    the amortised cost constant.
    """

    def __init__(self, capacity=64):
        self._capacity = capacity
        self._size = 0
        self._buckets = [[] for _ in range(capacity)]

    # -- internal ------------------------------------------------------
    def _slot(self, key, capacity=None):
        capacity = capacity or self._capacity
        if isinstance(key, int):
            h = fnv1a(str(key))
        else:
            h = fnv1a(key)
        return h % capacity

    def _resize(self):
        new_capacity = self._capacity * 2
        new_buckets = [[] for _ in range(new_capacity)]
        for bucket in self._buckets:
            for key, value in bucket:
                new_buckets[self._slot(key, new_capacity)].append((key, value))
        self._buckets = new_buckets
        self._capacity = new_capacity

    # -- public --------------------------------------------------------
    def put(self, key, value):
        bucket = self._buckets[self._slot(key)]
        for i, (existing, _) in enumerate(bucket):
            if existing == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._size += 1
        if self._size / self._capacity > 0.75:
            self._resize()

    def get(self, key, default=None):
        for existing, value in self._buckets[self._slot(key)]:
            if existing == key:
                return value
        return default

    def setdefault(self, key, default):
        found = self.get(key, _MISSING)
        if found is _MISSING:
            self.put(key, default)
            return default
        return found

    def contains(self, key):
        return self.get(key, _MISSING) is not _MISSING

    def keys(self):
        for bucket in self._buckets:
            for key, _ in bucket:
                yield key

    def items(self):
        for bucket in self._buckets:
            for pair in bucket:
                yield pair

    def __len__(self):
        return self._size

    def __contains__(self, key):
        return self.contains(key)


class _Missing:
    pass


_MISSING = _Missing()


# ----------------------------------------------------------------------
# Min-heap (top-K selection)
# ----------------------------------------------------------------------

class MinHeap:
    """Binary min-heap over (priority, payload) pairs.

    push / pop are O(log n), peek is O(1).  Used to keep the K best
    candidates while scanning all of them: keeping a heap of size K costs
    O(n log K) instead of the O(n log n) a full sort would cost.
    """

    def __init__(self):
        self._items = []

    def __len__(self):
        return len(self._items)

    def peek(self):
        return self._items[0] if self._items else None

    def push(self, priority, payload):
        self._items.append((priority, payload))
        self._sift_up(len(self._items) - 1)

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty heap")
        top = self._items[0]
        last = self._items.pop()
        if self._items:
            self._items[0] = last
            self._sift_down(0)
        return top

    def push_bounded(self, priority, payload, k):
        """Keep only the K largest priorities seen so far."""
        if len(self._items) < k:
            self.push(priority, payload)
        elif priority > self._items[0][0]:
            self.pop()
            self.push(priority, payload)

    def drain_descending(self):
        out = []
        while self._items:
            out.append(self.pop())
        out.reverse()
        return out

    # -- internal ------------------------------------------------------
    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self._items[i][0] < self._items[parent][0]:
                self._items[i], self._items[parent] = self._items[parent], self._items[i]
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self._items)
        while True:
            left, right, smallest = 2 * i + 1, 2 * i + 2, i
            if left < n and self._items[left][0] < self._items[smallest][0]:
                smallest = left
            if right < n and self._items[right][0] < self._items[smallest][0]:
                smallest = right
            if smallest == i:
                return
            self._items[i], self._items[smallest] = self._items[smallest], self._items[i]
            i = smallest


# ----------------------------------------------------------------------
# Rabin-Karp rolling hash
# ----------------------------------------------------------------------

MOD = (1 << 61) - 1      # Mersenne prime, collision-safe for our sizes
BASE = 1_000_003


class RollingHash:
    """Fingerprints every k-token window of a document in O(n) total.

    Computing each window from scratch would be O(n*k).  The rolling step
    drops the outgoing token and adds the incoming one in O(1):

        H(i+1) = (H(i) - v[i] * BASE^(k-1)) * BASE + v[i+k]
    """

    def __init__(self, k):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self._high_power = pow(BASE, k - 1, MOD)

    def fingerprints(self, tokens):
        """Return the list of window hashes; index i covers tokens[i:i+k]."""
        k = self.k
        if len(tokens) < k:
            return []

        values = [fnv1a(t) % MOD for t in tokens]

        h = 0
        for i in range(k):
            h = (h * BASE + values[i]) % MOD
        out = [h]

        for i in range(1, len(values) - k + 1):
            h = (h - values[i - 1] * self._high_power) % MOD
            h = (h * BASE + values[i + k - 1]) % MOD
            out.append(h)
        return out
