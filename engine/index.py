"""The searchable catalogue: documents plus an inverted index.

A linear scan would compare the query against every song in full, costing
O(catalogue_size) work per query.  Instead every song is chopped into
overlapping k-token windows ("shingles"), each window is fingerprinted with
the rolling hash, and the fingerprint is used as an index key:

    fingerprint -> [(song_id, position), (song_id, position), ...]

Answering a query then only touches songs that actually share a window with
it, which is the same idea a search engine's posting lists use.
"""

try:
    from .normalize import tokenize, token_texts
    from .structures import HashMap, RollingHash
except ImportError:
    from normalize import tokenize, token_texts
    from structures import HashMap, RollingHash


class Song:
    """One catalogue entry, pre-tokenised and pre-fingerprinted."""

    def __init__(self, song_id, title, artist, text, k):
        self.id = song_id
        self.title = title
        self.artist = artist
        self.text = text
        self.lines = text.splitlines()
        self.tokens = tokenize(text)
        self.words = token_texts(self.tokens)
        self.fingerprints = RollingHash(k).fingerprints(self.words)

        # fingerprint -> positions inside this song (for passage alignment)
        self.positions = HashMap()
        for pos, fp in enumerate(self.fingerprints):
            self.positions.setdefault(fp, []).append(pos)

    @property
    def shingle_count(self):
        return len(self.fingerprints)

    @property
    def distinct_shingle_count(self):
        return len(self.positions)

    def label(self):
        return f"{self.title} — {self.artist}" if self.artist else self.title

    def __repr__(self):
        return f"<Song {self.id} {self.title!r} tokens={len(self.words)}>"


class CatalogueIndex:
    """Inverted index across every song in the catalogue."""

    def __init__(self, k=5):
        self.k = k
        self.songs = HashMap()          # song_id -> Song
        self._order = []                # insertion order, for reporting
        self._postings = HashMap()      # fingerprint -> [(song_id, pos)]

    # -- building ------------------------------------------------------
    def add_song(self, song_id, title, artist, text):
        if song_id in self.songs:
            raise ValueError(f"duplicate song id: {song_id}")
        song = Song(song_id, title, artist, text, self.k)
        self.songs.put(song_id, song)
        self._order.append(song_id)

        seen = set()
        for pos, fp in enumerate(song.fingerprints):
            self._postings.setdefault(fp, []).append((song_id, pos))
            seen.add(fp)
        return song

    def add_many(self, records):
        for rec in records:
            self.add_song(
                rec["id"],
                rec.get("title", rec["id"]),
                rec.get("artist", ""),
                rec["lyrics"],
            )

    # -- querying ------------------------------------------------------
    def postings(self, fingerprint):
        return self._postings.get(fingerprint, [])

    def get(self, song_id):
        return self.songs.get(song_id)

    def all_songs(self):
        return [self.songs.get(sid) for sid in self._order]

    def stats(self):
        songs = self.all_songs()
        return {
            "songs": len(songs),
            "k": self.k,
            "total_tokens": sum(len(s.words) for s in songs),
            "distinct_fingerprints": len(self._postings),
            "total_shingles": sum(s.shingle_count for s in songs),
        }

    def __len__(self):
        return len(self._order)
