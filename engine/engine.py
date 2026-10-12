"""The attribution engine itself.

Pipeline for one query:

  1. normalise + tokenise the query                        O(n)
  2. fingerprint every k-token window (Rabin-Karp)         O(n)
  3. probe the inverted index, tally hits per song         O(hits)
  4. keep the top-K candidates with a bounded min-heap     O(c log K)
  5. for each survivor: align diagonals into passages,     O(hits + n*m)
     run the LCS DP, blend the three signals into a score
  6. rank and report

Steps 1-4 are cheap and run against the whole catalogue; the expensive
quadratic work in step 5 only ever touches K songs. That split is the
whole design.
"""

import json

try:
    from .index import CatalogueIndex
    from .normalize import tokenize, token_texts, line_span
    from .similarity import align_runs, containment, jaccard, lcs_ratio
    from .structures import HashMap, MinHeap, RollingHash
except ImportError:
    from index import CatalogueIndex
    from normalize import tokenize, token_texts, line_span
    from similarity import align_runs, containment, jaccard, lcs_ratio
    from structures import HashMap, MinHeap, RollingHash

# Blend weights for the final score. Containment leads because lifting a
# chorus is the behaviour we actually care about catching.
W_CONTAINMENT = 0.45
W_JACCARD = 0.20
W_LCS = 0.35

VERDICTS = [
    (0.85, "near-identical"),
    (0.60, "strong match"),
    (0.35, "partial match"),
    (0.10, "weak overlap"),
    (0.00, "no meaningful overlap"),
]


class PassageMatch:
    def __init__(self, query_lines, source_lines, tokens, query_text, source_text):
        self.query_lines = query_lines
        self.source_lines = source_lines
        self.tokens = tokens
        self.query_text = query_text
        self.source_text = source_text

    def to_dict(self):
        return {
            "query_lines": list(self.query_lines),
            "source_lines": list(self.source_lines),
            "matched_tokens": self.tokens,
            "query_text": self.query_text,
            "source_text": self.source_text,
        }


class Attribution:
    def __init__(self, song, score, signals, passages):
        self.song = song
        self.score = score
        self.signals = signals
        self.passages = passages

    @property
    def verdict(self):
        for threshold, label in VERDICTS:
            if self.score >= threshold:
                return label
        return VERDICTS[-1][1]

    def to_dict(self):
        return {
            "song_id": self.song.id,
            "title": self.song.title,
            "artist": self.song.artist,
            "score": round(self.score, 4),
            "verdict": self.verdict,
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "passages": [p.to_dict() for p in self.passages],
        }


class AttributionReport:
    def __init__(self, query_title, query_stats, matches):
        self.query_title = query_title
        self.query_stats = query_stats
        self.matches = matches

    @property
    def best(self):
        return self.matches[0] if self.matches else None

    def to_dict(self):
        return {
            "query": self.query_title,
            "query_stats": self.query_stats,
            "matches": [m.to_dict() for m in self.matches],
        }


class AttributionEngine:
    """Ingests a catalogue, then attributes query songs against it."""

    def __init__(self, k=5, top_k=3, min_run_shingles=2, lcs_cap=1200):
        self.k = k
        self.top_k = top_k
        self.min_run_shingles = min_run_shingles
        self.lcs_cap = lcs_cap
        self.index = CatalogueIndex(k=k)
        self._hasher = RollingHash(k)

    # -- ingestion -----------------------------------------------------
    def ingest_records(self, records):
        self.index.add_many(records)
        return self.index.stats()

    def ingest_json(self, path):
        with open(path, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        return self.ingest_records(records)

    # -- querying ------------------------------------------------------
    def attribute(self, text, title="query"):
        tokens = tokenize(text)
        words = token_texts(tokens)
        query_fps = self._hasher.fingerprints(words)
        query_set = set(query_fps)

        stats = {
            "tokens": len(words),
            "shingles": len(query_fps),
            "distinct_shingles": len(query_set),
            "k": self.k,
        }

        if not query_fps:
            return AttributionReport(title, stats, [])

        # --- step 3: tally index hits per candidate song ---------------
        hits = HashMap()                      # song_id -> matched distinct shingles
        for fp in query_set:
            for song_id, _pos in self.index.postings(fp):
                hits.put(song_id, hits.get(song_id, 0) + 1)

        # --- step 4: bounded heap keeps only the best candidates -------
        heap = MinHeap()
        for song_id, matched in hits.items():
            heap.push_bounded(matched / len(query_set), song_id, self.top_k)
        candidates = [song_id for _, song_id in heap.drain_descending()]

        # --- step 5: expensive scoring, candidates only ----------------
        results = []
        for song_id in candidates:
            song = self.index.get(song_id)
            song_set = set(song.fingerprints)

            signals = {
                "containment": containment(query_set, song_set),
                "jaccard": jaccard(query_set, song_set),
                "lcs_ratio": lcs_ratio(words, song.words, cap=self.lcs_cap),
            }
            score = (W_CONTAINMENT * signals["containment"]
                     + W_JACCARD * signals["jaccard"]
                     + W_LCS * signals["lcs_ratio"])

            passages = self._flag_passages(tokens, query_fps, song)
            results.append(Attribution(song, score, signals, passages))

        results.sort(key=lambda r: -r.score)
        return AttributionReport(title, stats, results)

    def attribute_file(self, path):
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        return self.attribute(text, title=path.split("/")[-1])

    # -- passage flagging ----------------------------------------------
    def _flag_passages(self, query_tokens, query_fps, song, limit=5):
        runs = align_runs(query_fps, song.positions,
                          min_shingles=self.min_run_shingles)
        passages = []
        for run in runs[:limit]:
            span = run["shingles"] + self.k - 1        # shingles -> tokens
            q_start = run["query_start"]
            s_start = run["source_start"]

            q_lines = line_span(query_tokens, q_start, q_start + span - 1)
            s_lines = line_span(song.tokens, s_start, s_start + span - 1)

            passages.append(PassageMatch(
                query_lines=q_lines,
                source_lines=s_lines,
                tokens=span,
                query_text=_excerpt(query_tokens, q_start, span),
                source_text=_excerpt(song.tokens, s_start, span),
            ))
        return passages


def _excerpt(tokens, start, span, max_words=14):
    chunk = [t.text for t in tokens[start:start + span]]
    if len(chunk) > max_words:
        chunk = chunk[:max_words] + ["..."]
    return " ".join(chunk)
