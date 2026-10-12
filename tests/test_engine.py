"""Unit tests.  Run from the project root:  python3 -m unittest discover tests"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import AttributionEngine                      # noqa: E402
from engine.normalize import tokenize, token_texts        # noqa: E402
from engine.similarity import align_runs, jaccard, lcs_length  # noqa: E402
from engine.structures import HashMap, MinHeap, RollingHash    # noqa: E402

CATALOGUE = [
    {"id": "A", "title": "Alpha", "artist": "One",
     "lyrics": "the river keeps a key beneath the stone\nand every door it opens leads back home\ni counted all the corners of the dark\nand found a smaller room inside my heart"},
    {"id": "B", "title": "Beta", "artist": "Two",
     "lyrics": "engines in the orchard eating light\nsomething mechanical about the night\nwe traded all our apples for a spark\nand drove until the highway lost its mark"},
]


class TestHashMap(unittest.TestCase):
    def test_put_get_overwrite(self):
        m = HashMap(capacity=4)
        m.put("a", 1)
        m.put("b", 2)
        m.put("a", 3)
        self.assertEqual(m.get("a"), 3)
        self.assertEqual(m.get("b"), 2)
        self.assertEqual(len(m), 2)
        self.assertIsNone(m.get("missing"))

    def test_resize_preserves_entries(self):
        m = HashMap(capacity=4)
        for i in range(200):
            m.put(f"key{i}", i)
        self.assertEqual(len(m), 200)
        self.assertEqual(m.get("key199"), 199)
        self.assertEqual(sorted(m.keys()), sorted(f"key{i}" for i in range(200)))


class TestMinHeap(unittest.TestCase):
    def test_bounded_keeps_largest(self):
        heap = MinHeap()
        for value in [5, 1, 9, 3, 7, 2]:
            heap.push_bounded(value, f"item{value}", k=3)
        result = [p for p, _ in heap.drain_descending()]
        self.assertEqual(result, [9, 7, 5])


class TestRollingHash(unittest.TestCase):
    def test_matches_naive_recomputation(self):
        tokens = "a rolling hash should agree with the slow version".split()
        rh = RollingHash(3)
        rolled = rh.fingerprints(tokens)
        # naive: hash each window independently by re-running the same recurrence
        naive = []
        for i in range(len(tokens) - 2):
            naive.append(RollingHash(3).fingerprints(tokens[i:i + 3])[0])
        self.assertEqual(rolled, naive)

    def test_short_document_yields_nothing(self):
        self.assertEqual(RollingHash(5).fingerprints(["a", "b"]), [])


class TestSimilarity(unittest.TestCase):
    def test_jaccard_bounds(self):
        self.assertEqual(jaccard({1, 2, 3}, {1, 2, 3}), 1.0)
        self.assertEqual(jaccard({1, 2}, {3, 4}), 0.0)
        self.assertAlmostEqual(jaccard({1, 2, 3}, {2, 3, 4}), 0.5)

    def test_lcs(self):
        self.assertEqual(lcs_length(list("ABCBDAB"), list("BDCABA")), 4)
        self.assertEqual(lcs_length([], list("abc")), 0)

    def test_align_runs_chains_consecutive_hits(self):
        query = [10, 11, 12, 99]
        positions = {10: [0], 11: [1], 12: [2], 99: []}
        runs = align_runs(query, positions, min_shingles=2)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["shingles"], 3)
        self.assertEqual(runs[0]["query_start"], 0)


class TestNormalize(unittest.TestCase):
    def test_punctuation_case_and_elongation(self):
        tokens = tokenize("Sooooo, LOUD -- tonight!")
        self.assertEqual(token_texts(tokens), ["soo", "loud", "tonight"])

    def test_fillers_dropped_and_lines_tracked(self):
        tokens = tokenize("oh yeah the light\nis gone")
        self.assertEqual(token_texts(tokens), ["the", "light", "is", "gone"])
        self.assertEqual(tokens[0].line_no, 1)
        self.assertEqual(tokens[-1].line_no, 2)


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AttributionEngine(k=4, top_k=3)
        self.engine.ingest_records(CATALOGUE)

    def test_identical_song_scores_near_one(self):
        report = self.engine.attribute(CATALOGUE[0]["lyrics"])
        self.assertEqual(report.best.song.id, "A")
        self.assertGreater(report.best.score, 0.95)
        self.assertEqual(report.best.verdict, "near-identical")

    def test_unrelated_text_finds_nothing(self):
        report = self.engine.attribute(
            "a treatise on municipal drainage policy and its quarterly review")
        self.assertEqual(report.matches, [])

    def test_lifted_passage_is_flagged_with_line_numbers(self):
        query = ("i wrote this part myself it owes nobody anything\n"
                 "and every door it opens leads back home\n"
                 "i counted all the corners of the dark\n"
                 "then went outside to watch it rain")
        report = self.engine.attribute(query)
        best = report.best
        self.assertEqual(best.song.id, "A")
        self.assertTrue(best.passages)
        top = best.passages[0]
        self.assertEqual(top.query_lines, (2, 3))
        self.assertEqual(top.source_lines, (2, 3))

    def test_edited_paraphrase_still_ranks_first(self):
        query = ("the river keeps a key beneath the stone\n"
                 "and every doorway opens leading back to home\n"
                 "i counted all the corners of the darkness here")
        self.assertEqual(self.engine.attribute(query).best.song.id, "A")

    def test_partial_borrow_ranks_source_above_the_rest(self):
        query = ("we traded all our apples for a spark\n"
                 "and drove until the highway lost its mark\n"
                 "now nothing here belongs to anyone at all")
        report = self.engine.attribute(query)
        self.assertEqual(report.best.song.id, "B")
        self.assertGreater(report.best.signals["containment"], 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
