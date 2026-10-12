"""Similarity measures.

Three signals, because no single one is honest on its own:

  containment  what fraction of the QUERY is borrowed.  A 30-second
               interpolation inside a long song still scores high here.
  jaccard      symmetric overlap of the two shingle sets.  Catches
               wholesale copying, stays low for a short lifted couplet.
  lcs ratio    longest common subsequence over tokens.  Survives small
               edits ("walking home alone" vs "walking back home alone")
               that shingle hashing alone would miss.
"""


def jaccard(set_a, set_b):
    """|A ∩ B| / |A ∪ B|  —  O(min(|A|, |B|))."""
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a) + len(set_b) - inter
    return inter / union if union else 0.0


def containment(set_a, set_b):
    """|A ∩ B| / |A|  —  how much of A is present in B."""
    if not set_a:
        return 0.0
    return len(set_a & set_b) / len(set_a)


def lcs_length(a, b):
    """Longest common subsequence length.

    Classic DP: dp[i][j] = 1 + dp[i-1][j-1] on a match, else the better of
    dropping a token from either side.  Time O(n*m); memory reduced to two
    rows, so O(min(n, m)) space.
    """
    if not a or not b:
        return 0
    if len(a) < len(b):          # keep the inner loop over the shorter list
        a, b = b, a

    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                left, up = curr[j - 1], prev[j]
                curr[j] = left if left >= up else up
        prev = curr
    return prev[-1]


def lcs_ratio(a, b, cap=1_200):
    """LCS normalised by the shorter sequence.

    The DP is quadratic, so long documents are trimmed to `cap` tokens
    before comparing. That keeps a query against a 5000-song catalogue
    from turning into an afternoon.
    """
    a, b = a[:cap], b[:cap]
    if not a or not b:
        return 0.0
    return lcs_length(a, b) / min(len(a), len(b))


def align_runs(query_fps, song_positions, min_shingles=1):
    """Chain isolated shingle hits into maximal contiguous passages.

    A single shared 5-gram is noise. A run of shared 5-grams, each one step
    further along in BOTH songs, is a copied passage.

    Every match (i, j) sits on the diagonal d = i - j; a copied passage is a
    run of consecutive i values on one diagonal. Grouping by diagonal turns
    passage-finding into a linear pass instead of a pairwise search.

    Returns a list of dicts: query/source start positions and run length,
    measured in shingles.
    """
    diagonals = {}
    for i, fp in enumerate(query_fps):
        for j in song_positions.get(fp, []) or []:
            diagonals.setdefault(i - j, []).append((i, j))

    runs = []
    for d, hits in diagonals.items():
        hits.sort()
        start_i, start_j = hits[0]
        length = 1
        for idx in range(1, len(hits)):
            i, j = hits[idx]
            prev_i = hits[idx - 1][0]
            if i == prev_i + 1:
                length += 1
            else:
                if length >= min_shingles:
                    runs.append({"query_start": start_i, "source_start": start_j,
                                 "shingles": length})
                start_i, start_j = i, j
                length = 1
        if length >= min_shingles:
            runs.append({"query_start": start_i, "source_start": start_j,
                         "shingles": length})

    runs.sort(key=lambda r: (-r["shingles"], r["query_start"]))
    return runs
