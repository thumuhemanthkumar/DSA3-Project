# Lyrics Similarity & Attribution Engine

Python engine (rolling-hash fingerprints, inverted index, top-K heap, LCS) plus a web UI
that talks to it through a small standard-library server.

    engine/        core package
    data/          catalogue.json, query_song.txt
    web/index.html the UI (calls the Python engine via server.py)
    server.py      serves the UI + /api endpoints
    main.py        command-line report
    benchmark.py   indexed vs naive timing
    tests/         unit tests

No third-party packages needed (Python 3.8+).

    python3 server.py          # open http://127.0.0.1:8000
    python3 main.py --query data/query_song.txt
    python3 benchmark.py --songs 400
    python3 -m unittest discover tests

On Windows use `python` or `py` instead of `python3`.
