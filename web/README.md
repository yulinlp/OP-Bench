# OPBench web demo

This directory contains a small full-stack research explorer for OP-Bench.
The frontend is dependency-free HTML/CSS/JavaScript. The backend uses Python's
standard library to serve the UI and expose read-only JSON endpoints.

## Start locally

From the repository root:

```bash
python web/backend/server.py --host 127.0.0.1 --port 8765
```

Open <http://127.0.0.1:8765> in a browser.

The header language button switches the explorer between English and Chinese;
the selected language is remembered in the browser.

## API routes

- `GET /api/health`
- `GET /api/summary`
- `GET /api/examples?category=irrelevance`
- `GET /api/examples?category=repetition`
- `GET /api/examples?category=sycophancy`
- `GET /api/results`

The QA examples are curated in `web/data/qa_cases.json` from Appendix Figures
21–32 (PDF pages 33–46). Each case includes the retrieved memory, a
rubric-aligned high-score reference, and an abbreviated appendix response with
the reported score. The research result payload is in
`web/data/research_results.json`, with values transcribed from the paper's main
tables, figures, and Appendix Table 11.

The high-score references are explanatory contrasts rather than new model
runs. Chain-of-thought from the appendix is intentionally not displayed.

The paper explicitly labels three research questions. The fourth tab in the
demo is a clearly marked timing view derived from the appendix latency table.

## Build a static GitHub Pages site

GitHub Pages does not run the Python server. The frontend first tries the
local API and automatically falls back to the JSON payloads in
<code>web/data/</code> when the API is unavailable.

Build a Pages-ready directory locally:

~~~bash
python scripts/build_pages.py --output .pages
python -m http.server 8765 --directory .pages
~~~

Open <http://127.0.0.1:8765> to verify the static version. The
<code>.github/workflows/deploy-pages.yml</code> workflow performs the same
build on relevant pushes to <code>main</code> and publishes the generated site.
After enabling GitHub Pages with **GitHub Actions** as the source, the project
site will be available at
<https://yulinlp.github.io/OP-Bench/>.
