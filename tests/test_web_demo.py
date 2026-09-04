import json
from pathlib import Path

from scripts.build_pages import build_site
from web.backend.server import EXAMPLES, QA_CASES, RESEARCH_DATA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_INDEX = Path(__file__).resolve().parents[1] / "web" / "frontend" / "index.html"
FRONTEND_APP = FRONTEND_INDEX.with_name("app.js")
FRONTEND_I18N = FRONTEND_INDEX.with_name("i18n.js")
QA_CASES_FILE = FRONTEND_INDEX.parents[1] / "data" / "qa_cases.json"


def test_demo_has_examples_for_each_benchmark_category():
    assert set(EXAMPLES) == {"irrelevance", "repetition", "sycophancy"}
    assert all(category["groups"] for category in EXAMPLES.values())
    assert len(EXAMPLES["irrelevance"]["groups"][0]["items"]) == 2
    assert len(EXAMPLES["repetition"]["groups"][0]["items"]) == 2
    assert len(EXAMPLES["sycophancy"]["groups"][2]["items"]) == 2


def test_demo_results_include_all_requested_views():
    assert set(RESEARCH_DATA) >= {"meta", "overview", "workflow", "rq1", "rq2", "rq3", "rq4"}
    assert len(RESEARCH_DATA["rq1"]["models"]) == 4
    assert len(RESEARCH_DATA["rq4"]["latency"]) == 5


def test_qa_cases_include_memory_answer_contrasts_and_reasons():
    assert "Appendix Figures 21–32" in QA_CASES["meta"]["source"]
    for category in EXAMPLES.values():
        for group in category["groups"]:
            for case in group["items"]:
                for prompt in case["prompts"]:
                    assert prompt["memory"]
                    assert prompt["high_answer"]["text"]
                    assert prompt["low_answer"]["text"]
                    assert prompt["high_reason"]
                    assert prompt["low_reason"]


def test_chapter_navigation_is_visible_and_declared_in_html():
    markup = FRONTEND_INDEX.read_text(encoding="utf-8")
    script = FRONTEND_APP.read_text(encoding="utf-8")

    assert '<nav class="main-nav"' not in markup
    assert '<link rel="stylesheet" href="./styles.css?v=readability-2" />' in markup
    assert '<script defer src="./app.js"></script>' in markup
    assert "fetch(`./api/${path}`)" in script
    assert "function getStaticJson(path)" in script
    assert "fetch(`./data/${dataFile}`)" in script
    assert "bindSectionNavigation();" in script
    for section in ("pipeline", "examples", "results"):
        assert f'data-section-tab="{section}"' in markup
        assert f'data-explorer-page="{section}"' in markup


def test_frontend_has_complete_bilingual_language_contract():
    markup = FRONTEND_INDEX.read_text(encoding="utf-8")
    script = FRONTEND_APP.read_text(encoding="utf-8")
    translations = FRONTEND_I18N.read_text(encoding="utf-8")
    qa_cases = json.loads(QA_CASES_FILE.read_text(encoding="utf-8"))

    assert 'id="language-toggle"' in markup
    assert '<script defer src="./i18n.js"></script>' in markup
    assert "function applyLanguage()" in script
    assert "function toggleLanguage()" in script
    assert "localStorage.setItem('opbench-language', state.language)" in script
    assert "document.documentElement.lang" in script
    assert "rq1Tab" in translations
    assert "rq4Tab" in translations

    item_ids = {
        item["id"]
        for category in qa_cases["categories"].values()
        for group in category["groups"]
        for item in group["items"]
    }
    assert item_ids
    assert all(f"'{item_id}':" in translations for item_id in item_ids)


def test_pages_builder_creates_a_rooted_static_site(tmp_path):
    output_dir = tmp_path / "pages"
    build_site(PROJECT_ROOT, output_dir)

    assert (output_dir / "index.html").is_file()
    assert (output_dir / "styles.css").is_file()
    assert (output_dir / "app.js").is_file()
    assert (output_dir / "i18n.js").is_file()
    assert (output_dir / "data" / "qa_cases.json").is_file()
    assert (output_dir / "data" / "research_results.json").is_file()
    assert (output_dir / ".nojekyll").is_file()
