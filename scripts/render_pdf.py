"""Render an HTML file to a PDF with Playwright's Chromium (headless), with page numbers in the footer.

    <a Python with Playwright> scripts/render_pdf.py docs/report.html docs/report.pdf

Playwright is not a dependency of the agent; any Python with it installed can run this.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

FOOTER = (
    '<div style="width:100%;font-size:8px;color:#666;font-family:Helvetica,Arial,sans-serif;padding:0 0.7in;'
    'display:flex;justify-content:space-between"><span>Klaxon Diagnosis Agent. Aravind Sundaravadivelu, CSYE 7374,'
    ' Fall 2026</span><span><span class="pageNumber"></span> of <span class="totalPages"></span></span></div>'
)


def main(source: str, target: str) -> int:
    with sync_playwright() as play:
        browser = play.chromium.launch(channel="chromium")
        page = browser.new_page()
        page.goto(Path(source).resolve().as_uri())
        page.pdf(
            path=target,
            format="Letter",
            print_background=True,
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=FOOTER,
            margin={"top": "0.6in", "bottom": "0.7in", "left": "0.7in", "right": "0.7in"},
        )
        browser.close()
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
