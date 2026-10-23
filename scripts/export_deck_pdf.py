"""Render the presentation slides (1920x1080 HTML sections) to presentation/presentation.pdf."""
import asyncio, json, re, sys
from pathlib import Path
from playwright.async_api import async_playwright

DECK = Path(sys.argv[1])            # folder holding project/deck.json and project/slides/
ROOT = Path(__file__).resolve().parents[1]
BLOBS = {"f5710e0159b7883bb177ebb649cc69fd": "reports/figures/prototype_screenshot.png",
         "be8b7ad9188a46f5422bcd96b51e91d2": "docs/architecture.png",
         "b1264d52b447583bb9a96919ab3e18ea": "reports/figures/auprc.png",
         "79b9e4b939f5f83ea26ba6b1d435a98f": "reports/figures/noise_robustness.png",
         "51d9ee47b19380e00fe70021becdac76": "reports/figures/alert_tradeoff.png",
         "4b60d3b07da6166e5da699d89be95753": "reports/figures/mae_by_horizon.png"}
order = json.load(open(DECK / "project/deck.json"))["order"]
parts = []
for sid in order:
    h = (DECK / f"project/slides/{sid}.html").read_text()
    h = re.sub(r"<aside>.*?</aside>", "", h, flags=re.S)
    for k, v in BLOBS.items():
        h = h.replace(f"/_blob/{k}", (ROOT / v).as_uri())
    h = re.sub(r'<x-icon name="Warning"[^>]*></x-icon>', '<span style="font-size:48px;color:#B3261E">&#9888;</span>', h)
    parts.append(h)
css = """@page { size: 1920px 1080px; margin: 0 }
html, body { margin: 0; padding: 0 }
section { width: 1920px; height: 1080px; position: relative; box-sizing: border-box; overflow: hidden; page-break-after: always; break-after: page }
section * { box-sizing: border-box }
h1, h2, h3, p, ul { margin: 0 }
ul { padding-left: 1.2em }
table { border-collapse: collapse; width: 100% }
th, td { padding: 0.35em 0.6em; border-bottom: 1px solid #C9D3D0; text-align: left }
th { font-weight: 700 }
div > table { width: auto; flex: 1 }"""
html = f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(parts)}</body></html>"
tmp = ROOT / "presentation" / "_deck_render.html"
tmp.write_text(html, encoding="utf-8")

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1920, "height": 1080})
        await pg.goto(tmp.as_uri()); await pg.wait_for_timeout(800)
        await pg.pdf(path=str(ROOT / "presentation" / "presentation.pdf"), width="1920px", height="1080px", print_background=True)
        for i in (0, 4, 8, 12):
            await pg.evaluate(f"document.querySelectorAll('section')[{i}].scrollIntoView()")
            await pg.screenshot(path=f"/tmp/claude-0/slide{i}.png")
        await b.close()
asyncio.run(main())
tmp.unlink()
print("wrote presentation/presentation.pdf")
