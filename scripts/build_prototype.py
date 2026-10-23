"""Build the self-contained web prototype: prototype/template.html + prototype/replay.json -> prototype/index.html"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tpl = (ROOT / "prototype" / "template.html").read_text(encoding="utf-8")
data = (ROOT / "prototype" / "replay.json").read_text(encoding="utf-8").replace("</", "<\\/")
out = ROOT / "prototype" / "index.html"
out.write_text(tpl.replace("__REPLAY_JSON__", data), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
