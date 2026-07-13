"""
src/render_reports.py — Renderiza los informes Markdown a HTML autocontenido.

Convierte cada `outputs/report/*.md` a un `.html` con estilo sobrio y apto para
imprimir, con las figuras **embebidas** (base64) para que cada fichero sea un
único documento portable (se abre con doble clic en cualquier navegador; para PDF:
abrir en el navegador y "Imprimir → Guardar como PDF").

Usa `mistune` (ya presente como dependencia de nbconvert). Solo lectura sobre los
datos; escribe únicamente los `.html` en `outputs/report/`.

Uso (Windows/CMD, con el venv activado):
    python src\\render_reports.py
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import mistune

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "report"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", system-ui, -apple-system, Arial, sans-serif;
  color: #1a1a1a; background: #f4f4f2; line-height: 1.62;
  margin: 0; padding: 2.2rem 1rem;
}
main {
  max-width: 900px; margin: 0 auto; background: #ffffff;
  padding: 2.6rem 3rem; border: 1px solid #e2e1db; border-radius: 10px;
  box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
h1, h2, h3 { line-height: 1.25; color: #0b0b0b; }
h1 { font-size: 1.85rem; margin: 0 0 1.2rem; padding-bottom: .5rem; border-bottom: 3px solid #2a78d6; }
h2 { font-size: 1.35rem; margin: 2.2rem 0 .8rem; padding-bottom: .3rem; border-bottom: 1px solid #e2e1db; }
h3 { font-size: 1.1rem; margin: 1.6rem 0 .5rem; color: #2a3340; }
p, li { font-size: .98rem; }
a { color: #1f5fac; text-decoration: none; }
a:hover { text-decoration: underline; }
code {
  font-family: "Cascadia Code", Consolas, monospace; font-size: .86em;
  background: #f0f1f3; padding: .1em .38em; border-radius: 4px; color: #b02a5b;
}
table {
  border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem;
  display: block; overflow-x: auto;
}
th, td { border: 1px solid #dcdbd5; padding: .5rem .7rem; text-align: left; vertical-align: top; }
th { background: #eef2f7; font-weight: 600; color: #17324f; }
tbody tr:nth-child(even) { background: #fafafa; }
blockquote {
  margin: 1rem 0; padding: .6rem 1.1rem; border-left: 4px solid #2a78d6;
  background: #f5f8fd; color: #33445a; border-radius: 0 6px 6px 0;
}
blockquote p { margin: .3rem 0; }
img { max-width: 100%; height: auto; display: block; margin: 1.2rem auto;
      border: 1px solid #e2e1db; border-radius: 6px; }
hr { border: none; border-top: 1px solid #e2e1db; margin: 2rem 0; }
ul, ol { padding-left: 1.4rem; }
li { margin: .25rem 0; }
footer { max-width: 900px; margin: 1rem auto 0; color: #8a8a84; font-size: .8rem; text-align: center; }
@media print {
  body { background: #fff; padding: 0; }
  main { border: none; box-shadow: none; max-width: none; padding: 0; }
  h2 { page-break-after: avoid; }
  table, img, blockquote { page-break-inside: avoid; }
}
"""

_render = mistune.create_markdown(plugins=["table", "strikethrough", "url"])


def _embed_images(html: str) -> str:
    """Reemplaza <img src="../figures/x.png"> por un data URI base64 (portable)."""
    def repl(m: re.Match) -> str:
        src = m.group(1)
        p = FIG_DIR / Path(src).name
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            return f'src="data:image/png;base64,{b64}"'
        return m.group(0)
    return re.sub(r'src="([^"]*\.png)"', repl, html)


def _rewrite_report_links(html: str) -> str:
    """Enlaces a otros informes .md -> su versión .html (navegación entre HTML)."""
    return re.sub(r'href="(?!\.\./)([^"/:]+)\.md"',
                  lambda m: f'href="{m.group(1)}.html"', html)


def render_one(md_path: Path) -> Path:
    body = _render(md_path.read_text(encoding="utf-8"))
    body = _embed_images(body)
    body = _rewrite_report_links(body)
    title = md_path.stem.replace("_", " ")
    doc = (f"<!doctype html>\n<html lang=\"es\">\n<head>\n<meta charset=\"utf-8\">\n"
           f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
           f"<title>{title}</title>\n<style>{CSS}</style>\n</head>\n<body>\n"
           f"<main>\n{body}\n</main>\n"
           f"<footer>MedXpertQA · TFG · generado desde {md_path.name} por src/render_reports.py</footer>\n"
           f"</body>\n</html>\n")
    out = md_path.with_suffix(".html")
    out.write_text(doc, encoding="utf-8")
    return out


def main() -> None:
    print("Renderizando informes Markdown -> HTML autocontenido ...")
    md_files = sorted(REPORT_DIR.glob("*.md"))
    for md in md_files:
        out = render_one(md)
        print(f"  {md.name:28s} -> {out.name}  ({out.stat().st_size/1024:.0f} KB)")
    print(f"Hecho. {len(md_files)} documentos en {REPORT_DIR.relative_to(PROJECT_ROOT).as_posix()}/")
    print("Ábrelos con doble clic (navegador). Para PDF: navegador -> Imprimir -> Guardar como PDF.")


if __name__ == "__main__":
    main()
