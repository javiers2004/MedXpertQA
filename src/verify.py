"""
src/verify.py — Tarea B del Paso 1 (TFG MedXpertQA).

Verificación de integridad del dataset descargado. Es de SOLO LECTURA: no
modifica ni corrige nada. Documenta todas las comprobaciones en
outputs/report/00_integridad.md y marca las anomalías para decisión humana.

Comprobaciones:
  1. Recuentos de registros (MM/test=2000, Text/test=2450; dev informativo).
  2. Referencias imagen<->archivo (solo MM): faltantes en disco, huérfanas por
     exceso, y multiplicidad de referencias.
  3. Integridad de imágenes con Pillow (Image.open().verify()).
  4. Unicidad de los `id` dentro de cada subconjunto.
  5. Campos obligatorios + coherencia (label dentro de options, nº de opciones,
     vocabulario de metadatos dev<->test).

Uso (Windows/CMD, con el venv activado):
    python src\\verify.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

# --- Rutas y configuración -----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "images"
REPORT_PATH = PROJECT_ROOT / "outputs" / "report" / "00_integridad.md"
PROVENANCE_PATH = RAW_DIR / "PROVENANCE.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"}

FILES = {
    "MM": {"test": RAW_DIR / "MM" / "test.jsonl", "dev": RAW_DIR / "MM" / "dev.jsonl"},
    "Text": {"test": RAW_DIR / "Text" / "test.jsonl", "dev": RAW_DIR / "Text" / "dev.jsonl"},
}
EXPECTED_TEST_COUNTS = {"MM": 2000, "Text": 2450}
EXPECTED_OPTIONS = {"MM": list("ABCDE"), "Text": list("ABCDEFGHIJ")}
REQUIRED_FIELDS = ["id", "question", "options", "label",
                   "body_system", "medical_task", "question_type"]
METADATA_FIELDS = ["body_system", "medical_task", "question_type"]


# --- Utilidades ----------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def list_disk_images() -> list[Path]:
    return sorted(f for f in IMAGES_DIR.rglob("*")
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


# --- Comprobaciones ------------------------------------------------------------
def check_counts(data: dict) -> dict:
    """Recuentos por fichero y desviación respecto a lo esperado."""
    res = {}
    for modality, splits in FILES.items():
        for split, path in splits.items():
            n = len(data[(modality, split)])
            expected = EXPECTED_TEST_COUNTS[modality] if split == "test" else None
            ok = (expected is None) or (n == expected)
            res[(modality, split)] = {"n": n, "expected": expected, "ok": ok}
    return res


def check_image_refs(data: dict) -> dict:
    """Referencias imagen<->archivo para MM (test y dev)."""
    disk = list_disk_images()
    disk_names = {f.name for f in disk}

    def refs_of(split: str):
        pairs = []  # (qid, basename)
        for r in data[("MM", split)]:
            for ref in (r.get("images") or []):
                pairs.append((r["id"], Path(ref).name))
        return pairs

    test_pairs = refs_of("test")
    dev_pairs = refs_of("dev")
    test_refs = [b for _, b in test_pairs]
    dev_refs = [b for _, b in dev_pairs]
    referenced_union = set(test_refs) | set(dev_refs)

    # Multiplicidad: basename -> conjunto de qids que lo referencian
    multiplicity = defaultdict(set)
    for qid, b in test_pairs + dev_pairs:
        multiplicity[b].add(qid)
    shared = {b: sorted(q) for b, q in multiplicity.items() if len(q) > 1}

    return {
        "n_disk": len(disk),
        "n_refs_test": len(test_refs),
        "n_refs_dev": len(dev_refs),
        "n_unique_test": len(set(test_refs)),
        "n_unique_union": len(referenced_union),
        "missing_test": sorted(set(test_refs) - disk_names),
        "missing_dev": sorted(set(dev_refs) - disk_names),
        "orphan_excess": sorted(disk_names - referenced_union),
        "shared_refs": shared,
    }


def check_image_integrity() -> dict:
    """Abre cada imagen con Pillow y llama a verify() (integridad estructural)."""
    disk = list_disk_images()
    corrupt = []
    for i, f in enumerate(disk, 1):
        try:
            with Image.open(f) as im:
                im.verify()
        except Exception as exc:  # noqa: BLE001
            corrupt.append((f.name, repr(exc)))
        if i % 500 == 0:
            print(f"      [integridad] {i}/{len(disk)} imágenes verificadas ...")
    return {"n_checked": len(disk), "corrupt": corrupt}


def check_id_uniqueness(data: dict) -> dict:
    """Unicidad de `id` por fichero y solapamiento test<->dev por modalidad."""
    res = {}
    for modality, splits in FILES.items():
        per_split_ids = {}
        for split in splits:
            ids = [r.get("id") for r in data[(modality, split)]]
            dup = [i for i, c in Counter(ids).items() if c > 1]
            per_split_ids[split] = set(ids)
            res[(modality, split)] = {"n": len(ids), "n_unique": len(set(ids)), "dups": dup}
        overlap = per_split_ids.get("test", set()) & per_split_ids.get("dev", set())
        res[(modality, "overlap_test_dev")] = sorted(overlap)
    return res


def check_required_and_coherence(data: dict) -> dict:
    """Campos obligatorios + coherencia (label en options, nº de opciones)."""
    res = {}
    for modality, splits in FILES.items():
        exp_opts = EXPECTED_OPTIONS[modality]
        for split in splits:
            missing_fields = []       # (id, [campos ausentes/vacios])
            label_not_in_opts = []    # (id, label, keys)
            wrong_n_options = []      # (id, keys)
            missing_images = []       # (id) solo MM
            for r in data[(modality, split)]:
                qid = r.get("id", "<sin id>")
                miss = [f for f in REQUIRED_FIELDS
                        if f not in r or r[f] in (None, "", [], {})]
                if miss:
                    missing_fields.append((qid, miss))
                opts = r.get("options")
                label = r.get("label")
                if isinstance(opts, dict):
                    keys = sorted(opts.keys())
                    if keys != exp_opts:
                        wrong_n_options.append((qid, keys))
                    if label not in opts:
                        label_not_in_opts.append((qid, label, keys))
                if modality == "MM":
                    if not r.get("images"):
                        missing_images.append(qid)
            res[(modality, split)] = {
                "missing_fields": missing_fields,
                "label_not_in_opts": label_not_in_opts,
                "wrong_n_options": wrong_n_options,
                "missing_images": missing_images,
            }
    return res


def check_metadata_vocab(data: dict) -> dict:
    """Vocabulario de metadatos y valores presentes en dev pero no en test."""
    res = {}
    for modality, splits in FILES.items():
        for field in METADATA_FIELDS:
            test_vals = {r.get(field) for r in data[(modality, "test")]}
            dev_vals = {r.get(field) for r in data[(modality, "dev")]}
            res[(modality, field)] = {
                "test": sorted(v for v in test_vals if v is not None),
                "dev_only": sorted(v for v in (dev_vals - test_vals) if v is not None),
            }
    return res


# --- Renderizado del informe ---------------------------------------------------
def render_report(data, counts, refs, integrity, ids, req, vocab) -> tuple[str, list[str]]:
    anomalies: list[str] = []
    L: list[str] = []

    def add(line=""):
        L.append(line)

    # Cabecera / procedencia
    prov = {}
    if PROVENANCE_PATH.exists():
        prov = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    add("# Informe de Integridad — MedXpertQA (Paso 1 · Tarea B)")
    add()
    add(f"- **Generado**: {datetime.now(timezone.utc).isoformat()} (UTC)")
    add(f"- **Repositorio**: `{prov.get('repo_id', 'TsinghuaC3I/MedXpertQA')}` "
        f"(revisión `{prov.get('pinned_revision', 'N/D')}`)")
    add(f"- **Script**: `src/verify.py` (solo lectura; no corrige anomalías)")
    add()
    add("> Este informe verifica la fiabilidad del dato antes del análisis. Las "
        "anomalías se **documentan y marcan**, no se corrigen automáticamente.")
    add()

    # Placeholder del veredicto (se rellena al final, insertándolo aquí)
    verdict_idx = len(L)
    add()
    add()

    # 1. Recuentos
    add("## 1. Recuentos de registros")
    add()
    add("| Fichero | Registros | Esperado | Estado |")
    add("|---|---:|---:|:--:|")
    for modality, splits in FILES.items():
        for split in splits:
            c = counts[(modality, split)]
            exp = c["expected"] if c["expected"] is not None else "—"
            status = "OK" if c["ok"] else "ANOMALÍA"
            if not c["ok"]:
                anomalies.append(
                    f"Recuento de {modality}/{split}: {c['n']} != esperado {c['expected']}.")
            add(f"| `{modality}/{split}.jsonl` | {c['n']} | {exp} | {status} |")
    add()
    add("Los `dev.jsonl` (few-shot) se reportan como informativos: no hay recuento "
        "canónico esperado más allá de que sean no vacíos.")
    add()

    # 2. Referencias imagen<->archivo
    add("## 2. Referencias imagen ↔ archivos (solo MM)")
    add()
    add(f"- Imágenes en disco (`data/processed/images/`): **{refs['n_disk']}**")
    add(f"- Referencias desde MM/test: **{refs['n_refs_test']}** "
        f"({refs['n_unique_test']} únicas)")
    add(f"- Referencias desde MM/dev: **{refs['n_refs_dev']}**")
    add(f"- Referencias únicas (test ∪ dev): **{refs['n_unique_union']}**")
    add(f"- Faltantes en disco referenciadas por **test**: **{len(refs['missing_test'])}**")
    add(f"- Faltantes en disco referenciadas por **dev**: **{len(refs['missing_dev'])}**")
    add(f"- Huérfanas por exceso (en disco, sin referencia en test ∪ dev): "
        f"**{len(refs['orphan_excess'])}**")
    add(f"- Imágenes referenciadas por más de una pregunta: **{len(refs['shared_refs'])}**")
    add()
    if refs["missing_test"]:
        anomalies.append(f"{len(refs['missing_test'])} imágenes referenciadas por MM/test "
                         f"no existen en disco.")
        add("**Faltantes (test) — muestra:** " +
            ", ".join(f"`{x}`" for x in refs["missing_test"][:20]))
        add()
    if refs["orphan_excess"]:
        anomalies.append(f"{len(refs['orphan_excess'])} imágenes en disco sin ninguna "
                         f"referencia (huérfanas por exceso).")
        add("**Huérfanas por exceso — muestra:** " +
            ", ".join(f"`{x}`" for x in refs["orphan_excess"][:20]))
        add()
    if not refs["missing_test"] and not refs["missing_dev"] and not refs["orphan_excess"]:
        add(f"Correspondencia **exacta**: cada una de las {refs['n_refs_test']} referencias de "
            f"test y las {refs['n_refs_dev']} de dev tiene su fichero, y los "
            f"{refs['n_disk']} ficheros están todos referenciados "
            f"({refs['n_unique_test']} test + {refs['n_refs_dev']} dev = {refs['n_disk']}).")
        add()

    # 3. Integridad Pillow
    add("## 3. Integridad de las imágenes (Pillow `verify()`)")
    add()
    add(f"- Imágenes verificadas: **{integrity['n_checked']}**")
    add(f"- Corruptas / no decodificables: **{len(integrity['corrupt'])}**")
    add()
    if integrity["corrupt"]:
        anomalies.append(f"{len(integrity['corrupt'])} imágenes corruptas.")
        add("| Fichero | Error |")
        add("|---|---|")
        for name, err in integrity["corrupt"][:50]:
            add(f"| `{name}` | {err} |")
        add()
    else:
        add("Todas las imágenes superan la verificación estructural de Pillow.")
        add()

    # 4. Unicidad de IDs
    add("## 4. Unicidad de identificadores (`id`)")
    add()
    add("| Subconjunto | Registros | IDs únicos | Duplicados |")
    add("|---|---:|---:|---:|")
    for modality, splits in FILES.items():
        for split in splits:
            u = ids[(modality, split)]
            if u["dups"]:
                anomalies.append(f"IDs duplicados en {modality}/{split}: {u['dups'][:10]}")
            add(f"| `{modality}/{split}` | {u['n']} | {u['n_unique']} | {len(u['dups'])} |")
    add()
    for modality in FILES:
        ov = ids[(modality, "overlap_test_dev")]
        if ov:
            anomalies.append(f"Solapamiento de IDs test/dev en {modality}: {ov[:10]}")
        add(f"- Solapamiento de IDs test↔dev en {modality}: **{len(ov)}** "
            f"{'(' + ', '.join(ov[:5]) + ')' if ov else '(ninguno)'}")
    add()

    # 5. Campos obligatorios y coherencia
    add("## 5. Campos obligatorios y coherencia")
    add()
    add(f"Campos obligatorios exigidos: {', '.join('`'+f+'`' for f in REQUIRED_FIELDS)} "
        f"(y `images` no vacío en MM).")
    add()
    add("| Subconjunto | Sin campos oblig. | label∉options | nº opciones incorrecto | MM sin imágenes |")
    add("|---|---:|---:|---:|---:|")
    for modality, splits in FILES.items():
        for split in splits:
            r = req[(modality, split)]
            for key, label_txt in [("missing_fields", "campos obligatorios"),
                                    ("label_not_in_opts", "label fuera de options"),
                                    ("wrong_n_options", "nº de opciones inesperado")]:
                if r[key]:
                    anomalies.append(
                        f"{modality}/{split}: {len(r[key])} registros con {label_txt}.")
            mm_img = len(r["missing_images"]) if modality == "MM" else "—"
            add(f"| `{modality}/{split}` | {len(r['missing_fields'])} | "
                f"{len(r['label_not_in_opts'])} | {len(r['wrong_n_options'])} | {mm_img} |")
    add()
    # Detalle de incumplimientos (si los hay)
    for modality, splits in FILES.items():
        for split in splits:
            r = req[(modality, split)]
            for key, title in [("missing_fields", "Registros con campos obligatorios ausentes/vacíos"),
                               ("label_not_in_opts", "Registros con `label` fuera de `options`"),
                               ("wrong_n_options", "Registros con nº de opciones inesperado")]:
                if r[key]:
                    add(f"**{modality}/{split} — {title}:**")
                    for item in r[key][:20]:
                        add(f"- `{item}`")
                    add()

    # 5b. Vocabulario de metadatos dev<->test
    add("### 5b. Coherencia de vocabulario de metadatos (dev ↔ test)")
    add()
    add("| Modalidad · campo | Valores en test | Valores solo en dev |")
    add("|---|---|---|")
    for modality in FILES:
        for field in METADATA_FIELDS:
            v = vocab[(modality, field)]
            dev_only = v["dev_only"]
            if dev_only:
                anomalies.append(
                    f"{modality}: `{field}` tiene valores en dev ausentes en test: {dev_only}.")
            add(f"| `{modality}` · `{field}` | {', '.join(v['test'])} | "
                f"{', '.join(dev_only) if dev_only else '—'} |")
    add()

    # Sección de anomalías
    add("## Anomalías detectadas (marcadas para decisión, NO corregidas)")
    add()
    if anomalies:
        for a in anomalies:
            add(f"- ⚠ {a}")
    else:
        add("**Ninguna.** Todas las comprobaciones se superan con los valores indicados arriba.")
    add()

    # Rellenar el veredicto
    if anomalies:
        verdict = (f"> **VEREDICTO: {len(anomalies)} anomalía(s) detectada(s)** "
                   f"— ver sección final. El dato es utilizable pero requiere tu decisión "
                   f"sobre los puntos marcados.")
    else:
        verdict = ("> **VEREDICTO: INTEGRIDAD CORRECTA (0 anomalías)** — recuentos, "
                   "correspondencia imagen↔archivo, integridad Pillow, unicidad de IDs y "
                   "campos obligatorios verificados sin incidencias.")
    L[verdict_idx] = verdict

    return "\n".join(L) + "\n", anomalies


# --- Main ----------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("TAREA B — Verificacion de integridad de MedXpertQA")
    print("=" * 70)

    # Carga
    data = {}
    for modality, splits in FILES.items():
        for split, path in splits.items():
            data[(modality, split)] = load_jsonl(path)
            print(f"  Cargado {modality}/{split}: {len(data[(modality, split)])} registros")

    print("[1/5] Recuentos ...")
    counts = check_counts(data)
    print("[2/5] Referencias imagen<->archivo ...")
    refs = check_image_refs(data)
    print("[3/5] Integridad de imagenes (Pillow verify) ...")
    integrity = check_image_integrity()
    print("[4/5] Unicidad de IDs ...")
    ids = check_id_uniqueness(data)
    print("[5/5] Campos obligatorios y coherencia ...")
    req = check_required_and_coherence(data)
    vocab = check_metadata_vocab(data)

    report, anomalies = render_report(data, counts, refs, integrity, ids, req, vocab)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("-" * 70)
    print("RESUMEN")
    print(f"  MM/test={counts[('MM','test')]['n']}  Text/test={counts[('Text','test')]['n']}")
    print(f"  Imagenes en disco={refs['n_disk']}  refs_test={refs['n_refs_test']}  "
          f"faltantes_test={len(refs['missing_test'])}  huerfanas_exceso={len(refs['orphan_excess'])}")
    print(f"  Imagenes corruptas={len(integrity['corrupt'])}")
    print(f"  ANOMALIAS={len(anomalies)}")
    print(f"  Informe: {REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
