"""
src/download.py — Tarea A del Paso 1 (TFG MedXpertQA).

Descarga reproducible del benchmark MedXpertQA desde Hugging Face, extracción de
las imágenes y registro de procedencia (provenance). El script es IDEMPOTENTE:
si los datos ya están descargados y las imágenes ya extraídas, no vuelve a
descargar ni a extraer; solo verifica y regenera el resumen de procedencia.

No modifica los ficheros originales descargados. No borra el images.zip.

Uso (Windows/CMD, con el entorno virtual activado):
    python src\\download.py
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# --- Configuración reproducible ------------------------------------------------
REPO_ID = "TsinghuaC3I/MedXpertQA"
REPO_TYPE = "dataset"
# Revisión (commit) FIJADA para trazabilidad del dato. Confirmada por inspección
# read-only de la HF API el 2025-07-10 (lastModified del repo: 2025-07-09).
REVISION = "7e7c465a68eb2b866926bfa59c8c9d17a8daba65"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGES_ZIP = RAW_DIR / "images.zip"
PROVENANCE_PATH = RAW_DIR / "PROVENANCE.json"

JSONL_RELPATHS = ["MM/dev.jsonl", "MM/test.jsonl", "Text/dev.jsonl", "Text/test.jsonl"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"}


# --- Utilidades ----------------------------------------------------------------
def human_size(num_bytes: int) -> str:
    """Formatea un tamaño en bytes de forma legible."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def dir_size_bytes(path: Path) -> int:
    """Suma recursiva del tamaño de todos los ficheros bajo `path`."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def count_jsonl_records(path: Path) -> int:
    """Cuenta las líneas no vacías (un registro por línea) de un JSONL."""
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 de un fichero (para anclar la procedencia del dato)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTS


def count_image_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for f in path.rglob("*") if f.is_file() and is_image_name(f.name))


# --- Descarga ------------------------------------------------------------------
def download_snapshot() -> Path:
    """Descarga el snapshot completo del dataset a data/raw/ (idempotente).

    snapshot_download compara etags: los ficheros ya presentes y sin cambios no
    se vuelven a descargar. Se fija la revisión para reproducibilidad.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    local_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=REVISION,
        local_dir=str(RAW_DIR),
    )
    return Path(local_path)


# --- Extracción de imágenes ----------------------------------------------------
def resolve_images_layout(zf: zipfile.ZipFile) -> tuple[Path, Path, int]:
    """Determina, sin extraer, dónde dejar las imágenes de forma que acaben
    siempre en data/processed/images/.

    Devuelve (destino_extraccion, directorio_imagenes, n_imagenes_en_zip).
    Maneja dos layouts típicos del zip:
      - con carpeta contenedora única (p. ej. 'images/xxx.jpg'): se extrae al
        padre (data/processed/) y las imágenes quedan en data/processed/<carpeta>/.
      - plano (p. ej. 'xxx.jpg' en la raíz del zip): se extrae directamente a
        data/processed/images/.
    """
    names = [n for n in zf.namelist() if not n.endswith("/")]
    image_names = [n for n in names if is_image_name(n)]
    top_dirs = {n.split("/")[0] for n in names if "/" in n}
    flat_files = [n for n in names if "/" not in n]

    if len(top_dirs) == 1 and not flat_files:
        extract_to = PROCESSED_DIR
        images_dir = PROCESSED_DIR / next(iter(top_dirs))
    else:
        extract_to = PROCESSED_DIR / "images"
        images_dir = PROCESSED_DIR / "images"
    return extract_to, images_dir, len(image_names)


def extract_images() -> tuple[Path, int, int]:
    """Extrae images.zip de forma idempotente.

    Devuelve (directorio_imagenes, n_extraidas, n_en_zip). Si ya está extraído
    (mismo recuento), no vuelve a extraer.
    """
    if not IMAGES_ZIP.exists():
        raise FileNotFoundError(f"No se encontró {IMAGES_ZIP}. Ejecuta primero la descarga.")

    with zipfile.ZipFile(IMAGES_ZIP) as zf:
        extract_to, images_dir, n_in_zip = resolve_images_layout(zf)
        existing = count_image_files(images_dir)
        if existing >= n_in_zip > 0:
            print(f"  [extracción] Imágenes ya presentes ({existing} >= {n_in_zip}); se omite.")
            return images_dir, existing, n_in_zip
        print(f"  [extracción] Extrayendo {n_in_zip} imágenes a {images_dir} ...")
        extract_to.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_to)

    extracted = count_image_files(images_dir)
    return images_dir, extracted, n_in_zip


# --- Procedencia ---------------------------------------------------------------
def build_provenance(images_dir: Path, n_images: int, n_in_zip: int) -> dict:
    """Construye el diccionario de procedencia con recuentos y hashes."""
    # Metadatos remotos (best-effort; no debe hacer fallar una re-ejecución offline).
    remote = {}
    try:
        info = HfApi().dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
        remote = {
            "resolved_sha": info.sha,
            "last_modified": str(info.lastModified),
            "remote_files": {s.rfilename: (s.size or 0) for s in info.siblings},
        }
    except Exception as exc:  # noqa: BLE001 — provenance no debe romper por red
        remote = {"warning": f"No se pudieron obtener metadatos remotos: {exc!r}"}

    raw_files = []
    for f in sorted(RAW_DIR.rglob("*")):
        if f.is_file() and ".cache" not in f.parts:
            raw_files.append(
                {
                    "path": f.relative_to(RAW_DIR).as_posix(),
                    "size_bytes": f.stat().st_size,
                    "sha256": sha256_of_file(f),
                }
            )

    record_counts = {}
    for rel in JSONL_RELPATHS:
        p = RAW_DIR / rel
        record_counts[rel] = count_jsonl_records(p) if p.exists() else None

    return {
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "pinned_revision": REVISION,
        "download_utc": datetime.now(timezone.utc).isoformat(),
        "remote": remote,
        "raw_files": raw_files,
        "record_counts": record_counts,
        "images": {
            "zip_entries_images": n_in_zip,
            "extracted_files": n_images,
            "images_dir": images_dir.relative_to(PROJECT_ROOT).as_posix(),
        },
        "disk_usage": {
            "data_raw_bytes": dir_size_bytes(RAW_DIR),
            "data_processed_bytes": dir_size_bytes(PROCESSED_DIR),
        },
    }


def main() -> None:
    print("=" * 70)
    print("TAREA A — Descarga y extracción de MedXpertQA")
    print("=" * 70)
    print(f"Repo:      {REPO_ID} ({REPO_TYPE})")
    print(f"Revisión:  {REVISION}")
    print(f"Destino:   {RAW_DIR}")
    print("-" * 70)

    print("[1/3] Descargando snapshot (idempotente) ...")
    download_snapshot()
    present = [rel for rel in JSONL_RELPATHS if (RAW_DIR / rel).exists()]
    print(f"      JSONL presentes: {len(present)}/{len(JSONL_RELPATHS)} -> {present}")
    print(f"      images.zip presente: {IMAGES_ZIP.exists()} "
          f"({human_size(IMAGES_ZIP.stat().st_size) if IMAGES_ZIP.exists() else 'N/A'})")

    print("[2/3] Extrayendo imágenes ...")
    images_dir, n_images, n_in_zip = extract_images()

    print("[3/3] Registrando procedencia ...")
    provenance = build_provenance(images_dir, n_images, n_in_zip)
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Resumen -------------------------------------------------------------
    print("-" * 70)
    print("RESUMEN")
    print(f"  Revisión descargada : {REVISION}")
    print(f"  SHA remoto resuelto : {provenance['remote'].get('resolved_sha', 'N/D')}")
    print(f"  Ficheros JSONL      : {len(present)}")
    for rel in JSONL_RELPATHS:
        print(f"      {rel:16s} -> {provenance['record_counts'][rel]} registros")
    print(f"  Imágenes extraídas  : {n_images} (entradas de imagen en zip: {n_in_zip})")
    print(f"  Directorio imágenes : {provenance['images']['images_dir']}")
    print(f"  Tamaño data/raw       : {human_size(provenance['disk_usage']['data_raw_bytes'])}")
    print(f"  Tamaño data/processed : {human_size(provenance['disk_usage']['data_processed_bytes'])}")
    print(f"  Procedencia escrita : {PROVENANCE_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
