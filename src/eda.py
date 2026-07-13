"""
src/eda.py — Tarea C del Paso 1 (TFG MedXpertQA).

Funciones reutilizables para el análisis exploratorio (C.1–C.5). Cada análisis:
  - calcula estadísticas y las guarda como tabla en outputs/tables/,
  - genera figuras a 300 dpi (etiquetadas en español) en outputs/figures/,
  - devuelve un diccionario de resultados usado por el informe.

El módulo es la única fuente de cómputo: el notebook (notebooks/01_eda.ipynb) y
los informes (01_eda.md, data_card.md) se construyen a partir de estas funciones,
de modo que toda cifra procede de código ejecutado.

Reproducibilidad: semilla fija (SEED=42). Los cómputos pesados (propiedades de
imagen + hashes, longitudes de texto) se cachean en outputs/tables/ y se reutilizan.

Uso (genera TODAS las tablas, figuras e informes):
    python src\\eda.py
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import imagehash

# --- Reproducibilidad ----------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- Rutas ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "images"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TAB_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "outputs" / "report"
for _d in (FIG_DIR, TAB_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FILES = {
    "MM": {"test": RAW_DIR / "MM" / "test.jsonl", "dev": RAW_DIR / "MM" / "dev.jsonl"},
    "Text": {"test": RAW_DIR / "Text" / "test.jsonl", "dev": RAW_DIR / "Text" / "dev.jsonl"},
}
EXPECTED_OPTIONS = {"MM": 5, "Text": 10}

# --- Paleta (dataviz: hues validados CVD-safe, orden fijo) ---------------------
PALETTE = {
    "blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948", "magenta": "#e87ba4", "orange": "#eb6834",
}
CAT_ORDER = [PALETTE[k] for k in
             ("blue", "aqua", "yellow", "green", "violet", "red", "magenta", "orange")]
C_MM = PALETTE["blue"]        # serie MM
C_TEXT = PALETTE["orange"]    # serie Text
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BLUE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "medx_blue", ["#eef4fc", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])


def setup_style() -> None:
    """Estilo sobrio y homogéneo para todas las figuras (apto para memoria)."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
        "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "legend.frameon": False,
    })


# --- Carga ---------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_subset(modality: str, split: str = "test") -> pd.DataFrame:
    df = pd.DataFrame(load_jsonl(FILES[modality][split]))
    df["n_options"] = df["options"].apply(len)
    df["n_labels"] = df["label"].apply(lambda x: 1 if isinstance(x, str) else len(x))
    if "images" in df.columns:
        df["n_images"] = df["images"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    return df


def load_all() -> dict:
    """DataFrames del conjunto de test (objeto principal del EDA)."""
    return {"MM": load_subset("MM", "test"), "Text": load_subset("Text", "test")}


# --- Helper de guardado de figuras ---------------------------------------------
def savefig(fig, name: str, dpi: int = 300) -> Path:
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    return out


def _annotate_bars(ax, bars, fmt="{:.0f}", horizontal=False, pad=1.0):
    for b in bars:
        if horizontal:
            w = b.get_width()
            ax.text(w + pad, b.get_y() + b.get_height() / 2, fmt.format(w),
                    va="center", ha="left", fontsize=8.5, color=INK2)
        else:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + pad, fmt.format(h),
                    va="bottom", ha="center", fontsize=8.5, color=INK2)


# ============================================================================
# C.1 — Metadatos y estructura
# ============================================================================
def c1_metadata_distributions(dfs: dict) -> dict:
    results = {}
    fields = ["body_system", "medical_task", "question_type"]
    for field in fields:
        mm = dfs["MM"][field].value_counts()
        tx = dfs["Text"][field].value_counts()
        cats = sorted(set(mm.index) | set(tx.index))
        tab = pd.DataFrame(index=cats)
        tab["MM_n"] = [int(mm.get(c, 0)) for c in cats]
        tab["Text_n"] = [int(tx.get(c, 0)) for c in cats]
        tab["MM_pct"] = (100 * tab["MM_n"] / tab["MM_n"].sum()).round(2)
        tab["Text_pct"] = (100 * tab["Text_n"] / tab["Text_n"].sum()).round(2)
        tab = tab.assign(_t=tab["MM_n"] + tab["Text_n"]).sort_values(
            "_t", ascending=False).drop(columns="_t")
        tab.index.name = field
        tab.to_csv(TAB_DIR / f"c1_dist_{field}.csv")

        # Figura: barras agrupadas MM vs Text (en %)
        cats_sorted = list(tab.index)
        mm_v, tx_v = tab["MM_pct"].values, tab["Text_pct"].values
        horizontal = len(cats_sorted) > 4
        if horizontal:
            fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(cats_sorted) + 1.2))
            y = np.arange(len(cats_sorted))[::-1]
            hgt = 0.38
            b1 = ax.barh(y + hgt / 2, mm_v, height=hgt, color=C_MM, label="MM (n=%d)" % tab["MM_n"].sum())
            b2 = ax.barh(y - hgt / 2, tx_v, height=hgt, color=C_TEXT, label="Text (n=%d)" % tab["Text_n"].sum())
            ax.set_yticks(y); ax.set_yticklabels(cats_sorted)
            ax.set_xlabel("Porcentaje del subconjunto (%)")
            _annotate_bars(ax, b1, "{:.1f}", horizontal=True, pad=0.2)
            _annotate_bars(ax, b2, "{:.1f}", horizontal=True, pad=0.2)
            ax.grid(axis="y", visible=False)
        else:
            fig, ax = plt.subplots(figsize=(7, 4.4))
            x = np.arange(len(cats_sorted))
            wd = 0.38
            b1 = ax.bar(x - wd / 2, mm_v, width=wd, color=C_MM, label="MM (n=%d)" % tab["MM_n"].sum())
            b2 = ax.bar(x + wd / 2, tx_v, width=wd, color=C_TEXT, label="Text (n=%d)" % tab["Text_n"].sum())
            ax.set_xticks(x); ax.set_xticklabels(cats_sorted)
            ax.set_ylabel("Porcentaje del subconjunto (%)")
            _annotate_bars(ax, b1, "{:.1f}", pad=0.3)
            _annotate_bars(ax, b2, "{:.1f}", pad=0.3)
            ax.grid(axis="x", visible=False)
        titles = {"body_system": "Distribución por sistema corporal (body_system)",
                  "medical_task": "Distribución por tarea médica (medical_task)",
                  "question_type": "Distribución por tipo de pregunta (question_type)"}
        ax.set_title(titles[field])
        ax.legend(loc="lower right" if horizontal else "upper right")
        savefig(fig, f"c1_dist_{field}")
        results[field] = tab
    return results


def c1_crosstabs(dfs: dict) -> dict:
    results = {}
    pairs = [("question_type", "medical_task"), ("body_system", "medical_task")]
    for rowf, colf in pairs:
        for modality in ("MM", "Text"):
            ct = pd.crosstab(dfs[modality][rowf], dfs[modality][colf])
            ct.to_csv(TAB_DIR / f"c1_cross_{rowf}_x_{colf}_{modality}.csv")
            results[(rowf, colf, modality)] = ct
        # Figura: heatmaps MM y Text lado a lado
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6 if rowf == "question_type" else 7.2))
        for ax, modality in zip(axes, ("MM", "Text")):
            ct = results[(rowf, colf, modality)]
            im = ax.imshow(ct.values, cmap=BLUE_CMAP, aspect="auto")
            ax.set_xticks(range(len(ct.columns))); ax.set_xticklabels(ct.columns, rotation=30, ha="right")
            ax.set_yticks(range(len(ct.index))); ax.set_yticklabels(ct.index)
            ax.set_title(f"{modality}")
            ax.grid(False)
            vmax = ct.values.max()
            for i in range(ct.shape[0]):
                for j in range(ct.shape[1]):
                    v = ct.values[i, j]
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                            color="white" if v > 0.6 * vmax else INK)
        fig.suptitle(f"Tabla cruzada: {rowf} × {colf}", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        savefig(fig, f"c1_cross_{rowf}_x_{colf}")
    return results


def c1_options_answers(dfs: dict) -> dict:
    rows = []
    for modality in ("MM", "Text"):
        df = dfs[modality]
        opt_counts = df["n_options"].value_counts().to_dict()
        lab_counts = df["n_labels"].value_counts().to_dict()
        rows.append({
            "subconjunto": modality,
            "n_opciones_esperado": EXPECTED_OPTIONS[modality],
            "n_opciones_valores": json.dumps({int(k): int(v) for k, v in opt_counts.items()}),
            "todas_con_n_esperado": bool((df["n_options"] == EXPECTED_OPTIONS[modality]).all()),
            "n_respuestas_correctas_valores": json.dumps({int(k): int(v) for k, v in lab_counts.items()}),
            "siempre_1_correcta": bool((df["n_labels"] == 1).all()),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB_DIR / "c1_options_answers.csv", index=False)
    return {"table": tab}


# ============================================================================
# C.2 — Chequeo de sesgos (posición de la respuesta)
# ============================================================================
def c2_label_bias(dfs: dict) -> dict:
    from scipy.stats import chisquare
    results = {}
    dist_rows = {}
    baseline_rows = []
    for modality in ("MM", "Text"):
        df = dfs[modality]
        n_opt = EXPECTED_OPTIONS[modality]
        letters = [chr(ord("A") + i) for i in range(n_opt)]
        counts = df["label"].value_counts()
        obs = np.array([int(counts.get(l, 0)) for l in letters])
        total = obs.sum()
        pct = 100 * obs / total
        dist_rows[modality] = pd.Series(obs, index=letters)

        expected = np.full(n_opt, total / n_opt)
        chi2, pval = chisquare(obs, expected)
        majority_letter = letters[int(obs.argmax())]
        majority_acc = obs.max() / total
        random_acc = 1.0 / n_opt
        baseline_rows.append({
            "subconjunto": modality, "n_opciones": n_opt, "n": int(total),
            "baseline_azar": round(random_acc, 4),
            "baseline_mayoria": round(majority_acc, 4),
            "letra_mayoritaria": majority_letter,
            "chi2": round(float(chi2), 3), "p_valor": float(pval),
            "uniforme_p>0.05": bool(pval > 0.05),
        })

        # Figura por subconjunto
        fig, ax = plt.subplots(figsize=(7, 4.2))
        color = C_MM if modality == "MM" else C_TEXT
        bars = ax.bar(letters, pct, color=color, width=0.72)
        ax.axhline(100 / n_opt, ls="--", lw=1.4, color=INK2,
                   label=f"Uniforme = {100/n_opt:.1f}%")
        _annotate_bars(ax, bars, "{:.1f}", pad=0.15)
        ax.set_xlabel("Letra de la respuesta correcta")
        ax.set_ylabel("Porcentaje de preguntas (%)")
        ax.set_title(f"Distribución de la respuesta correcta — {modality}\n"
                     f"(χ²={chi2:.1f}, p={pval:.3g})")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right")
        savefig(fig, f"c2_label_{modality}")

    dist = pd.DataFrame(dist_rows)
    dist.index.name = "letra"
    dist.to_csv(TAB_DIR / "c2_label_distribution.csv")
    baselines = pd.DataFrame(baseline_rows)
    baselines.to_csv(TAB_DIR / "c2_baselines.csv", index=False)
    results["distribution"] = dist
    results["baselines"] = baselines
    return results


# ============================================================================
# C.3 — Análisis del texto
# ============================================================================
_TOKENIZERS = {}


def get_tokenizers() -> dict:
    """Carga perezosa de tokenizadores. cl100k (tiktoken) siempre; Qwen2.5-VL si se puede."""
    if _TOKENIZERS:
        return _TOKENIZERS
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001
        print("  [tokenizador] tiktoken no disponible:", repr(exc))
    qwen = None
    try:
        from transformers import AutoTokenizer
        qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    except Exception as exc:  # noqa: BLE001
        print("  [tokenizador] Qwen2.5-VL no disponible (se usará solo cl100k):", repr(exc)[:120])
    _TOKENIZERS["cl100k"] = enc
    _TOKENIZERS["qwen"] = qwen
    return _TOKENIZERS


def _full_prompt_text(row) -> str:
    """Enunciado + opciones (estimador del tamaño real del prompt)."""
    opts = row["options"]
    opt_txt = "\n".join(f"{k}. {v}" for k, v in opts.items())
    return f"{row['question']}\n{opt_txt}"


def build_text_lengths(dfs: dict, force: bool = False) -> pd.DataFrame:
    cache = TAB_DIR / "c3_text_lengths.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)
    tok = get_tokenizers()
    enc, qwen = tok["cl100k"], tok["qwen"]
    recs = []
    for modality in ("MM", "Text"):
        df = dfs[modality]
        for _, r in df.iterrows():
            q = r["question"]
            full = _full_prompt_text(r)
            rec = {
                "id": r["id"], "subconjunto": modality,
                "question_type": r["question_type"], "medical_task": r["medical_task"],
                "body_system": r["body_system"],
                "chars": len(q), "words": len(q.split()),
                "chars_full": len(full), "words_full": len(full.split()),
            }
            rec["tok_cl100k"] = len(enc.encode(q)) if enc else np.nan
            rec["tok_cl100k_full"] = len(enc.encode(full)) if enc else np.nan
            rec["tok_qwen"] = len(qwen.encode(q, add_special_tokens=False)) if qwen else np.nan
            rec["tok_qwen_full"] = len(qwen.encode(full, add_special_tokens=False)) if qwen else np.nan
            recs.append(rec)
    out = pd.DataFrame(recs)
    out.to_csv(cache, index=False)
    return out


def c3_text_lengths(dfs: dict, lengths: pd.DataFrame | None = None) -> dict:
    if lengths is None:
        lengths = build_text_lengths(dfs)
    metrics = ["chars", "words", "tok_cl100k", "tok_qwen", "chars_full",
               "words_full", "tok_cl100k_full", "tok_qwen_full"]
    # Resumen por subconjunto
    summ = lengths.groupby("subconjunto")[metrics].agg(
        ["mean", "std", "min", "median", "max",
         lambda s: s.quantile(0.95)]).round(1)
    summ.columns = [f"{m}_{s if not callable(s) else 'p95'}"
                    for m, s in summ.columns.to_flat_index()]
    # renombrar el lambda
    summ = summ.rename(columns=lambda c: c.replace("<lambda_0>", "p95"))
    summ.to_csv(TAB_DIR / "c3_length_summary.csv")

    # Por question_type
    by_qt = lengths.groupby(["subconjunto", "question_type"])[
        ["words", "tok_cl100k", "tok_qwen"]].agg(["mean", "median"]).round(1)
    by_qt.to_csv(TAB_DIR / "c3_length_by_qtype.csv")

    has_qwen = lengths["tok_qwen"].notna().any()
    tok_col = "tok_cl100k"

    # Figura 1: histograma de tokens (cl100k) por subconjunto
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for modality, color in (("MM", C_MM), ("Text", C_TEXT)):
        vals = lengths.loc[lengths.subconjunto == modality, tok_col].dropna()
        ax.hist(vals, bins=40, color=color, alpha=0.6, label=f"{modality} (mediana={vals.median():.0f})")
    ax.set_xlabel("Longitud del enunciado (tokens, cl100k_base)")
    ax.set_ylabel("Nº de preguntas")
    ax.set_title("Distribución de longitud del enunciado en tokens")
    ax.grid(axis="x", visible=False)
    ax.legend()
    savefig(fig, "c3_tokens_hist")

    # Figura 2: boxplot de tokens por question_type y subconjunto
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    groups, labels, colors = [], [], []
    for modality, color in (("MM", C_MM), ("Text", C_TEXT)):
        for qt in ("Reasoning", "Understanding"):
            v = lengths[(lengths.subconjunto == modality) & (lengths.question_type == qt)][tok_col].dropna()
            groups.append(v.values); labels.append(f"{modality}\n{qt}"); colors.append(color)
    bp = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.6); patch.set_edgecolor(INK2)
    for med in bp["medians"]:
        med.set_color(INK); med.set_linewidth(1.6)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Longitud (tokens, cl100k_base)")
    ax.set_title("Longitud del enunciado por tipo de pregunta")
    ax.grid(axis="x", visible=False)
    savefig(fig, "c3_tokens_by_qtype")

    return {"summary": summ, "by_qtype": by_qt, "has_qwen": bool(has_qwen),
            "lengths": lengths}


def _normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def c3_near_duplicates(dfs: dict, sim_threshold: float = 0.85) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    results = {}
    all_pairs = []
    exact_groups_total = 0
    for modality in ("MM", "Text"):
        df = dfs[modality].reset_index(drop=True)
        texts = df["question"].tolist()
        ids = df["id"].tolist()

        # Duplicados exactos (texto normalizado)
        norm = [_normalize_text(t) for t in texts]
        groups = {}
        for i, h in enumerate(norm):
            groups.setdefault(hashlib.md5(h.encode("utf-8")).hexdigest(), []).append(ids[i])
        exact = {k: v for k, v in groups.items() if len(v) > 1}
        exact_groups_total += len(exact)

        # Casi-duplicados por TF-IDF + coseno
        X = TfidfVectorizer(min_df=2, ngram_range=(1, 2)).fit_transform(norm)
        sim = cosine_similarity(X)
        np.fill_diagonal(sim, 0.0)
        max_sim = sim.max(axis=1)
        iu = np.triu_indices_from(sim, k=1)
        mask = sim[iu] >= sim_threshold
        pairs = [(ids[i], ids[j], round(float(sim[i, j]), 3))
                 for i, j in zip(iu[0][mask], iu[1][mask])]
        pairs.sort(key=lambda x: -x[2])
        for a, b, s in pairs:
            all_pairs.append({"subconjunto": modality, "id_a": a, "id_b": b, "coseno": s})

        # Figura: histograma de la máxima similitud por pregunta
        fig, ax = plt.subplots(figsize=(7, 4.2))
        color = C_MM if modality == "MM" else C_TEXT
        ax.hist(max_sim, bins=40, color=color, alpha=0.7)
        ax.axvline(sim_threshold, ls="--", color=INK2, lw=1.4,
                   label=f"Umbral casi-duplicado = {sim_threshold}")
        ax.set_xlabel("Máxima similitud coseno con otra pregunta")
        ax.set_ylabel("Nº de preguntas")
        ax.set_title(f"Similitud textual intra-subconjunto — {modality}")
        ax.grid(axis="x", visible=False)
        ax.legend()
        savefig(fig, f"c3_similarity_{modality}")
        results[modality] = {"n_exact_groups": len(exact), "n_near_pairs": len(pairs),
                             "max_sim_median": float(np.median(max_sim))}

    pairs_df = pd.DataFrame(all_pairs)
    pairs_df.to_csv(TAB_DIR / "c3_near_duplicates.csv", index=False)
    results["n_exact_groups_total"] = exact_groups_total
    results["n_near_pairs_total"] = len(pairs_df)
    results["threshold"] = sim_threshold
    return results


# ============================================================================
# C.4 — Componente visual (solo MM)
# ============================================================================
def _is_grayscale(im: Image.Image, thumb: int = 64, thresh: float = 8.0) -> bool:
    if im.mode in ("L", "1", "I", "I;16", "LA"):
        return True
    rgb = im.convert("RGB").resize((thumb, thumb))
    arr = np.asarray(rgb, dtype=np.int16)
    spread = (np.abs(arr[..., 0] - arr[..., 1]) +
              np.abs(arr[..., 1] - arr[..., 2]) +
              np.abs(arr[..., 0] - arr[..., 2])).mean() / 3.0
    return bool(spread < thresh)


def build_image_table(dfs: dict, force: bool = False) -> pd.DataFrame:
    cache = TAB_DIR / "c4_image_properties.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)
    df = dfs["MM"]
    recs = []
    for _, r in df.iterrows():
        for ref in r["images"]:
            recs.append({"qid": r["id"], "file": Path(ref).name,
                         "body_system": r["body_system"],
                         "question_type": r["question_type"],
                         "medical_task": r["medical_task"]})
    idf = pd.DataFrame(recs)

    props = []
    files = idf["file"].unique()
    for i, name in enumerate(files, 1):
        p = IMAGES_DIR / name
        with Image.open(p) as im:
            w, h = im.size
            fmt = im.format
            mode = im.mode
            gray = _is_grayscale(im)
            ph = str(imagehash.phash(im))
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        props.append({"file": name, "format": fmt, "mode": mode, "width": w, "height": h,
                      "area_px": w * h, "aspect": round(w / h, 4),
                      "filesize": p.stat().st_size, "is_gray": gray, "md5": md5, "phash": ph})
        if i % 500 == 0:
            print(f"      [imagenes] {i}/{len(files)} procesadas ...")
    pdf = pd.DataFrame(props)
    out = idf.merge(pdf, on="file", how="left")
    out.to_csv(cache, index=False)
    return out


def c4_images_per_question(dfs: dict) -> dict:
    df = dfs["MM"]
    counts = df["n_images"].value_counts().sort_index()
    tab = pd.DataFrame({"n_imagenes": counts.index, "n_preguntas": counts.values})
    tab["pct"] = (100 * tab["n_preguntas"] / tab["n_preguntas"].sum()).round(2)
    tab.to_csv(TAB_DIR / "c4_images_per_question.csv", index=False)
    multi = int((df["n_images"] > 1).sum())
    pct_multi = 100 * multi / len(df)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(tab["n_imagenes"].astype(str), tab["n_preguntas"], color=C_MM, width=0.7)
    _annotate_bars(ax, bars, "{:.0f}", pad=4)
    ax.set_xlabel("Nº de imágenes por pregunta")
    ax.set_ylabel("Nº de preguntas")
    ax.set_title(f"Imágenes por pregunta en MM (multi-imagen: {pct_multi:.1f}%)")
    ax.grid(axis="x", visible=False)
    savefig(fig, "c4_images_per_question")
    return {"table": tab, "n_multi": multi, "pct_multi": pct_multi,
            "max_images": int(df["n_images"].max()),
            "total_refs": int(df["n_images"].sum())}


def c4_image_properties(img_df: pd.DataFrame) -> dict:
    uniq = img_df.drop_duplicates("file").copy()
    # Resumen de formato y color
    fmt_counts = uniq["format"].value_counts()
    gray_counts = uniq["is_gray"].value_counts()
    total_bytes = int(uniq["filesize"].sum())
    summary = {
        "n_imagenes": int(len(uniq)),
        "formatos": {str(k): int(v) for k, v in fmt_counts.items()},
        "color_n": int((~uniq["is_gray"]).sum()),
        "gris_n": int(uniq["is_gray"].sum()),
        "footprint_bytes": total_bytes,
        "footprint_mb": round(total_bytes / 1e6, 1),
    }
    # Percentiles de dimensiones
    dims = uniq[["width", "height", "area_px", "aspect", "filesize"]].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(2)
    dims.to_csv(TAB_DIR / "c4_image_dimensions.csv")
    (TAB_DIR / "c4_image_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Figura compuesta: ancho, alto, área, aspecto, tamaño, formato/color
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    ax = axes[0, 0]
    ax.hist(uniq["width"], bins=40, color=C_MM, alpha=0.75)
    ax.set_title("Anchura (px)"); ax.set_xlabel("px"); ax.set_ylabel("Nº imágenes"); ax.grid(axis="x", visible=False)
    ax = axes[0, 1]
    ax.hist(uniq["height"], bins=40, color=C_MM, alpha=0.75)
    ax.set_title("Altura (px)"); ax.set_xlabel("px"); ax.grid(axis="x", visible=False)
    ax = axes[0, 2]
    ax.hist(uniq["area_px"] / 1e6, bins=40, color=C_MM, alpha=0.75)
    ax.set_title("Área (megapíxeles)"); ax.set_xlabel("MP"); ax.grid(axis="x", visible=False)
    ax = axes[1, 0]
    ax.hist(uniq["aspect"], bins=40, color=C_MM, alpha=0.75)
    ax.axvline(1.0, ls="--", color=INK2, lw=1.2)
    ax.set_title("Relación de aspecto (ancho/alto)"); ax.set_xlabel("ratio"); ax.set_ylabel("Nº imágenes"); ax.grid(axis="x", visible=False)
    ax = axes[1, 1]
    ax.hist(uniq["filesize"] / 1024, bins=40, color=C_MM, alpha=0.75)
    ax.set_title("Tamaño de archivo (KB)"); ax.set_xlabel("KB"); ax.grid(axis="x", visible=False)
    ax = axes[1, 2]
    cats = ["Color", "Escala de grises"]
    vals = [summary["color_n"], summary["gris_n"]]
    bars = ax.bar(cats, vals, color=[C_TEXT, MUTED], width=0.6)
    _annotate_bars(ax, bars, "{:.0f}", pad=4)
    ax.set_title("Color vs escala de grises (proxy)"); ax.set_ylabel("Nº imágenes"); ax.grid(axis="x", visible=False)
    fig.suptitle("Propiedades del conjunto de imágenes (MM)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig(fig, "c4_image_properties")

    # Figura de formatos (barra)
    fig2, ax2 = plt.subplots(figsize=(6.4, 4))
    fmts = list(fmt_counts.index)
    bars = ax2.bar(fmts, fmt_counts.values, color=CAT_ORDER[:len(fmts)], width=0.6)
    _annotate_bars(ax2, bars, "{:.0f}", pad=4)
    ax2.set_title("Formato de archivo de imagen"); ax2.set_ylabel("Nº imágenes"); ax2.grid(axis="x", visible=False)
    savefig(fig2, "c4_image_formats")

    return {"summary": summary, "dims": dims}


def c4_contact_sheet(dfs: dict, img_df: pd.DataFrame, per_system: int = 3) -> dict:
    rng = random.Random(SEED)
    df = dfs["MM"]
    # Una imagen (la primera) por pregunta, con su body_system
    first_img = {r["id"]: (r["images"][0], r["body_system"]) for _, r in df.iterrows()}
    by_sys = {}
    for qid, (img, sysname) in first_img.items():
        by_sys.setdefault(sysname, []).append((qid, img))
    systems = sorted(by_sys)
    picks = []
    for s in systems:
        pool = sorted(by_sys[s])
        rng.shuffle(pool)
        for qid, img in pool[:per_system]:
            picks.append((s, qid, img))
    # Rejilla
    n = len(picks)
    ncol = 6
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 2.6))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (sysname, qid, img) in zip(axes, picks):
        try:
            with Image.open(IMAGES_DIR / Path(img).name) as im:
                ax.imshow(im.convert("RGB"))
        except Exception:  # noqa: BLE001
            pass
        ax.set_title(f"{sysname}\n{qid}", fontsize=7.5, color=INK)
    fig.suptitle("Hoja de contactos — muestra estratificada por sistema corporal (MM)\n"
                 f"(semilla={SEED}; {per_system} preguntas por sistema; para anotación manual de modalidad)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, "c4_contact_sheet", dpi=150)  # 150 dpi basta para inspección visual
    manifest = pd.DataFrame(picks, columns=["body_system", "qid", "file"])
    manifest.to_csv(TAB_DIR / "c4_contact_sheet_manifest.csv", index=False)
    return {"n": n, "manifest": manifest}


def c4_image_duplicates(img_df: pd.DataFrame) -> dict:
    uniq = img_df.drop_duplicates("file")
    # Duplicados EXACTOS por md5 (mismos bytes)
    by_md5 = uniq.groupby("md5")["file"].apply(list)
    dup_groups = by_md5[by_md5.apply(len) > 1]
    dup_rows = []
    for md5, files in dup_groups.items():
        qids = img_df[img_df["file"].isin(files)]["qid"].unique().tolist()
        dup_rows.append({"md5": md5, "n_ficheros": len(files),
                         "ficheros": ";".join(sorted(files)), "qids": ";".join(sorted(qids))})
    dup_df = pd.DataFrame(dup_rows)
    dup_df.to_csv(TAB_DIR / "c4_exact_duplicates.csv", index=False)
    return {"n_exact_dup_groups": int(len(dup_df)),
            "n_files_involved": int(sum(r["n_ficheros"] for r in dup_rows)) if dup_rows else 0}


def c4_duplicate_question_pairs(dfs: dict, img_df: pd.DataFrame) -> dict:
    """Cruza imágenes duplicadas (md5) con similitud textual: identifica pares de
    preguntas MM que comparten ≥1 imagen idéntica y mide su parecido de enunciado.
    Un texto muy similar + imagen idéntica sugiere pregunta casi-duplicada
    (relevante para calidad del benchmark y control de fuga de datos)."""
    from itertools import combinations
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    df = dfs["MM"].reset_index(drop=True)
    qid_to_idx = {q: i for i, q in enumerate(df["id"])}
    file_qid = dict(zip(img_df["file"], img_df["qid"]))

    # Nº de imágenes idénticas compartidas por cada par (no ordenado) de preguntas
    uniq = img_df.drop_duplicates("file")
    pair_shared: dict[tuple, int] = {}
    for _, grp in uniq.groupby("md5"):
        files = grp["file"].tolist()
        if len(files) < 2:
            continue
        qids = sorted(set(file_qid[f] for f in files))
        for a, b in combinations(qids, 2):
            pair_shared[(a, b)] = pair_shared.get((a, b), 0) + 1

    if not pair_shared:
        empty = pd.DataFrame(columns=["qid_a", "qid_b", "n_shared_images",
                                      "text_cosine", "near_dup_question"])
        empty.to_csv(TAB_DIR / "c4_duplicate_question_pairs.csv", index=False)
        return {"n_pairs": 0, "n_near_dup_q": 0, "table": empty}

    norm = [_normalize_text(t) for t in df["question"]]
    X = TfidfVectorizer(min_df=2, ngram_range=(1, 2)).fit_transform(norm)
    rows = []
    for (a, b), n_shared in pair_shared.items():
        cos = float(cosine_similarity(X[qid_to_idx[a]], X[qid_to_idx[b]])[0, 0])
        rows.append({"qid_a": a, "qid_b": b, "n_shared_images": n_shared,
                     "text_cosine": round(cos, 3), "near_dup_question": cos > 0.8})
    tab = pd.DataFrame(rows).sort_values(
        ["near_dup_question", "n_shared_images", "text_cosine"], ascending=False)
    tab.to_csv(TAB_DIR / "c4_duplicate_question_pairs.csv", index=False)
    return {"n_pairs": int(len(tab)), "n_near_dup_q": int(tab["near_dup_question"].sum()),
            "table": tab}


# ============================================================================
# C.5 — Relación texto ↔ imagen (solo MM)
# ============================================================================
FIGURE_TERMS = [
    r"figure", r"\bfig\b", r"\bfigs\b", r"image", r"images", r"radiograph", r"photograph",
    r"micrograph", r"\bscan\b", r"shown", r"depicted", r"illustrated", r"pictured",
    r"\bpanel", r"arrow", r"arrowhead", r"\bimaging\b", r"the following",
    r"as seen", r"\bslide\b", r"\bgraph\b", r"tracing",
]
FIGURE_RE = re.compile("|".join(FIGURE_TERMS), re.IGNORECASE)
OPTION_FIGURE_RE = re.compile(r"\bfigure\b|\bpanel\b|\bimage\b", re.IGNORECASE)


def c5_text_image_coupling(dfs: dict) -> dict:
    df = dfs["MM"]
    rows = []
    for _, r in df.iterrows():
        q = r["question"]
        matches = sorted(set(m.group(0).lower() for m in FIGURE_RE.finditer(q)))
        opt_fig = [k for k, v in r["options"].items() if OPTION_FIGURE_RE.search(str(v))]
        rows.append({"id": r["id"], "n_images": r["n_images"],
                     "explicit_mention": bool(matches),
                     "matched_terms": ";".join(matches),
                     "options_reference_figure": bool(opt_fig),
                     "n_options_ref": len(opt_fig)})
    cdf = pd.DataFrame(rows)
    cdf.to_csv(TAB_DIR / "c5_text_image_coupling.csv", index=False)

    n = len(cdf)
    n_explicit = int(cdf["explicit_mention"].sum())
    n_opt_ref = int(cdf["options_reference_figure"].sum())
    pct_explicit = 100 * n_explicit / n
    pct_opt = 100 * n_opt_ref / n

    # Ejemplos
    ex_explicit = df[df["id"].isin(cdf[cdf.explicit_mention].id.head(3))][["id", "question"]]
    ex_implicit = df[df["id"].isin(cdf[~cdf.explicit_mention].id.head(3))][["id", "question"]]

    # Figura: barras (explícita vs implícita) y opciones que refieren figura
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cats = ["Mención explícita\nde la figura", "Sin mención\nexplícita",
            "Opciones que\nrefieren figura"]
    vals = [pct_explicit, 100 - pct_explicit, pct_opt]
    colors = [C_MM, MUTED, C_TEXT]
    bars = ax.bar(cats, vals, color=colors, width=0.62)
    _annotate_bars(ax, bars, "{:.1f}%", pad=0.6)
    ax.set_ylabel("Porcentaje de preguntas MM (%)")
    ax.set_title("Acoplamiento texto ↔ imagen en MM")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    savefig(fig, "c5_text_image_coupling")

    return {"n": n, "n_explicit": n_explicit, "pct_explicit": pct_explicit,
            "n_opt_ref": n_opt_ref, "pct_opt": pct_opt,
            "ex_explicit": ex_explicit, "ex_implicit": ex_implicit,
            "table": cdf}


# ============================================================================
# Informe y data card
# ============================================================================
def _fig(name: str) -> str:
    return f"![{name}](../figures/{name}.png)"


def _provenance() -> dict:
    p = RAW_DIR / "PROVENANCE.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def build_report(res: dict, dfs: dict) -> Path:
    prov = _provenance()
    c1 = res["c1"]
    L = []
    A = L.append
    A("# Análisis Exploratorio de Datos — MedXpertQA (Paso 1 · Tarea C)")
    A("")
    A(f"- **Generado**: {datetime.now(timezone.utc).isoformat()} (UTC), por `src/eda.py` (semilla {SEED}).")
    A(f"- **Fuente**: `{prov.get('repo_id', 'TsinghuaC3I/MedXpertQA')}` "
      f"(revisión `{prov.get('pinned_revision', 'N/D')}`).")
    A(f"- **Alcance**: conjunto de *test* (MM = {len(dfs['MM'])}, Text = {len(dfs['Text'])}). "
      f"El *dev* (5+5) se reserva como pool de few-shot.")
    A("- **Trazabilidad**: cada cifra procede de una función de `src/eda.py`; las tablas están "
      "en `outputs/tables/` y las figuras en `outputs/figures/`.")
    A("")

    # Esquema
    A("## 0. Esquema real de los datos")
    A("")
    A("Confirmado por inspección de los ficheros descargados (no asumido):")
    A("")
    A("| Campo | Tipo | MM | Text |")
    A("|---|---|---|---|")
    A("| `id` | str | `MM-<n>` | `Text-<n>` |")
    A("| `question` | str | enunciado | enunciado |")
    A("| `options` | **dict** | claves A–E (5) | claves A–J (10) |")
    A("| `label` | **str** (1 letra) | ∈ opciones | ∈ opciones |")
    A("| `images` | **list[str]** | 1–6 basenames | *(ausente)* |")
    A("| `medical_task` | str | Diagnosis/Treatment/Basic Science | idem |")
    A("| `body_system` | str | 12 categorías | 12 categorías |")
    A("| `question_type` | str | Reasoning/Understanding | idem |")
    A("")
    A("La integridad del dato está verificada en "
      "[`00_integridad.md`](00_integridad.md) (recuentos 2000/2450, correspondencia "
      "imagen↔archivo exacta, 0 imágenes corruptas, IDs únicos, `label`∈`options` al 100 %).")
    A("")

    # C.1
    A("## C.1 — Metadatos y estructura")
    A("")
    bs = c1["body_system"]
    mt = c1["medical_task"]
    qt = c1["question_type"]
    top_mm_bs = bs["MM_n"].idxmax()
    top_tx_bs = bs["Text_n"].idxmax()
    A("### Distribución por sistema corporal")
    A("")
    A(_fig("c1_dist_body_system"))
    A("")
    A(f"El eje `body_system` cubre 12 categorías. En **MM** domina **{top_mm_bs}** "
      f"({bs.loc[top_mm_bs, 'MM_n']}, {bs.loc[top_mm_bs, 'MM_pct']} %), mientras que en **Text** "
      f"domina **{top_tx_bs}** ({bs.loc[top_tx_bs, 'Text_n']}, {bs.loc[top_tx_bs, 'Text_pct']} %). "
      f"La diferencia más relevante para el diseño es *Integumentary* (piel): "
      f"{bs.loc['Integumentary', 'MM_pct']} % en MM frente a {bs.loc['Integumentary', 'Text_pct']} % "
      f"en Text — coherente con que la dermatología es intrínsecamente visual. Lo mismo ocurre con "
      f"*Skeletal* (traumatología/radiología), sobre-representado en MM.")
    A("")
    A("### Distribución por tarea médica y tipo de pregunta")
    A("")
    A(_fig("c1_dist_medical_task"))
    A("")
    A(_fig("c1_dist_question_type"))
    A("")
    A(f"*Diagnosis* es la tarea mayoritaria en ambos (MM {mt.loc['Diagnosis', 'MM_pct']} %, "
      f"Text {mt.loc['Diagnosis', 'Text_pct']} %). En cuanto al tipo, **Reasoning** predomina "
      f"claramente (MM {qt.loc['Reasoning', 'MM_pct']} %, Text {qt.loc['Reasoning', 'Text_pct']} %): "
      f"el benchmark está diseñado para exigir razonamiento clínico, no mero reconocimiento.")
    A("")
    A("### Tablas cruzadas")
    A("")
    A(_fig("c1_cross_question_type_x_medical_task"))
    A("")
    A(_fig("c1_cross_body_system_x_medical_task"))
    A("")
    A("El cuadrante dominante es **Reasoning × Diagnosis**, seguido de Reasoning × Treatment. "
      "Las tablas completas están en `outputs/tables/c1_cross_*.csv`.")
    A("")
    oa = res["c1o"]["table"]
    A("### Nº de opciones y de respuestas correctas")
    A("")
    A(f"- MM: todas las preguntas tienen exactamente 5 opciones "
      f"(`{oa.loc[0, 'todas_con_n_esperado']}`); Text, 10 opciones "
      f"(`{oa.loc[1, 'todas_con_n_esperado']}`).")
    A(f"- **Siempre 1 respuesta correcta**: MM `{oa.loc[0, 'siempre_1_correcta']}`, "
      f"Text `{oa.loc[1, 'siempre_1_correcta']}`. El formato es de opción única "
      f"(single-label), lo que habilita una métrica de *accuracy* directa por letra.")
    A("")

    # C.2
    bl = res["c2"]["baselines"].set_index("subconjunto")
    A("## C.2 — Chequeo de sesgos (posición de la respuesta)")
    A("")
    A(_fig("c2_label_MM"))
    A("")
    A(_fig("c2_label_Text"))
    A("")
    A(f"La respuesta correcta se reparte entre las letras sin sesgo posicional explotable. "
      f"En **MM**, la baseline de azar es {bl.loc['MM', 'baseline_azar']*100:.0f} % y la de "
      f"«responder siempre la letra más frecuente» ({bl.loc['MM', 'letra_mayoritaria']}) sube solo a "
      f"{bl.loc['MM', 'baseline_mayoria']*100:.1f} %. En **Text**, azar "
      f"{bl.loc['Text', 'baseline_azar']*100:.0f} % vs mayoría "
      f"{bl.loc['Text', 'baseline_mayoria']*100:.1f} % (letra {bl.loc['Text', 'letra_mayoritaria']}). "
      f"El test χ² de uniformidad da p={bl.loc['MM', 'p_valor']:.3g} (MM) y "
      f"p={bl.loc['Text', 'p_valor']:.3g} (Text): "
      + ("no se rechaza la uniformidad, por lo que no hay atajo posicional."
         if (bl.loc['MM', 'p_valor'] > 0.05 and bl.loc['Text', 'p_valor'] > 0.05)
         else "conviene reportar la baseline de mayoría junto a la de azar como referencia honesta.")
      )
    A("")
    A("> **Implicación**: cualquier modelo debe superar con holgura estas baselines "
      f"({bl.loc['MM', 'baseline_mayoria']*100:.1f} % en MM, {bl.loc['Text', 'baseline_mayoria']*100:.1f} % en Text) "
      "para considerarse informativo.")
    A("")

    # C.3
    c3 = res["c3"]
    summ = c3["summary"]
    dd = res["c3d"]
    A("## C.3 — Análisis del texto")
    A("")
    A(_fig("c3_tokens_hist"))
    A("")
    A(_fig("c3_tokens_by_qtype"))
    A("")
    qwen_note = ("Se reporta también el tokenizador de Qwen2.5-VL." if c3["has_qwen"]
                 else "El tokenizador de Qwen2.5-VL no pudo cargarse en este entorno; "
                      "se usa cl100k_base como estimador (ver `c3_length_summary.csv`).")
    A(f"Longitud del **enunciado** (mediana de tokens cl100k): "
      f"MM ≈ {summ.loc['MM', 'tok_cl100k_median']:.0f}, Text ≈ {summ.loc['Text', 'tok_cl100k_median']:.0f}; "
      f"el percentil 95 llega a {summ.loc['MM', 'tok_cl100k_p95']:.0f} (MM) y "
      f"{summ.loc['Text', 'tok_cl100k_p95']:.0f} (Text). Incluyendo las opciones "
      f"(prompt completo), la mediana sube a {summ.loc['MM', 'tok_cl100k_full_median']:.0f} (MM) y "
      f"{summ.loc['Text', 'tok_cl100k_full_median']:.0f} (Text). {qwen_note} "
      f"Las preguntas de *Text* son más largas (10 opciones, muchas con viñetas clínicas extensas), "
      f"lo que anticipa mayor coste de contexto en la fase de evaluación.")
    A("")
    A("### Casi-duplicados de texto")
    A("")
    A(_fig("c3_similarity_MM"))
    A("")
    A(_fig("c3_similarity_Text"))
    A("")
    A(f"Con normalización + hash exacto no se detectan grupos de enunciados idénticos "
      f"(**{dd['n_exact_groups_total']}** grupos). Por TF-IDF + coseno (umbral {dd['threshold']}) "
      f"aparecen **{dd['n_near_pairs_total']}** pares de alta similitud "
      f"(ver `c3_near_duplicates.csv`), útiles como línea base para el control de contaminación "
      f"cuando se introduzcan datasets de entrenamiento.")
    A("")

    # C.4
    c4a = res["c4a"]
    c4b = res["c4b"]["summary"]
    c4d = res["c4d"]
    c4e = res["c4e"]
    A("## C.4 — Componente visual (solo MM)")
    A("")
    A(_fig("c4_images_per_question"))
    A("")
    A(f"El **{c4a['pct_multi']:.1f} %** de las preguntas MM son multi-imagen "
      f"(hasta {c4a['max_images']} imágenes; {c4a['total_refs']} referencias totales). Esto es "
      f"determinante para el pipeline: no todos los VLM open-source manejan bien varias imágenes "
      f"por prompt, y habrá que decidir estrategia de composición (mosaico único vs varias imágenes).")
    A("")
    A(_fig("c4_image_properties"))
    A("")
    A(_fig("c4_image_formats"))
    A("")
    fmt_str = ", ".join(f"{k}: {v}" for k, v in c4b["formatos"].items())
    A(f"Formatos: {fmt_str}. Reparto color/gris (proxy): **{c4b['color_n']}** en color y "
      f"**{c4b['gris_n']}** en escala de grises — una señal indirecta de modalidad "
      f"(radiografía/TAC/RM tienden a gris; histología y fotografía clínica, a color). "
      f"Huella total en disco: **{c4b['footprint_mb']} MB** para {c4b['n_imagenes']} imágenes "
      f"(relevante para el almacenamiento en el VPS).")
    A("")
    A("### Hoja de contactos para anotación de modalidad")
    A("")
    A(_fig("c4_contact_sheet"))
    A("")
    A("**Limitación documentada**: el dataset **no** incluye etiqueta de modalidad de imagen "
      "(radiografía / TAC / RM / histología / fotografía clínica / ECG…). No se infiere ni se inventa. "
      "Como caracterización cualitativa, la hoja de contactos anterior es una muestra aleatoria "
      "estratificada por `body_system` (semilla fija) para inspección y anotación manual. "
      "*Método reproducible propuesto* (marcado como aproximación, no ejecutado aquí): heurística "
      "color/aspecto + *clustering* de embeddings visuales (p. ej. del propio encoder del VLM) para "
      "agrupar modalidades sin supervisión, validado sobre una muestra anotada a mano.")
    A("")
    A(f"### Hashes y duplicados exactos")
    A("")
    A(f"Se calcularon hash perceptual (pHash) y hash exacto (md5) de las {c4b['n_imagenes']} imágenes "
      f"(tabla `c4_image_properties.csv`), reutilizables para detectar solapamiento con datasets de "
      f"entrenamiento (control de fuga de datos). Duplicados **exactos** (mismos bytes) dentro de MM: "
      f"**{c4d['n_exact_dup_groups']}** grupos"
      + (f" ({c4d['n_files_involved']} ficheros implicados; ver `c4_exact_duplicates.csv`)."
         if c4d['n_exact_dup_groups'] else " (ninguno)."))
    A("")
    if c4e["n_pairs"]:
        near = c4e["table"][c4e["table"]["near_dup_question"]]
        ejemplo = ""
        if len(near):
            r0 = near.iloc[0]
            ejemplo = (f" El caso más claro es **{r0['qid_a']} / {r0['qid_b']}** "
                       f"(imagen idéntica y similitud textual {r0['text_cosine']}).")
        verbo = "tiene" if c4e["n_near_dup_q"] == 1 else "tienen"
        A(f"**Cruce imagen↔texto (hallazgo de calidad del benchmark).** "
          f"Estos duplicados exactos implican **{c4e['n_pairs']}** pares de preguntas MM que "
          f"comparten al menos una imagen idéntica. De ellos, **{c4e['n_near_dup_q']}** {verbo} además "
          f"un enunciado muy similar (coseno TF-IDF > 0,8), lo que apunta a preguntas "
          f"**casi-duplicadas** dentro del propio test set.{ejemplo} El resto reutiliza la misma "
          f"imagen en preguntas clínicamente distintas. Detalle en "
          f"`c4_duplicate_question_pairs.csv`. Es un dato a tener en cuenta al interpretar la "
          f"*accuracy* agregada y como línea base para el control de fuga de datos.")
        A("")

    # C.5
    c5 = res["c5"]
    A("## C.5 — Relación texto ↔ imagen (solo MM)")
    A("")
    A(_fig("c5_text_image_coupling"))
    A("")
    A(f"El **{c5['pct_explicit']:.1f} %** de las preguntas MM menciona explícitamente la figura en el "
      f"enunciado (términos como *figure, shown, radiograph, image, arrow…*); el resto la referencia "
      f"de forma implícita (p. ej. «este paciente presenta…»). Además, en el "
      f"**{c5['pct_opt']:.1f} %** de las preguntas alguna **opción de respuesta** remite a una figura, "
      f"casos en los que la imagen es imprescindible para responder.")
    A("")
    ex = c5["ex_explicit"].head(1)
    im = c5["ex_implicit"].head(1)
    if len(ex):
        A(f"*Ejemplo con mención explícita* (`{ex.iloc[0]['id']}`): "
          f"«{ex.iloc[0]['question'][:180].strip()}…»")
        A("")
    if len(im):
        A(f"*Ejemplo sin mención explícita* (`{im.iloc[0]['id']}`): "
          f"«{im.iloc[0]['question'][:180].strip()}…»")
        A("")
    A("> **Conclusión de acoplamiento multimodal**: aunque no todas las preguntas nombran la figura, "
      "todas las de MM traen imagen y una fracción sustancial es visualmente dependiente "
      "(mención explícita y/o opciones que remiten a figuras). El subconjunto MM **no** es resoluble "
      "de forma fiable solo con texto; la imagen es central, no accesoria.")
    A("")

    # Hallazgos clave
    A("## Hallazgos clave e implicaciones para el diseño del sistema")
    A("")
    A(f"1. **VLM con soporte multi-imagen**: el {c4a['pct_multi']:.1f} % de MM usa varias imágenes "
      f"(hasta {c4a['max_images']}). Conviene un VLM que acepte múltiples imágenes por prompt "
      "(Qwen2.5-VL las soporta) o una estrategia de mosaico bien definida.")
    A(f"2. **Presupuesto de contexto**: mediana ≈ {summ.loc['Text', 'tok_cl100k_full_median']:.0f} tokens "
      f"de prompt en Text y percentil 95 en torno a {summ.loc['Text', 'tok_cl100k_full_p95']:.0f}; "
      "más los tokens visuales en MM. Dimensionar la ventana y el coste en consecuencia.")
    A("3. **Baselines honestas**: reportar azar y mayoría (C.2) evita sobre-estimar el mérito del "
      "modelo; el sesgo posicional no es explotable.")
    A("4. **Dificultad = razonamiento**: predominio de *Reasoning × Diagnosis* → el sistema se pondrá "
      "a prueba en integración de evidencia clínica, no en reconocimiento superficial; el CoT y el RAG "
      "deberían aportar aquí.")
    A("5. **Modalidad no etiquetada**: limitación real; si la modalidad importa para el análisis de "
      "errores, habrá que anotarla (heurística/clustering) como paso aparte.")
    A(f"6. **Sesgo temático MM**: sobre-peso de dermatología y traumatología; el rendimiento por "
      "`body_system` debe reportarse desagregado para no ocultar debilidades por especialidad.")
    A("")

    # Decisiones pendientes
    A("## Decisiones pendientes (para validación)")
    A("")
    A("- **dev.jsonl**: se mantiene como pool de few-shot; el EDA se centra en *test*. (Anomalía "
      "documentada: `Basic Medicine` en MM/dev ≡ `Basic Science` de test.)")
    A("- **Preguntas multi-imagen**: cuantificadas aquí; el tratamiento en el pipeline "
      "(mosaico vs multi-imagen nativa) queda para el diseño del sistema.")
    A(f"- **Tokenizador Qwen2.5-VL**: {'cargado y reportado.' if c3['has_qwen'] else 'no disponible en este entorno; se usó cl100k_base.'}")
    A("")

    out = REPORT_DIR / "01_eda.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def build_data_card(res: dict, dfs: dict) -> Path:
    prov = _provenance()
    c4b = res["c4b"]["summary"]
    c1 = res["c1"]
    L = []
    A = L.append
    A("# Data Card — MedXpertQA (subconjunto de test)")
    A("")
    A("| Campo | Valor |")
    A("|---|---|")
    A(f"| Nombre | MedXpertQA (ICML 2025) |")
    A(f"| Repositorio | `{prov.get('repo_id', 'TsinghuaC3I/MedXpertQA')}` (Hugging Face, dataset) |")
    A(f"| Revisión (commit) | `{prov.get('pinned_revision', 'N/D')}` |")
    A(f"| Licencia | MIT |")
    A(f"| Descargado | {prov.get('download_utc', 'N/D')} |")
    A(f"| Splits | MM/test={len(dfs['MM'])}, Text/test={len(dfs['Text'])}, dev=5+5 (few-shot) |")
    A(f"| Formato preguntas | opción única; MM 5 opciones (A–E), Text 10 (A–J) |")
    A(f"| Metadatos | body_system (12), medical_task (3), question_type (2) |")
    A(f"| Imágenes (MM) | {c4b['n_imagenes']} ficheros, {c4b['footprint_mb']} MB; "
      f"formatos: {', '.join(f'{k}={v}' for k, v in c4b['formatos'].items())} |")
    A(f"| Color / gris (proxy) | {c4b['color_n']} color, {c4b['gris_n']} gris |")
    A(f"| Multi-imagen | {res['c4a']['pct_multi']:.1f} % de MM (hasta {res['c4a']['max_images']}) |")
    A(f"| Modalidad de imagen | **no etiquetada** en el dataset (limitación) |")
    A("")
    A("## Distribuciones principales (test)")
    A("")
    A("**body_system** (Top-5 por total):")
    A("")
    bs = c1["body_system"].head(5)
    A("| Sistema | MM (n / %) | Text (n / %) |")
    A("|---|---|---|")
    for idx, row in bs.iterrows():
        A(f"| {idx} | {int(row['MM_n'])} / {row['MM_pct']}% | {int(row['Text_n'])} / {row['Text_pct']}% |")
    A("")
    A("## Procedencia y reproducción")
    A("")
    A("- Descarga: `python src/download.py` (revisión fijada; provenance en `data/raw/PROVENANCE.json`).")
    A("- Integridad: `python src/verify.py` → `outputs/report/00_integridad.md`.")
    A("- EDA: `python src/eda.py` o `notebooks/01_eda.ipynb` → tablas, figuras y `01_eda.md`.")
    A("")
    A("## Uso previsto y limitaciones")
    A("")
    A("- Benchmark de **evaluación** (no de entrenamiento) de QA médico experto.")
    A("- El subconjunto MM requiere la imagen para responder de forma fiable (no resoluble solo con texto).")
    A("- Sin etiqueta de modalidad de imagen; distribución temática sesgada hacia algunas especialidades.")
    A("")
    out = REPORT_DIR / "data_card.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


# ============================================================================
# Entregables transversales: índice navegable y one-pager
# ============================================================================
# (título, descripción) por figura y por tabla, en orden de aparición.
FIGURE_CATALOG = [
    ("c1_dist_body_system", "Distribución por sistema corporal (body_system), MM vs Text (%)."),
    ("c1_dist_medical_task", "Distribución por tarea médica (medical_task), MM vs Text (%)."),
    ("c1_dist_question_type", "Distribución por tipo de pregunta (question_type), MM vs Text (%)."),
    ("c1_cross_question_type_x_medical_task", "Tabla cruzada question_type × medical_task (mapas de calor MM y Text)."),
    ("c1_cross_body_system_x_medical_task", "Tabla cruzada body_system × medical_task (mapas de calor MM y Text)."),
    ("c2_label_MM", "Distribución de la letra correcta en MM (A–E) frente a la uniforme."),
    ("c2_label_Text", "Distribución de la letra correcta en Text (A–J) frente a la uniforme."),
    ("c3_tokens_hist", "Histograma de longitud del enunciado en tokens (cl100k) por subconjunto."),
    ("c3_tokens_by_qtype", "Longitud del enunciado por tipo de pregunta (diagramas de caja)."),
    ("c3_similarity_MM", "Máxima similitud coseno intra-subconjunto en MM."),
    ("c3_similarity_Text", "Máxima similitud coseno intra-subconjunto en Text."),
    ("c4_images_per_question", "Número de imágenes por pregunta en MM."),
    ("c4_image_properties", "Propiedades de las imágenes MM: anchura, altura, área, aspecto, tamaño y color/gris."),
    ("c4_image_formats", "Formato de archivo de las imágenes MM."),
    ("c4_contact_sheet", "Hoja de contactos: muestra estratificada por body_system para anotación manual de modalidad."),
    ("c5_text_image_coupling", "Acoplamiento texto↔imagen en MM: mención explícita, implícita y opciones que refieren figura."),
]
TABLE_CATALOG = [
    ("c1_dist_body_system.csv", "Conteos y porcentajes por sistema corporal (MM y Text)."),
    ("c1_dist_medical_task.csv", "Conteos y porcentajes por tarea médica."),
    ("c1_dist_question_type.csv", "Conteos y porcentajes por tipo de pregunta."),
    ("c1_cross_question_type_x_medical_task_MM.csv", "Tabla cruzada question_type × medical_task (MM)."),
    ("c1_cross_question_type_x_medical_task_Text.csv", "Tabla cruzada question_type × medical_task (Text)."),
    ("c1_cross_body_system_x_medical_task_MM.csv", "Tabla cruzada body_system × medical_task (MM)."),
    ("c1_cross_body_system_x_medical_task_Text.csv", "Tabla cruzada body_system × medical_task (Text)."),
    ("c1_options_answers.csv", "Verificación del nº de opciones y de respuestas correctas por subconjunto."),
    ("c2_baselines.csv", "Baselines de azar y de mayoría + test χ² de uniformidad."),
    ("c2_label_distribution.csv", "Frecuencia de cada letra correcta por subconjunto."),
    ("c3_length_summary.csv", "Resumen estadístico de longitudes (caracteres/palabras/tokens) por subconjunto."),
    ("c3_length_by_qtype.csv", "Longitudes por subconjunto y tipo de pregunta."),
    ("c3_near_duplicates.csv", "Pares de preguntas casi-duplicadas por texto (coseno ≥ 0,85)."),
    ("c3_text_lengths.csv", "Tabla base por pregunta: caracteres, palabras y tokens (cl100k y Qwen2.5-VL)."),
    ("c4_images_per_question.csv", "Distribución del nº de imágenes por pregunta."),
    ("c4_image_dimensions.csv", "Percentiles de anchura, altura, área, aspecto y tamaño de archivo."),
    ("c4_image_summary.json", "Resumen: formatos, reparto color/gris y huella en disco."),
    ("c4_image_properties.csv", "Propiedades + hashes (pHash, md5) por imagen (2.852 filas)."),
    ("c4_contact_sheet_manifest.csv", "Manifiesto de las imágenes incluidas en la hoja de contactos."),
    ("c4_exact_duplicates.csv", "Grupos de imágenes byte-idénticas (md5) dentro de MM."),
    ("c4_duplicate_question_pairs.csv", "Pares de preguntas que comparten imagen idéntica + su similitud textual."),
    ("c5_text_image_coupling.csv", "Por pregunta: mención de figura, términos detectados y opciones que refieren figura."),
]
REPORT_CATALOG = [
    ("00_integridad.md", "Informe de verificación de integridad (Tarea B)."),
    ("01_eda.md", "Informe de EDA integrado con hallazgos e implicaciones de diseño (Tarea C)."),
    ("data_card.md", "Data card resumida del dataset."),
    ("indice_figuras_tablas.md", "Este índice navegable de figuras, tablas e informes."),
    ("hallazgos_one_pager.md", "Resumen ejecutivo de una página con los hallazgos clave."),
]


def build_figure_table_index() -> Path:
    """Índice navegable (lista de figuras y de tablas) para la memoria del TFG."""
    L = []
    A = L.append
    A("# Índice de figuras, tablas e informes — EDA MedXpertQA")
    A("")
    A(f"Generado por `src/eda.py`. {len(FIGURE_CATALOG)} figuras (PNG 300 dpi), "
      f"{len(TABLE_CATALOG)} tablas y {len(REPORT_CATALOG)} informes. Enlaces relativos "
      f"a este fichero (`outputs/report/`).")
    A("")
    A("## Lista de figuras")
    A("")
    A("| Nº | Figura | Descripción |")
    A("|---:|---|---|")
    for i, (name, desc) in enumerate(FIGURE_CATALOG, 1):
        exists = (FIG_DIR / f"{name}.png").exists()
        mark = "" if exists else " ⚠ (no encontrada)"
        A(f"| {i} | [`{name}.png`](../figures/{name}.png){mark} | {desc} |")
    A("")
    A("## Lista de tablas")
    A("")
    A("| Nº | Tabla | Descripción |")
    A("|---:|---|---|")
    for i, (name, desc) in enumerate(TABLE_CATALOG, 1):
        exists = (TAB_DIR / name).exists()
        mark = "" if exists else " ⚠ (no encontrada)"
        A(f"| {i} | [`{name}`](../tables/{name}){mark} | {desc} |")
    A("")
    A("## Informes")
    A("")
    A("| Documento | Descripción |")
    A("|---|---|")
    for name, desc in REPORT_CATALOG:
        A(f"| [`{name}`]({name}) | {desc} |")
    A("")
    out = REPORT_DIR / "indice_figuras_tablas.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def build_onepager(res: dict, dfs: dict) -> Path:
    """Resumen ejecutivo de una página con los hallazgos clave (cifras de `res`)."""
    bs = res["c1"]["body_system"]
    bl = res["c2"]["baselines"].set_index("subconjunto")
    summ = res["c3"]["summary"]
    c4a, c4b = res["c4a"], res["c4b"]["summary"]
    c4d, c4e, c5 = res["c4d"], res["c4e"], res["c5"]
    prov = _provenance()
    L = []
    A = L.append
    A("# Hallazgos clave del EDA — MedXpertQA (one-pager)")
    A("")
    A(f"*TFG · Paso 1. Fuente: `{prov.get('repo_id', 'TsinghuaC3I/MedXpertQA')}` "
      f"rev. `{prov.get('pinned_revision', 'N/D')[:12]}…`. Alcance: test "
      f"(MM={len(dfs['MM'])}, Text={len(dfs['Text'])}).*")
    A("")
    A("## El dataset en cifras")
    A(f"- **{len(dfs['MM'])}** preguntas MM (5 opciones, con imagen) y **{len(dfs['Text'])}** "
      f"Text (10 opciones, sin imagen); opción única (1 respuesta correcta, siempre ∈ opciones).")
    A(f"- **{c4b['n_imagenes']}** imágenes MM, {c4b['footprint_mb']} MB; "
      f"formatos {', '.join(f'{k} {v}' for k, v in c4b['formatos'].items())}.")
    A("- **Integridad verificada** (recuentos, imagen↔archivo, 0 corruptas, IDs únicos); "
      "1 anomalía menor documentada (`Basic Medicine` en dev).")
    A("")
    A("## Hallazgos")
    A(f"1. **Sesgo temático multimodal.** MM está dominado por *Skeletal* "
      f"({bs.loc['Skeletal', 'MM_pct']} %) e *Integumentary* "
      f"({bs.loc['Integumentary', 'MM_pct']} % vs {bs.loc['Integumentary', 'Text_pct']} % en Text): "
      f"las especialidades visuales pesan más en MM.")
    A(f"2. **Predominio del razonamiento diagnóstico.** *Reasoning* y *Diagnosis* son mayoritarios "
      f"(cuadrante dominante Reasoning × Diagnosis).")
    A(f"3. **Sin sesgo posicional explotable.** Baseline de mayoría "
      f"{bl.loc['MM', 'baseline_mayoria']*100:.1f} % (MM) / {bl.loc['Text', 'baseline_mayoria']*100:.1f} % "
      f"(Text) frente a azar {bl.loc['MM', 'baseline_azar']*100:.0f} / "
      f"{bl.loc['Text', 'baseline_azar']*100:.0f} %; χ² no rechaza la uniformidad.")
    A(f"4. **Coste de contexto.** Prompt (enunciado+opciones): mediana "
      f"{summ.loc['MM', 'tok_cl100k_full_median']:.0f} (MM) / {summ.loc['Text', 'tok_cl100k_full_median']:.0f} "
      f"(Text) tokens; p95 hasta {summ.loc['Text', 'tok_cl100k_full_p95']:.0f} (Text), más tokens visuales.")
    A(f"5. **Componente visual exigente.** {c4a['pct_multi']:.1f} % de MM es multi-imagen "
      f"(hasta {c4a['max_images']}); ~{100*c4b['gris_n']/c4b['n_imagenes']:.0f} % en escala de grises. "
      f"Modalidad **no etiquetada** (limitación).")
    A(f"6. **Acoplamiento multimodal.** {c5['pct_explicit']:.1f} % de MM menciona la figura "
      f"explícitamente; MM **no** es resoluble de forma fiable solo con texto.")
    A(f"7. **Calidad del benchmark.** {c4d['n_exact_dup_groups']} grupos de imágenes idénticas → "
      f"{c4e['n_pairs']} pares de preguntas comparten imagen; **{c4e['n_near_dup_q']}** es una "
      f"pregunta casi-duplicada real (MM-121/MM-174).")
    A("")
    A("## Implicaciones para el diseño")
    A("- VLM con **soporte multi-imagen** (Qwen2.5-VL) o estrategia de mosaico definida.")
    A("- Dimensionar **ventana de contexto y coste** según las longitudes anteriores.")
    A("- Reportar **baselines de azar y mayoría** y *accuracy* **desagregada por `body_system`**.")
    A("- El margen de mejora está en el **razonamiento** → CoT y RAG como palancas.")
    A("")
    A("## Decisiones pendientes")
    A("- `dev.jsonl` como pool de few-shot (EDA sobre test). — *a confirmar*")
    A("- Tratamiento de preguntas multi-imagen en el pipeline. — *para el Paso 2*")
    A("- Inferencia de modalidad (heurística/clustering) vs anotación manual. — *a decidir*")
    A("")
    out = REPORT_DIR / "hallazgos_one_pager.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


# ============================================================================
# Orquestación
# ============================================================================
def run_all(dfs: dict | None = None) -> dict:
    setup_style()
    if dfs is None:
        dfs = load_all()
    res = {}
    print("[C.1] Metadatos y estructura ...")
    res["c1"] = c1_metadata_distributions(dfs)
    res["c1x"] = c1_crosstabs(dfs)
    res["c1o"] = c1_options_answers(dfs)
    print("[C.2] Sesgos de posición ...")
    res["c2"] = c2_label_bias(dfs)
    print("[C.3] Texto: longitudes ...")
    res["c3"] = c3_text_lengths(dfs)
    print("[C.3] Texto: casi-duplicados ...")
    res["c3d"] = c3_near_duplicates(dfs)
    print("[C.4] Imágenes: tabla de propiedades (puede tardar) ...")
    img_df = build_image_table(dfs)
    res["c4a"] = c4_images_per_question(dfs)
    res["c4b"] = c4_image_properties(img_df)
    res["c4c"] = c4_contact_sheet(dfs, img_df)
    res["c4d"] = c4_image_duplicates(img_df)
    res["c4e"] = c4_duplicate_question_pairs(dfs, img_df)
    print("[C.5] Acoplamiento texto-imagen ...")
    res["c5"] = c5_text_image_coupling(dfs)
    print("[informe] Generando 01_eda.md, data_card.md, índice y one-pager ...")
    build_report(res, dfs)
    build_data_card(res, dfs)
    build_figure_table_index()
    build_onepager(res, dfs)
    plt.close("all")
    return res


def main() -> None:
    matplotlib.use("Agg")  # backend headless para la ejecución por CLI
    print("=" * 70)
    print("TAREA C — Analisis Exploratorio de Datos (EDA)")
    print("=" * 70)
    dfs = load_all()
    run_all(dfs)
    print("-" * 70)
    print("EDA completado. Salidas en outputs/{figures,tables,report}.")
    print("=" * 70)


if __name__ == "__main__":
    main()
