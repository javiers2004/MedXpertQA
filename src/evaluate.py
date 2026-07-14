"""
src/evaluate.py — Paso 2 (harness de evaluación): baseline zero-shot con Qwen2.5-VL.

Ejecuta inferencia determinista sobre un subconjunto de MedXpertQA (**Text**,
sin imágenes, o **MM**, con 1–6 imágenes nativas por pregunta), parsea la letra
de respuesta y calcula *accuracy* global y desagregada por `body_system` /
`medical_task` / `question_type` (y, en MM, por nº de imágenes), comparada con
las baselines de azar/mayoría de `outputs/tables/c2_baselines.csv` (Paso 1).

Reutiliza la carga de datos de `src/eda.py` (misma fuente de verdad que el EDA).

Modelo: Qwen2.5-VL-7B-Instruct en 4-bit (bitsandbytes NF4). Si no cabe en VRAM
(CUDA OOM), cae automáticamente a Qwen2.5-VL-3B-Instruct (también 4-bit) y lo
deja registrado como anomalía en `metrics.json` (campo `cayo_a_3b_por_oom`).
En MM, las imágenes se pasan de forma **nativa** (varias por prompt, sin
mosaico) tal como soporta Qwen2.5-VL, con un tope de resolución por imagen
(`MM_MAX_PIXELS`) para controlar la VRAM en preguntas multi-imagen.

Modo **CoT** (`--mode cot`, Paso 3): pide razonamiento breve terminado en la
línea ancla "Respuesta final: <letra>" (`MAX_NEW_TOKENS_COT` tokens, ~14
tok/s medidos en RTX 3070 -> ~18s/pregunta). Por coste (2450 preguntas x CoT
~= 12h), se recomienda pilotar con `--sample N` (muestra estratificada por
question_type x medical_task, semilla fija) antes de la ejecución completa.

Self-consistency (`--sc N`, solo con `--mode cot`): N muestras con
`do_sample=True` (`SC_TEMPERATURE`/`SC_TOP_P`) por pregunta en vez de una
greedy; se parsea cada una con `parse_answer_cot` y se vota por mayoría entre
las parseables (empate -> gana la primera muestra, orden determinista dada la
semilla). Multiplica el coste por N.

Uso (Windows/CMD, venv activado, requirements-eval.txt instalado):
    python src\\evaluate.py --limit 50               # sanity check Text (recomendado primero)
    python src\\evaluate.py                          # ejecución completa Text (2450)
    python src\\evaluate.py --subset mm --limit 50   # sanity check MM
    python src\\evaluate.py --subset mm              # ejecución completa MM (2000)
    python src\\evaluate.py --model 3b               # forzar el modelo 3B
    python src\\evaluate.py --mode cot --sample 300  # piloto CoT (Paso 3) sobre 300 preguntas
    python src\\evaluate.py --mode cot --sample 300 --sc 5  # + self-consistency (5 muestras)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda import (  # noqa: E402  (reutiliza carga y rutas del Paso 1)
    C_MM, C_TEXT, IMAGES_DIR, INK, INK2, MUTED, PALETTE, PROJECT_ROOT, TAB_DIR,
    _annotate_bars, load_subset, savefig, setup_style,
)

# --- Reproducibilidad ------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)

# --- Rutas -------------------------------------------------------------------
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# --- Modelo y generación -----------------------------------------------------
MODEL_IDS = {"7b": "Qwen/Qwen2.5-VL-7B-Instruct", "3b": "Qwen/Qwen2.5-VL-3B-Instruct"}
MAX_NEW_TOKENS = 8
MAX_NEW_TOKENS_COT = 256  # ~18s/pregunta medidos en RTX 3070 (14 tok/s)
SC_TEMPERATURE = 0.7  # self-consistency (Wang et al. 2022): sampling, no greedy
SC_TOP_P = 0.9
# Tope de resolución por imagen (múltiplo de 28x28, patch size de Qwen2.5-VL).
# Conservador a propósito: hasta 6 imágenes por pregunta MM en 8GB de VRAM.
MM_MIN_PIXELS = 256 * 28 * 28
MM_MAX_PIXELS = 640 * 28 * 28
SYSTEM_PROMPTS = {
    "direct": (
        "Actúa como examinador médico. Se te presentará una pregunta de opción múltiple "
        "(puede incluir una o varias imágenes clínicas) y debes responder ÚNICAMENTE con "
        "la letra de la opción correcta (por ejemplo: A). No incluyas explicación ni texto adicional."
    ),
    "cot": (
        "Act as a medical examiner. You will be given a multiple-choice question "
        "(it may include one or more clinical images). Reason briefly in AT MOST 3 short "
        "sentences of plain prose (no numbered lists, no headers, no markdown formatting), "
        "focusing only on the key finding(s) that decide the answer. Then end your response "
        "with EXACTLY one line of the form 'Final answer: <letter>' (e.g. Final answer: A) "
        "and nothing after it."
    ),
}

SUBSETS = {
    "text": {"jsonl_name": "Text", "n_options": 10, "has_images": False},
    "mm": {"jsonl_name": "MM", "n_options": 5, "has_images": True},
}

# Prioridad: "Answer: X" / "Respuesta: X" > letra aislada al inicio > primera letra A-J suelta.
ANSWER_PATTERNS = [
    re.compile(r"(?:answer|respuesta)\s*(?:is|es)?\s*:?\s*\(?([A-J])\)?", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-J])\)?[\.\:\)]", re.IGNORECASE),
    re.compile(r"\b([A-J])\b"),
]
# CoT: prioriza la línea ancla "Respuesta final: X" / "Final answer: X"; si hay varias
# (el modelo divaga y se corrige), se queda con la ÚLTIMA — es la conclusión real.
COT_ANCHOR_PATTERN = re.compile(
    r"(?:respuesta\s*final|final\s*answer)\s*:?\s*\(?([A-J])\)?", re.IGNORECASE)


def build_messages(row, has_images: bool, mode: str = "direct") -> list[dict]:
    opt_txt = "\n".join(f"{k}. {v}" for k, v in row["options"].items())
    text = f"{row['question']}\n\n{opt_txt}\n\nRespuesta:"
    content = []
    if has_images:
        for img_name in row.get("images") or []:
            content.append({
                "type": "image", "image": str(IMAGES_DIR / Path(img_name).name),
                "min_pixels": MM_MIN_PIXELS, "max_pixels": MM_MAX_PIXELS,
            })
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[mode]},
        {"role": "user", "content": content},
    ]


def parse_answer(text: str, valid_letters: list[str]) -> str | None:
    text = text.strip()
    for pat in ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            letter = m.group(1).upper()
            if letter in valid_letters:
                return letter
    return None


def parse_answer_cot(text: str, valid_letters: list[str]) -> str | None:
    """SOLO acepta la línea ancla 'Respuesta final: X' / 'Final answer: X'
    (última aparición, por si el modelo divaga y se corrige). Deliberadamente
    NO cae a un parseo genérico sobre el razonamiento: una letra mencionada
    de pasada al descartar una opción ("no es H, es...") no es la respuesta,
    y contarla como tal inflaría la accuracy de forma espuria. Si el modelo
    se queda sin tokens (MAX_NEW_TOKENS_COT) antes de concluir, se cuenta
    honestamente como no parseable."""
    matches = list(COT_ANCHOR_PATTERN.finditer(text))
    if matches:
        letter = matches[-1].group(1).upper()
        if letter in valid_letters:
            return letter
    return None


def stratified_sample(df, n: int, seed: int = SEED):
    """Muestra estratificada por (question_type, medical_task), proporcional
    al tamaño de cada estrato, con semilla fija para reproducibilidad."""
    import pandas as pd
    frac = n / len(df)
    parts = [g.sample(frac=frac, random_state=seed)
             for _, g in df.groupby(["question_type", "medical_task"])]
    return pd.concat(parts).reset_index(drop=True)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * ((phat * (1 - phat) / n + z**2 / (4 * n**2)) ** 0.5)
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


# --- Carga del modelo ---------------------------------------------------------
def load_model(size: str):
    model_id = MODEL_IDS[size]
    print(f"  Cargando {model_id} en 4-bit (bitsandbytes NF4) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="cuda:0", dtype=torch.bfloat16,
    )
    model.eval()
    return model, processor, model_id


def load_model_with_fallback(preferred: str):
    order = ["7b", "3b"] if preferred == "7b" else ["3b"]
    last_exc = None
    for i, size in enumerate(order):
        try:
            model, processor, model_id = load_model(size)
            return model, processor, model_id, size, i > 0
        except torch.cuda.OutOfMemoryError as exc:
            last_exc = exc
            print(f"  [OOM] {MODEL_IDS[size]} no cabe en la GPU. Liberando memoria "
                  f"y probando el siguiente tamaño ...")
            torch.cuda.empty_cache()
    raise RuntimeError("No se pudo cargar ningún tamaño de modelo (OOM incluso en 3B).") from last_exc


@torch.inference_mode()
def generate_answer(model, processor, messages: list[dict], max_new_tokens: int = MAX_NEW_TOKENS,
                     do_sample: bool = False) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt").to(model.device)
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs.update(temperature=SC_TEMPERATURE, top_p=SC_TOP_P)
    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


def majority_vote(letras: list[str | None]) -> tuple[str | None, int, int]:
    """Vota por mayoría entre las letras parseables. Empate -> gana la que
    aparezca antes (Counter conserva el orden de inserción en Python 3.7+,
    determinista dada la semilla). Devuelve (letra_ganadora, votos_ganadora, n_parseables)."""
    from collections import Counter
    valid = [l for l in letras if l is not None]
    if not valid:
        return None, 0, 0
    counts = Counter(valid)
    winner, votes = counts.most_common(1)[0]
    return winner, votes, len(valid)


# --- Métricas y figuras --------------------------------------------------------
def compute_disaggregated(records: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r[field], []).append(r)
    out = {}
    for key, rs in groups.items():
        n = len(rs)
        k = sum(r["acierto"] for r in rs)
        out[key] = {"n": n, "n_correctas": k, "accuracy": k / n if n else 0.0}
    return out


def make_disaggregated_figure(disagg: dict, field: str, baseline_mayoria: float,
                               baseline_azar: float, tag: str, subset_label: str) -> Path:
    setup_style()
    import matplotlib.pyplot as plt
    import numpy as np
    cats = sorted(disagg, key=lambda c: -disagg[c]["n"])
    accs = [100 * disagg[c]["accuracy"] for c in cats]
    ns = [disagg[c]["n"] for c in cats]
    fig, ax = plt.subplots(figsize=(8.2, 0.5 * len(cats) + 1.6))
    y = np.arange(len(cats))[::-1]
    bars = ax.barh(y, accs, color=C_TEXT, height=0.6)
    for yi, n in zip(y, ns):
        ax.text(1, yi, f"n={n}", va="center", ha="left", fontsize=7.5, color="white")
    ax.axvline(100 * baseline_mayoria, ls="--", lw=1.4, color=INK2,
               label=f"Baseline mayoría = {100*baseline_mayoria:.1f}%")
    ax.axvline(100 * baseline_azar, ls=":", lw=1.4, color=MUTED,
               label=f"Baseline azar = {100*baseline_azar:.1f}%")
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(f"Accuracy por {field} — {tag} ({subset_label})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    return savefig(fig, f"c6_eval_{tag}_{subset_label.lower()}_by_{field}")


# --- Ejecución principal ------------------------------------------------------
def run(limit: int | None, model_pref: str, subset: str, mode: str = "direct",
        sample: int | None = None, sc: int | None = None) -> None:
    meta = SUBSETS[subset]
    jsonl_name = meta["jsonl_name"]
    has_images = meta["has_images"]
    df = load_subset(jsonl_name, "test")
    if sample:
        df = stratified_sample(df, sample)
    elif limit:
        df = df.head(limit)
    max_new_tokens = MAX_NEW_TOKENS_COT if mode == "cot" else MAX_NEW_TOKENS
    answer_parser = parse_answer_cot if mode == "cot" else parse_answer
    sc = sc if (sc and mode == "cot") else None
    modo_label = f"{mode}+sc{sc}" if sc else mode
    print(f"Evaluando {len(df)} preguntas de {jsonl_name}/test (modo={modo_label}) ...")

    model, processor, model_id, actual_size, fell_back = load_model_with_fallback(model_pref)
    if fell_back:
        print(f"  [ANOMALÍA] Se solicitó '{model_pref}' pero se usó '{actual_size}' por OOM.")

    records = []
    n_correct = n_unparsed = 0
    t0 = time.time()
    for _, row in df.iterrows():
        messages = build_messages(row, has_images, mode)
        valid_letters = list(row["options"].keys())
        if sc:
            raws = [generate_answer(model, processor, messages, max_new_tokens, do_sample=True)
                    for _ in range(sc)]
            letras_muestreadas = [answer_parser(r, valid_letters) for r in raws]
            letra, votos, n_validas = majority_vote(letras_muestreadas)
            raw = raws  # se persisten todas las muestras, no solo una
        else:
            raw = generate_answer(model, processor, messages, max_new_tokens)
            letra = answer_parser(raw, valid_letters)
        acierto = letra is not None and letra == row["label"]
        n_unparsed += letra is None
        n_correct += acierto
        prompt_hash = hashlib.md5(json.dumps(messages, sort_keys=True).encode("utf-8")).hexdigest()
        rec = {
            "id": row["id"], "prompt_hash": prompt_hash,
            "salida_cruda": raw if not sc else raws[0],
            "letra": letra, "label": row["label"], "acierto": acierto,
            "body_system": row["body_system"], "medical_task": row["medical_task"],
            "question_type": row["question_type"],
        }
        if sc:
            rec["muestras_cruda"] = raws
            rec["letras_muestreadas"] = letras_muestreadas
            rec["votos_ganadora"] = votos
            rec["n_muestras_parseables"] = n_validas
        if has_images:
            rec["n_images"] = len(row.get("images") or [])
        records.append(rec)
        n = len(records)
        if n % 50 == 0 or n == len(df):
            elapsed = time.time() - t0
            print(f"  [{n}/{len(df)}] accuracy parcial={n_correct/n:.3f} "
                  f"no_parseables={n_unparsed} ({elapsed:.0f}s, {elapsed/n:.2f}s/pregunta)")

    tag = f"qwen2.5-vl-{actual_size}-4bit"
    mode_suffix = "" if mode == "direct" else f"_{mode}"
    sc_suffix = f"_sc{sc}" if sc else ""
    sample_suffix = f"_sample{len(df)}" if sample else ""
    suffix = f"_sanity{limit}" if limit and not sample else ""
    suffix = f"{mode_suffix}{sc_suffix}{sample_suffix}{suffix}"
    out_jsonl = EVAL_DIR / f"{jsonl_name.lower()}_{tag}{suffix}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    baselines_path = TAB_DIR / "c2_baselines.csv"
    import pandas as pd
    baselines = pd.read_csv(baselines_path).set_index("subconjunto")
    baseline_azar = float(baselines.loc[jsonl_name, "baseline_azar"])
    baseline_mayoria = float(baselines.loc[jsonl_name, "baseline_mayoria"])

    n_total = len(records)
    acc_global = n_correct / n_total
    ci_lo, ci_hi = wilson_ci(n_correct, n_total)

    # Comparación pareada con el baseline directo (mismo modelo/preguntas) si existe,
    # para no comparar contra el nº global (2450/2000) con distinta varianza muestral.
    paired = None
    if mode != "direct":
        direct_path = EVAL_DIR / f"{jsonl_name.lower()}_{tag}.jsonl"
        if direct_path.exists():
            direct_by_id = {}
            with direct_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    d = json.loads(line)
                    direct_by_id[d["id"]] = d["acierto"]
            matched = [(r["acierto"], direct_by_id[r["id"]]) for r in records if r["id"] in direct_by_id]
            if matched:
                n_m = len(matched)
                k_new = sum(a for a, _ in matched)
                k_direct = sum(b for _, b in matched)
                paired = {
                    "n_parejas": n_m,
                    f"accuracy_{mode}": round(k_new / n_m, 4),
                    "accuracy_direct_mismas_preguntas": round(k_direct / n_m, 4),
                    "delta": round((k_new - k_direct) / n_m, 4),
                }

    disagg_fields = ["body_system", "medical_task", "question_type"]
    if has_images:
        disagg_fields.append("n_images")
    disagg = {field: compute_disaggregated(records, field) for field in disagg_fields}
    fig_tag = f"{tag}{mode_suffix}{sc_suffix}{sample_suffix}"  # incluye modo/muestra: nombres únicos
    if not limit:
        for field in ("body_system", "medical_task", "question_type"):
            make_disaggregated_figure(disagg[field], field, baseline_mayoria, baseline_azar,
                                       fig_tag, jsonl_name)

    prov = json.loads((RAW_DIR / "PROVENANCE.json").read_text(encoding="utf-8")) \
        if (RAW_DIR / "PROVENANCE.json").exists() else {}
    import transformers as _tf
    metrics = {
        "modo": mode,
        "muestra_estratificada": sample,
        "modelo_solicitado": MODEL_IDS[model_pref],
        "modelo_usado": model_id,
        "tamano_efectivo": actual_size,
        "cayo_a_3b_por_oom": fell_back,
        "cuantizacion": "4-bit NF4 (bitsandbytes), compute_dtype=bfloat16",
        "n_evaluado": n_total,
        "n_no_parseables": n_unparsed,
        "accuracy_global": round(acc_global, 4),
        "accuracy_global_ic95_wilson": [round(ci_lo, 4), round(ci_hi, 4)],
        "baseline_azar": baseline_azar,
        "baseline_mayoria": baseline_mayoria,
        "supera_baseline_mayoria": acc_global > baseline_mayoria,
        "comparacion_pareada_vs_direct": paired,
        "desagregado": disagg,
        "generacion": {
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(sc), "estrategia": "greedy" if not sc else "self-consistency",
            "self_consistency_n": sc,
            **({"temperature": SC_TEMPERATURE, "top_p": SC_TOP_P} if sc else {}),
        },
        "entorno": {
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "transformers": _tf.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        },
        "semilla": SEED,
        "revision_dataset": prov.get("pinned_revision"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tiempo_total_s": round(time.time() - t0, 1),
    }
    metrics_path = EVAL_DIR / f"metrics_{jsonl_name.lower()}_{tag}{suffix}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("-" * 70)
    print(f"RESUMEN — {jsonl_name}/{tag}{suffix}")
    print(f"  accuracy_global = {acc_global:.4f}  (IC95 Wilson: [{ci_lo:.4f}, {ci_hi:.4f}])")
    print(f"  baseline_mayoria = {baseline_mayoria:.4f}  -> "
          f"{'SUPERADA' if acc_global > baseline_mayoria else 'NO SUPERADA'}")
    if paired:
        print(f"  vs. direct (mismas {paired['n_parejas']} preguntas): "
              f"{paired['accuracy_direct_mismas_preguntas']:.4f} -> {paired[f'accuracy_{mode}']:.4f} "
              f"(delta={paired['delta']:+.4f})")
    print(f"  no_parseables = {n_unparsed}/{n_total}")
    print(f"  predicciones: {out_jsonl.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"  métricas:     {metrics_path.relative_to(PROJECT_ROOT).as_posix()}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                         help="Limita el nº de preguntas (sanity check, p.ej. 50).")
    parser.add_argument("--model", choices=["7b", "3b"], default="7b",
                         help="Tamaño de modelo preferido (cae a 3b si hay OOM).")
    parser.add_argument("--subset", choices=["text", "mm"], default="text",
                         help="Subconjunto a evaluar: text (sin imágenes) o mm (con imágenes).")
    parser.add_argument("--mode", choices=["direct", "cot"], default="direct",
                         help="direct = respuesta directa (Paso 2); cot = razonamiento paso a paso (Paso 3).")
    parser.add_argument("--sample", type=int, default=None,
                         help="Muestra estratificada de N preguntas (question_type x medical_task), "
                              "semilla fija. Recomendado para pilotar --mode cot antes de la ejecución completa.")
    parser.add_argument("--sc", type=int, default=None,
                         help="Self-consistency: nº de muestras con sampling + voto mayoritario "
                              "(solo con --mode cot). Multiplica el coste por N.")
    args = parser.parse_args()
    run(limit=args.limit, model_pref=args.model, subset=args.subset, mode=args.mode,
        sample=args.sample, sc=args.sc)


if __name__ == "__main__":
    main()
