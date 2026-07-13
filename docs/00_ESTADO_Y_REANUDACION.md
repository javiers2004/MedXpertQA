# Estado del proyecto y guía de reanudación — TFG MedXpertQA

> **Propósito de este documento**: permitir continuar el trabajo (y la conversación con el
> asistente) **en otro ordenador**, exactamente en el punto en que se dejó. Es el punto de
> entrada; los otros documentos de `docs/` amplían el plan y el *setup*.
>
> **Fecha del handoff**: 2026-07-10 · **Punto exacto**: Paso 1 (datos + EDA) **completado**;
> a punto de planificar el **Paso 2 (evaluación)**, con dos decisiones ya tomadas (ver §4).

---

## 1. Objetivo del TFG
Construir y evaluar un **sistema multimodal de *question-answering* médico** sobre el benchmark
**MedXpertQA** (ICML 2025), combinando un **VLM open-source** (objetivo: **Qwen2.5-VL**) con
**RAG**, **chain-of-thought** y, opcionalmente, **fine-tuning con QLoRA**. Grado de Ingeniería
Informática, mención en Ciencia de Datos e IA.

## 2. Estado actual — Paso 1 COMPLETADO
Datos descargados con procedencia trazable, integridad verificada y EDA completo y reproducible.

- **Código**: `src/{download,verify,eda,render_reports}.py` · **Notebook**: `notebooks/01_eda.ipynb`
  (ejecutado sin errores) · **Entorno**: `requirements.txt` (pip freeze) · **Guía**: `README.md`.
- **Salidas**: `outputs/figures/` (16 PNG a 300 dpi), `outputs/tables/` (22 CSV/JSON),
  `outputs/report/` (5 informes `.md` + sus `.html`).
- **Datos** (no versionados, ~1 GB): `data/raw/` (JSONL + `images.zip` + `PROVENANCE.json`),
  `data/processed/images/` (2.858 imágenes).

**Dataset**: HF `TsinghuaC3I/MedXpertQA` (MIT), **revisión fijada**
`7e7c465a68eb2b866926bfa59c8c9d17a8daba65`. test: **MM=2000** (5 opciones A–E, con imagen),
**Text=2450** (10 opciones A–J, sin imagen). dev=5+5 → **pool de few-shot**.

**Esquema real** (confirmado, no asumido): `options`=**dict**, `label`=**str** de 1 letra
(siempre ∈ options), imágenes en campo **`images`**=list de basenames (planos en
`data/processed/images/`).

## 3. Hallazgos clave del EDA (resumen autocontenido)
Detalle en [`outputs/report/hallazgos_one_pager.md`](../outputs/report/hallazgos_one_pager.md)
y [`outputs/report/01_eda.md`](../outputs/report/01_eda.md).

1. **Integridad**: recuentos 2000/2450 ✓, correspondencia imagen↔archivo exacta, **0 corruptas**,
   IDs únicos, `label`∈`options` al 100 %. Única anomalía: `Basic Medicine` en MM/dev (vs
   `Basic Science` en test); solo afecta few-shot.
2. **Sesgo temático**: MM cargado a *Skeletal* (24,6 %) e *Integumentary* (11,2 % vs 2,0 % en
   Text → dermatología visual); Text a *Nervous*. Dominante **Reasoning × Diagnosis**.
3. **Sin sesgo posicional**: baseline de **mayoría 21,2 % (MM, letra E) / 10,7 % (Text, letra E)**
   vs azar 20 / 10 %; χ² no rechaza uniformidad (p=0,59 / 0,71). *(Vara de medir del Paso 2.)*
4. **Coste de contexto**: prompt (enunciado+opciones) mediana **187 (MM) / 355 (Text)** tokens
   (cl100k), p95 hasta **589** (Text). Tokenizador Qwen2.5-VL también reportado.
5. **Visual (MM)**: **20,9 % multi-imagen** (hasta 6); ~60 % escala de grises; formatos JPEG 2304
   / PNG 548; **524 MB**. Modalidad de imagen **NO etiquetada** (limitación documentada).
6. **Acoplamiento**: 78,8 % de MM menciona la figura explícitamente; **MM no es resoluble solo
   con texto**.
7. **Calidad del benchmark**: 35 grupos de imágenes idénticas → 42 pares de preguntas comparten
   imagen; **1 pregunta casi-duplicada real** (MM-121/MM-174, coseno 0,972). Útil para control de
   fuga de datos (los hashes pHash+md5 están en `outputs/tables/c4_image_properties.csv`).

## 4. Decisiones ya tomadas
- **EDA sobre `test`**; `dev` reservado como few-shot.
- **Paso 2 — infraestructura**: se ejecutará en **GPU local** (ordenador más potente).
- **Paso 2 — alcance inicial**: **empezar por el subconjunto `Text`** (sin imágenes; más barato
  de depurar), y luego extender a MM.

## 5. Decisiones PENDIENTES (a resolver al reanudar)
- **[Paso 2 · bloqueante]** **VRAM de la GPU** del equipo nuevo → fija el tamaño de modelo:
  ≥24 GB → Qwen2.5-VL-**7B** bf16; ~10–16 GB → **7B en 4-bit** (bitsandbytes); <8 GB → **3B**.
- **[Paso 1 · abiertas]** tratamiento de **multi-imagen** en el pipeline (entra en el diseño del
  prompt del Paso 2); inferencia de **modalidad** (heurística/clustering) vs anotación manual
  (sub-análisis opcional para el capítulo de errores).

## 6. Siguiente paso
**Paso 2 — Harness de evaluación + baseline zero-shot** con Qwen2.5-VL sobre Text.
Plan detallado en → [`docs/01_PLAN_PASO_2.md`](01_PLAN_PASO_2.md).
Preparación del equipo nuevo en → [`docs/02_SETUP_NUEVO_EQUIPO.md`](02_SETUP_NUEVO_EQUIPO.md).

## 7. Cómo reanudar la conversación en el equipo nuevo
1. Copiar/clonar el proyecto y montar el entorno siguiendo
   [`02_SETUP_NUEVO_EQUIPO.md`](02_SETUP_NUEVO_EQUIPO.md).
2. Abrir el asistente (Claude Code) en la carpeta del proyecto y **pegar el siguiente prompt de
   reanudación**:

> Retomo mi TFG (sistema multimodal de QA médico sobre MedXpertQA). El **Paso 1 (datos + EDA)
> está terminado**; lee `docs/00_ESTADO_Y_REANUDACION.md`, `docs/01_PLAN_PASO_2.md` y el informe
> `outputs/report/01_eda.md` para el contexto completo. Estábamos a punto de arrancar el
> **Paso 2 (harness de evaluación + baseline zero-shot con Qwen2.5-VL)**. Decisiones tomadas:
> **GPU local** y **empezar por el subconjunto Text**. Mi GPU tiene **____ GB de VRAM**
> (modelo: ______). Preséntame el **plan detallado del Paso 2** para darte el visto bueno antes
> de programar, siguiendo mi forma de trabajo habitual (plan primero, checkpoints, documentar
> anomalías y consultarme las decisiones de diseño).

3. Rellenar la VRAM/GPU en el prompt y continuar.

## 8. Índice de documentos del handoff
| Documento | Contenido |
|---|---|
| `docs/00_ESTADO_Y_REANUDACION.md` | **Este archivo**: estado, hallazgos, decisiones y cómo reanudar. |
| `docs/01_PLAN_PASO_2.md` | Plan detallado del Paso 2 (harness de evaluación). |
| `docs/02_SETUP_NUEVO_EQUIPO.md` | Puesta en marcha del proyecto y el entorno en el equipo nuevo. |
| `README.md` | Reproducción del Paso 1 (descarga, verificación, EDA, render HTML). |
| `outputs/report/01_eda.md` | Informe EDA completo con figuras e implicaciones de diseño. |
| `outputs/report/indice_figuras_tablas.md` | Índice navegable de todas las figuras y tablas. |
