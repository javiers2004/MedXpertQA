# Plan del Paso 2 — Harness de evaluación + baseline *zero-shot*

> Estado: **propuesto, pendiente de visto bueno**. Requiere fijar la VRAM de la GPU (§3).
> Contexto y estado global en [`00_ESTADO_Y_REANUDACION.md`](00_ESTADO_Y_REANUDACION.md).

## 1. Objetivo y motivación
Construir un **pipeline de evaluación reproducible** y medir un **baseline zero-shot** de un VLM
open-source sobre MedXpertQA. Es el cimiento del TFG: CoT, RAG y QLoRA se justificarán como
**mejora (delta) sobre este baseline**. Sin un harness y un número de referencia no se puede
demostrar aportación.

**Fuera de alcance en el Paso 2**: CoT, RAG, fine-tuning. Solo inferencia directa (zero-shot) y
medición.

## 2. Qué nos dejó listo el EDA (aprovechar, no reinventar)
- **Opción única** con `label ∈ options` → *accuracy* directa parseando la letra. Métrica simple.
- **Baselines a superar** (de C.2, `outputs/tables/c2_baselines.csv`): azar 20 %/10 %, **mayoría
  21,2 % (MM) / 10,7 % (Text)**. El modelo debe batirlas con holgura para ser informativo.
- **Multi-imagen** 20,9 % en MM (hasta 6) → el harness de MM debe pasar varias imágenes por prompt
  (Qwen2.5-VL lo soporta de forma nativa). *(Decisión de composición aún abierta.)*
- **Presupuesto de contexto** holgado: mediana ~187 (MM) / 355 (Text) tokens + tokens visuales.
- **Ejes de análisis** ya presentes en cada registro: `body_system`, `medical_task`,
  `question_type` → reportar *accuracy* **desagregada** (aportación analítica central del TFG).

## 3. Decisiones
- **Infra**: GPU local (equipo potente). — *tomada*
- **Alcance inicial**: **Text primero** (2.450 preguntas, sin imágenes), luego MM. — *tomada*
- **Modelo**: **Qwen2.5-VL-Instruct**. Tamaño según VRAM — **PENDIENTE**:
  - ≥24 GB → **7B** en bf16 (recomendado si cabe).
  - ~10–16 GB → **7B en 4-bit** (bitsandbytes / `load_in_4bit`).
  - <8 GB → **3B**.
  El tokenizador de este modelo **ya se descargó** en el Paso 1 (usado en C.3).

## 4. Entorno (a añadir; NO tocar el de EDA)
El `.venv` del Paso 1 tiene `transformers` **sin** PyTorch. Para inferencia:
- Nuevo fichero **`requirements-eval.txt`** con: `torch` (build con CUDA acorde a la GPU),
  `accelerate`, `bitsandbytes` (para 4-bit; usar wheel reciente con soporte Windows),
  `qwen-vl-utils` (procesado de imágenes para la fase MM). `transformers` 4.57.6 ya soporta
  Qwen2.5-VL.
- Registrar la versión de CUDA y de torch en el informe (trazabilidad).

## 5. Diseño del harness (subconjunto Text primero)
Nuevo módulo **`src/evaluate.py`** (reutilizable), con:
1. **Carga**: `Text/test.jsonl` (2.450) vía las utilidades ya existentes en `src/eda.py`.
2. **Construcción del prompt**: *system* ("actúa como examinador médico; responde únicamente con
   la letra de la opción correcta") + enunciado + opciones `A–J` formateadas. Plantilla de chat de
   Qwen (`apply_chat_template`).
3. **Inferencia determinista**: greedy / `temperature=0`, `max_new_tokens` corto. Registrar la
   configuración de generación.
4. **Parseo robusto de la respuesta**: regex para letra aislada `A–J` o patrones tipo
   `Answer: X` / `(X)`; contabilizar respuestas no parseables como fallo y **guardarlas** para
   inspección.
5. **Métricas**: *accuracy* global **y desagregada** por `body_system` / `medical_task` /
   `question_type`; comparación con las baselines de C.2 (intervalo de confianza por Wilson opcional).
6. **Persistencia** (reproducibilidad y análisis de errores): `outputs/eval/text_<modelo>.jsonl`
   con `id, prompt_hash, salida_cruda, letra, correcta, acierto` + metadatos; `metrics.json` con
   los agregados; semilla y `PROVENANCE`-eval (modelo, revisión HF, dtype, CUDA/torch).
7. **Sanidad primero**: correr sobre **~50 preguntas** para depurar prompt/parseo antes de las 2.450.

## 6. Entregables previstos del Paso 2
- `src/evaluate.py`, `requirements-eval.txt`.
- `outputs/eval/` (predicciones JSONL + `metrics.json`).
- Figuras: *accuracy* desagregada por los tres ejes (comparada con baselines).
- Informe `outputs/report/02_baseline.md` (setup, modelo, resultados, comparación con baselines,
  primeros errores observados) + su `.html` con `render_reports.py`.

## 7. Verificación (cómo sé que funciona)
- Sanidad de 50 preguntas sin errores de parseo groseros.
- Ejecución completa sobre Text; *accuracy* global **> baseline de mayoría (10,7 %)** —si no,
  revisar prompt/parseo antes de sacar conclusiones.
- Métricas desagregadas coherentes (suman al global; nº de preguntas por celda cuadra con C.1).

## 8. Después del baseline (roadmap)
1. **Extender a MM**: añadir imágenes al prompt; resolver la composición multi-imagen.
2. **Paso 3 — CoT**: zero-shot CoT y *self-consistency*.
3. **Paso 4 — RAG**: corpus médico + *retriever* sobre el texto.
4. **Paso 5 — QLoRA** (opcional): usar los hashes del EDA para control de contaminación.
5. **Paso 6 — Análisis de errores** por ejes de metadatos + ablaciones + redacción.
