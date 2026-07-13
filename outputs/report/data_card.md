# Data Card — MedXpertQA (subconjunto de test)

| Campo | Valor |
|---|---|
| Nombre | MedXpertQA (ICML 2025) |
| Repositorio | `TsinghuaC3I/MedXpertQA` (Hugging Face, dataset) |
| Revisión (commit) | `7e7c465a68eb2b866926bfa59c8c9d17a8daba65` |
| Licencia | MIT |
| Descargado | 2026-07-10T08:41:25.783261+00:00 |
| Splits | MM/test=2000, Text/test=2450, dev=5+5 (few-shot) |
| Formato preguntas | opción única; MM 5 opciones (A–E), Text 10 (A–J) |
| Metadatos | body_system (12), medical_task (3), question_type (2) |
| Imágenes (MM) | 2852 ficheros, 524.2 MB; formatos: JPEG=2304, PNG=548 |
| Color / gris (proxy) | 1144 color, 1708 gris |
| Multi-imagen | 20.9 % de MM (hasta 6) |
| Modalidad de imagen | **no etiquetada** en el dataset (limitación) |

## Distribuciones principales (test)

**body_system** (Top-5 por total):

| Sistema | MM (n / %) | Text (n / %) |
|---|---|---|
| Skeletal | 491 / 24.55% | 355 / 14.49% |
| Cardiovascular | 320 / 16.0% | 306 / 12.49% |
| Nervous | 231 / 11.55% | 386 / 15.76% |
| Digestive | 168 / 8.4% | 274 / 11.18% |
| Respiratory | 192 / 9.6% | 193 / 7.88% |

## Procedencia y reproducción

- Descarga: `python src/download.py` (revisión fijada; provenance en `data/raw/PROVENANCE.json`).
- Integridad: `python src/verify.py` → `outputs/report/00_integridad.md`.
- EDA: `python src/eda.py` o `notebooks/01_eda.ipynb` → tablas, figuras y `01_eda.md`.

## Uso previsto y limitaciones

- Benchmark de **evaluación** (no de entrenamiento) de QA médico experto.
- El subconjunto MM requiere la imagen para responder de forma fiable (no resoluble solo con texto).
- Sin etiqueta de modalidad de imagen; distribución temática sesgada hacia algunas especialidades.

