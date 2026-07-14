# Informe de Integridad — MedXpertQA (Paso 1 · Tarea B)

- **Generado**: 2026-07-13T11:41:05.333442+00:00 (UTC)
- **Repositorio**: `TsinghuaC3I/MedXpertQA` (revisión `7e7c465a68eb2b866926bfa59c8c9d17a8daba65`)
- **Script**: `src/verify.py` (solo lectura; no corrige anomalías)

> Este informe verifica la fiabilidad del dato antes del análisis. Las anomalías se **documentan y marcan**, no se corrigen automáticamente.

> **VEREDICTO: 1 anomalía(s) detectada(s)** — ver sección final. El dato es utilizable pero requiere tu decisión sobre los puntos marcados.

## 1. Recuentos de registros

| Fichero | Registros | Esperado | Estado |
|---|---:|---:|:--:|
| `MM/test.jsonl` | 2000 | 2000 | OK |
| `MM/dev.jsonl` | 5 | — | OK |
| `Text/test.jsonl` | 2450 | 2450 | OK |
| `Text/dev.jsonl` | 5 | — | OK |

Los `dev.jsonl` (few-shot) se reportan como informativos: no hay recuento canónico esperado más allá de que sean no vacíos.

## 2. Referencias imagen ↔ archivos (solo MM)

- Imágenes en disco (`data/processed/images/`): **2858**
- Referencias desde MM/test: **2852** (2852 únicas)
- Referencias desde MM/dev: **6**
- Referencias únicas (test ∪ dev): **2858**
- Faltantes en disco referenciadas por **test**: **0**
- Faltantes en disco referenciadas por **dev**: **0**
- Huérfanas por exceso (en disco, sin referencia en test ∪ dev): **0**
- Imágenes referenciadas por más de una pregunta: **0**

Correspondencia **exacta**: cada una de las 2852 referencias de test y las 6 de dev tiene su fichero, y los 2858 ficheros están todos referenciados (2852 test + 6 dev = 2858).

## 3. Integridad de las imágenes (Pillow `verify()`)

- Imágenes verificadas: **2858**
- Corruptas / no decodificables: **0**

Todas las imágenes superan la verificación estructural de Pillow.

## 4. Unicidad de identificadores (`id`)

| Subconjunto | Registros | IDs únicos | Duplicados |
|---|---:|---:|---:|
| `MM/test` | 2000 | 2000 | 0 |
| `MM/dev` | 5 | 5 | 0 |
| `Text/test` | 2450 | 2450 | 0 |
| `Text/dev` | 5 | 5 | 0 |

- Solapamiento de IDs test↔dev en MM: **0** (ninguno)
- Solapamiento de IDs test↔dev en Text: **0** (ninguno)

## 5. Campos obligatorios y coherencia

Campos obligatorios exigidos: `id`, `question`, `options`, `label`, `body_system`, `medical_task`, `question_type` (y `images` no vacío en MM).

| Subconjunto | Sin campos oblig. | label∉options | nº opciones incorrecto | MM sin imágenes |
|---|---:|---:|---:|---:|
| `MM/test` | 0 | 0 | 0 | 0 |
| `MM/dev` | 0 | 0 | 0 | 0 |
| `Text/test` | 0 | 0 | 0 | — |
| `Text/dev` | 0 | 0 | 0 | — |

### 5b. Coherencia de vocabulario de metadatos (dev ↔ test)

| Modalidad · campo | Valores en test | Valores solo en dev |
|---|---|---|
| `MM` · `body_system` | Cardiovascular, Digestive, Endocrine, Integumentary, Lymphatic, Muscular, Nervous, Other / NA, Reproductive, Respiratory, Skeletal, Urinary | — |
| `MM` · `medical_task` | Basic Science, Diagnosis, Treatment | Basic Medicine |
| `MM` · `question_type` | Reasoning, Understanding | — |
| `Text` · `body_system` | Cardiovascular, Digestive, Endocrine, Integumentary, Lymphatic, Muscular, Nervous, Other / NA, Reproductive, Respiratory, Skeletal, Urinary | — |
| `Text` · `medical_task` | Basic Science, Diagnosis, Treatment | — |
| `Text` · `question_type` | Reasoning, Understanding | — |

## Anomalías detectadas (marcadas para decisión, NO corregidas)

- ⚠ MM: `medical_task` tiene valores en dev ausentes en test: ['Basic Medicine'].

