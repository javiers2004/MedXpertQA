# TFG MedXpertQA — Paso 1: Datos y Análisis Exploratorio (EDA)

Descarga reproducible, verificación de integridad y análisis exploratorio del benchmark
**MedXpertQA** (ICML 2025) como base para un sistema multimodal de *question-answering*
médico. Este repositorio cubre **exclusivamente el Paso 1**: obtener los datos y entenderlos
a fondo. No entrena ni evalúa modelos, ni usa APIs de pago.

- **Dataset**: [`TsinghuaC3I/MedXpertQA`](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA)
  (Hugging Face, licencia MIT).
- **Revisión fijada (trazabilidad)**: `7e7c465a68eb2b866926bfa59c8c9d17a8daba65`.
- **Alcance del EDA**: conjunto de *test* (MM = 2000, Text = 2450). El *dev* (5+5) se reserva
  como pool de *few-shot*.

## Requisitos

- **Windows** con **Python 3.12** (desarrollado con 3.12.4).
- ~1,5 GB de disco libre (datos ≈ 500 MB + imágenes extraídas ≈ 500 MB + entorno).
- Conexión a internet para la primera descarga (dataset y tokenizador de Qwen2.5-VL).

## Reproducción completa (Windows / CMD)

Desde la raíz del proyecto, en **CMD** (`cmd.exe`):

```bat
:: 1) Entorno virtual e instalación de dependencias (versiones fijadas)
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:: 2) TAREA A — Descarga + extracción de imágenes (idempotente)
python src\download.py

:: 3) TAREA B — Verificación de integridad -> outputs\report\00_integridad.md
python src\verify.py

:: 4) TAREA C — EDA: tablas, figuras e informes (01_eda.md, data_card.md)
python src\eda.py

:: 5) (opcional) Renderizar los informes .md a .html autocontenido (para ver/imprimir)
python src\render_reports.py
```

Los informes en `outputs\report\` están en Markdown. Para verlos con formato:
**en VS Code** abre el `.md` y pulsa `Ctrl+Shift+V` (vista previa); o ejecuta
`python src\render_reports.py` para obtener los `.html` equivalentes (figuras embebidas)
que se abren con doble clic en el navegador y se pueden **Imprimir → Guardar como PDF**.

### Notebook narrado (opcional, mismo resultado que `eda.py`)

El cuaderno `notebooks\01_eda.ipynb` invoca las mismas funciones de `src\eda.py` de forma
narrada, con figuras *inline*. Reutiliza las cachés generadas por `python src\eda.py`
(propiedades de imagen y longitudes de texto), por lo que corre en segundos.

```bat
:: Registrar el venv como kernel de Jupyter (una sola vez)
python -m ipykernel install --user --name medxpertqa --display-name "Python (MedXpertQA .venv)"

:: Opción A — abrir y ejecutar manualmente
jupyter notebook notebooks\01_eda.ipynb

:: Opción B — ejecutar de principio a fin sin abrirlo
jupyter nbconvert --to notebook --execute --inplace notebooks\01_eda.ipynb
```

## Estructura del proyecto

```
MedXpertQA/
├─ data/
│  ├─ raw/                 # descarga original intacta (JSONL + images.zip) + PROVENANCE.json
│  └─ processed/images/    # 2858 imágenes extraídas
├─ src/
│  ├─ download.py          # Tarea A: descarga + extracción + procedencia (idempotente)
│  ├─ verify.py            # Tarea B: verificación de integridad
│  ├─ eda.py               # Tarea C: funciones de EDA (C.1–C.5) + generación de informes
│  └─ render_reports.py    # (opcional) informes .md -> .html autocontenido para ver/imprimir
├─ notebooks/
│  └─ 01_eda.ipynb         # EDA narrado, ejecutado end-to-end
├─ outputs/
│  ├─ figures/             # PNG a 300 dpi (etiquetados en español)
│  ├─ tables/              # CSV/JSON con todas las estadísticas
│  └─ report/              # 00_integridad.md, 01_eda.md, data_card.md,
│                          #   indice_figuras_tablas.md, hallazgos_one_pager.md
├─ docs/                   # handoff: estado del proyecto, plan del Paso 2, setup del equipo nuevo
├─ requirements.txt        # versiones EXACTAS (pip freeze)
└─ README.md
```

> `data/` y `.venv/` no se versionan (ver `.gitignore`); se regeneran con los comandos de
> arriba. `outputs/` sí se versiona: es el entregable del análisis.

## Qué produce cada script

| Script | Salidas principales |
|---|---|
| `download.py` | `data/raw/` (JSONL + `images.zip`), `data/processed/images/`, `data/raw/PROVENANCE.json` (repo, revisión, hashes SHA-256, recuentos). |
| `verify.py` | `outputs/report/00_integridad.md` (recuentos, correspondencia imagen↔archivo, integridad Pillow, unicidad de IDs, campos obligatorios). |
| `eda.py` | `outputs/tables/*.csv`+`*.json`, `outputs/figures/*.png`, y en `outputs/report/`: `01_eda.md`, `data_card.md`, `indice_figuras_tablas.md` (índice navegable) y `hallazgos_one_pager.md` (resumen ejecutivo). |

## Trazabilidad y reproducibilidad

- **Procedencia** del dato registrada en `data/raw/PROVENANCE.json` (revisión exacta de HF,
  fecha, tamaños, hashes SHA-256 y recuentos).
- **Versiones fijadas** en `requirements.txt` (`pip freeze` completo).
- **Semilla** fija (`SEED = 42`) en todo muestreo aleatorio (p. ej. la hoja de contactos).
- Cada cifra de los informes procede de una función ejecutada de `src/eda.py`; toda figura y
  tabla se regenera reejecutando el pipeline.

## Continuar en otro equipo / próximos pasos

El Paso 1 está terminado. El **Paso 2 (evaluación con un VLM)** se hará en un equipo con GPU.
Toda la documentación de *handoff* para retomar el trabajo (y la conversación) exactamente en
este punto está en **`docs/`**:
- [`docs/00_ESTADO_Y_REANUDACION.md`](docs/00_ESTADO_Y_REANUDACION.md) — estado, hallazgos,
  decisiones y **prompt de reanudación** para el asistente.
- [`docs/01_PLAN_PASO_2.md`](docs/01_PLAN_PASO_2.md) — plan detallado del harness de evaluación.
- [`docs/02_SETUP_NUEVO_EQUIPO.md`](docs/02_SETUP_NUEVO_EQUIPO.md) — puesta en marcha con GPU.

## Notas y limitaciones conocidas

- El dataset **no incluye etiqueta de modalidad de imagen** (radiografía/TAC/RM/histología/
  fotografía clínica/ECG…). No se infiere ni se inventa; se documenta como limitación y se
  aporta una hoja de contactos estratificada para anotación manual.
- Anomalía menor (documentada en `00_integridad.md`): `MM/dev` usa `Basic Medicine` donde el
  *test* usa `Basic Science`. Solo afecta a *few-shot*.
- Sobre OneDrive: el proyecto reside en una carpeta sincronizada; el entorno `.venv` y `data/`
  están excluidos del versionado para evitar sincronizar miles de ficheros.
