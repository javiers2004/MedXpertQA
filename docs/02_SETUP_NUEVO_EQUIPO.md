# Puesta en marcha en el equipo nuevo (más potente)

> Objetivo: dejar el proyecto reproducido y listo para el **Paso 2** en el ordenador con GPU.
> Contexto en [`00_ESTADO_Y_REANUDACION.md`](00_ESTADO_Y_REANUDACION.md).

## 0. Requisitos previos en el equipo nuevo
- **Python 3.12** (el proyecto se desarrolló con 3.12.4).
- **Git** (recomendado para transferir; opcional).
- **GPU NVIDIA** con drivers actualizados y **CUDA** (para PyTorch en el Paso 2).
- ~2 GB de disco libre (datos ~1 GB + entorno).

Comprobar la GPU (anota la **VRAM**, hace falta para elegir el tamaño de modelo):
```bat
nvidia-smi
```

## 1. Transferir el proyecto
**Opción 0 — Misma cuenta de OneDrive (probablemente tu caso):** el proyecto está en
`OneDrive\Documentos\MedXpertQA`. Si el equipo nuevo usa la **misma cuenta de OneDrive**, la
carpeta aparece **sincronizada sola** (código, `docs/`, `outputs/`). En ese caso, en el equipo
nuevo basta con: **(a) recrear `.venv`** (paso 2; el `.venv` sincronizado NO sirve entre
máquinas), y **(b)** si `data/` no se sincronizó del todo (son 500 MB / 2.858 ficheros),
re-descargar con `python src\download.py` (paso 3). Considera excluir `.venv/` de la
sincronización de OneDrive para no subir ~700 MB de binarios inservibles.

**Opción A — Git (recomendado, versiona el TFG):** en el equipo actual, `git init`, *commit* y
subir a un repo privado; en el nuevo, `git clone`. El `.gitignore` ya excluye `.venv/` y `data/`.

**Opción B — Copiar la carpeta:** copiar `MedXpertQA/` **excluyendo** `.venv/` (no es portable) y,
si se quiere ahorrar espacio, también `data/` (se regenera en el paso 3). Sí conviene copiar
`outputs/` (entregables del Paso 1) para no tener que reejecutar el EDA.

> Nota OneDrive: si el proyecto vive en OneDrive también en el equipo nuevo, mantener `.venv/` y
> `data/` fuera de la sincronización (ya excluidos en `.gitignore`).

## 2. Recrear el entorno del Paso 1 (EDA)
En **CMD**, desde la raíz del proyecto:
```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
Esto reproduce **exactamente** el entorno del EDA (versiones fijadas).

## 3. Recuperar los datos
- **Si copiaste `data/`**: nada que hacer.
- **Si no**: re-descargar (reproducible, revisión fijada `7e7c465…`):
```bat
python src\download.py
```
Descarga ~500 MB y extrae 2.858 imágenes. Idempotente. Regenera `data/raw/PROVENANCE.json`.

## 4. (Opcional) Regenerar salidas del Paso 1
Si no copiaste `outputs/`, regenéralas:
```bat
python src\verify.py
python src\eda.py
python src\render_reports.py
```
Y registra el kernel para el notebook:
```bat
python -m ipykernel install --user --name medxpertqa --display-name "Python (MedXpertQA .venv)"
```

## 5. Preparar el entorno de inferencia (Paso 2)
> Se instala **aparte** para no alterar el entorno de EDA. Se materializará en
> `requirements-eval.txt` durante el Paso 2. Guía de instalación:

1. **PyTorch con CUDA** acorde a la versión de CUDA de la GPU (ver `nvidia-smi`). Usar el índice
   oficial de PyTorch, p. ej. (ajustar `cu121`/`cu124` a tu CUDA):
   ```bat
   .venv\Scripts\activate.bat
   python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
2. **Resto de dependencias de inferencia**:
   ```bat
   python -m pip install accelerate bitsandbytes qwen-vl-utils
   ```
   (`transformers` 4.57.6 ya está instalado y soporta Qwen2.5-VL.)
3. **Verificar la GPU desde Python**:
   ```bat
   python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A', '| VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if torch.cuda.is_available() else 0)"
   ```
   Anota la **VRAM** que reporta: fija el tamaño de modelo (ver
   [`01_PLAN_PASO_2.md`](01_PLAN_PASO_2.md) §3).

## 6. Verificación rápida (todo en orden)
```bat
python src\verify.py
```
Debe informar recuentos 2000/2450 y 1 anomalía (la de `Basic Medicine`). Abrir
`outputs\report\01_eda.html` en el navegador para confirmar que las figuras se ven.

## 7. Reanudar
Seguir el **§7 de** [`00_ESTADO_Y_REANUDACION.md`](00_ESTADO_Y_REANUDACION.md): abrir el asistente
en la carpeta del proyecto y pegar el *prompt de reanudación* (con la VRAM ya conocida).
