# Hallazgos clave del EDA — MedXpertQA (one-pager)

*TFG · Paso 1. Fuente: `TsinghuaC3I/MedXpertQA` rev. `7e7c465a68eb…`. Alcance: test (MM=2000, Text=2450).*

## El dataset en cifras
- **2000** preguntas MM (5 opciones, con imagen) y **2450** Text (10 opciones, sin imagen); opción única (1 respuesta correcta, siempre ∈ opciones).
- **2852** imágenes MM, 524.2 MB; formatos JPEG 2304, PNG 548.
- **Integridad verificada** (recuentos, imagen↔archivo, 0 corruptas, IDs únicos); 1 anomalía menor documentada (`Basic Medicine` en dev).

## Hallazgos
1. **Sesgo temático multimodal.** MM está dominado por *Skeletal* (24.55 %) e *Integumentary* (11.25 % vs 1.96 % en Text): las especialidades visuales pesan más en MM.
2. **Predominio del razonamiento diagnóstico.** *Reasoning* y *Diagnosis* son mayoritarios (cuadrante dominante Reasoning × Diagnosis).
3. **Sin sesgo posicional explotable.** Baseline de mayoría 21.2 % (MM) / 10.7 % (Text) frente a azar 20 / 10 %; χ² no rechaza la uniformidad.
4. **Coste de contexto.** Prompt (enunciado+opciones): mediana 187 (MM) / 355 (Text) tokens; p95 hasta 589 (Text), más tokens visuales.
5. **Componente visual exigente.** 20.9 % de MM es multi-imagen (hasta 6); ~60 % en escala de grises. Modalidad **no etiquetada** (limitación).
6. **Acoplamiento multimodal.** 78.8 % de MM menciona la figura explícitamente; MM **no** es resoluble de forma fiable solo con texto.
7. **Calidad del benchmark.** 35 grupos de imágenes idénticas → 42 pares de preguntas comparten imagen; **1** es una pregunta casi-duplicada real (MM-121/MM-174).

## Implicaciones para el diseño
- VLM con **soporte multi-imagen** (Qwen2.5-VL) o estrategia de mosaico definida.
- Dimensionar **ventana de contexto y coste** según las longitudes anteriores.
- Reportar **baselines de azar y mayoría** y *accuracy* **desagregada por `body_system`**.
- El margen de mejora está en el **razonamiento** → CoT y RAG como palancas.

## Decisiones pendientes
- `dev.jsonl` como pool de few-shot (EDA sobre test). — *a confirmar*
- Tratamiento de preguntas multi-imagen en el pipeline. — *para el Paso 2*
- Inferencia de modalidad (heurística/clustering) vs anotación manual. — *a decidir*

