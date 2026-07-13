# Índice de figuras, tablas e informes — EDA MedXpertQA

Generado por `src/eda.py`. 16 figuras (PNG 300 dpi), 22 tablas y 5 informes. Enlaces relativos a este fichero (`outputs/report/`).

## Lista de figuras

| Nº | Figura | Descripción |
|---:|---|---|
| 1 | [`c1_dist_body_system.png`](../figures/c1_dist_body_system.png) | Distribución por sistema corporal (body_system), MM vs Text (%). |
| 2 | [`c1_dist_medical_task.png`](../figures/c1_dist_medical_task.png) | Distribución por tarea médica (medical_task), MM vs Text (%). |
| 3 | [`c1_dist_question_type.png`](../figures/c1_dist_question_type.png) | Distribución por tipo de pregunta (question_type), MM vs Text (%). |
| 4 | [`c1_cross_question_type_x_medical_task.png`](../figures/c1_cross_question_type_x_medical_task.png) | Tabla cruzada question_type × medical_task (mapas de calor MM y Text). |
| 5 | [`c1_cross_body_system_x_medical_task.png`](../figures/c1_cross_body_system_x_medical_task.png) | Tabla cruzada body_system × medical_task (mapas de calor MM y Text). |
| 6 | [`c2_label_MM.png`](../figures/c2_label_MM.png) | Distribución de la letra correcta en MM (A–E) frente a la uniforme. |
| 7 | [`c2_label_Text.png`](../figures/c2_label_Text.png) | Distribución de la letra correcta en Text (A–J) frente a la uniforme. |
| 8 | [`c3_tokens_hist.png`](../figures/c3_tokens_hist.png) | Histograma de longitud del enunciado en tokens (cl100k) por subconjunto. |
| 9 | [`c3_tokens_by_qtype.png`](../figures/c3_tokens_by_qtype.png) | Longitud del enunciado por tipo de pregunta (diagramas de caja). |
| 10 | [`c3_similarity_MM.png`](../figures/c3_similarity_MM.png) | Máxima similitud coseno intra-subconjunto en MM. |
| 11 | [`c3_similarity_Text.png`](../figures/c3_similarity_Text.png) | Máxima similitud coseno intra-subconjunto en Text. |
| 12 | [`c4_images_per_question.png`](../figures/c4_images_per_question.png) | Número de imágenes por pregunta en MM. |
| 13 | [`c4_image_properties.png`](../figures/c4_image_properties.png) | Propiedades de las imágenes MM: anchura, altura, área, aspecto, tamaño y color/gris. |
| 14 | [`c4_image_formats.png`](../figures/c4_image_formats.png) | Formato de archivo de las imágenes MM. |
| 15 | [`c4_contact_sheet.png`](../figures/c4_contact_sheet.png) | Hoja de contactos: muestra estratificada por body_system para anotación manual de modalidad. |
| 16 | [`c5_text_image_coupling.png`](../figures/c5_text_image_coupling.png) | Acoplamiento texto↔imagen en MM: mención explícita, implícita y opciones que refieren figura. |

## Lista de tablas

| Nº | Tabla | Descripción |
|---:|---|---|
| 1 | [`c1_dist_body_system.csv`](../tables/c1_dist_body_system.csv) | Conteos y porcentajes por sistema corporal (MM y Text). |
| 2 | [`c1_dist_medical_task.csv`](../tables/c1_dist_medical_task.csv) | Conteos y porcentajes por tarea médica. |
| 3 | [`c1_dist_question_type.csv`](../tables/c1_dist_question_type.csv) | Conteos y porcentajes por tipo de pregunta. |
| 4 | [`c1_cross_question_type_x_medical_task_MM.csv`](../tables/c1_cross_question_type_x_medical_task_MM.csv) | Tabla cruzada question_type × medical_task (MM). |
| 5 | [`c1_cross_question_type_x_medical_task_Text.csv`](../tables/c1_cross_question_type_x_medical_task_Text.csv) | Tabla cruzada question_type × medical_task (Text). |
| 6 | [`c1_cross_body_system_x_medical_task_MM.csv`](../tables/c1_cross_body_system_x_medical_task_MM.csv) | Tabla cruzada body_system × medical_task (MM). |
| 7 | [`c1_cross_body_system_x_medical_task_Text.csv`](../tables/c1_cross_body_system_x_medical_task_Text.csv) | Tabla cruzada body_system × medical_task (Text). |
| 8 | [`c1_options_answers.csv`](../tables/c1_options_answers.csv) | Verificación del nº de opciones y de respuestas correctas por subconjunto. |
| 9 | [`c2_baselines.csv`](../tables/c2_baselines.csv) | Baselines de azar y de mayoría + test χ² de uniformidad. |
| 10 | [`c2_label_distribution.csv`](../tables/c2_label_distribution.csv) | Frecuencia de cada letra correcta por subconjunto. |
| 11 | [`c3_length_summary.csv`](../tables/c3_length_summary.csv) | Resumen estadístico de longitudes (caracteres/palabras/tokens) por subconjunto. |
| 12 | [`c3_length_by_qtype.csv`](../tables/c3_length_by_qtype.csv) | Longitudes por subconjunto y tipo de pregunta. |
| 13 | [`c3_near_duplicates.csv`](../tables/c3_near_duplicates.csv) | Pares de preguntas casi-duplicadas por texto (coseno ≥ 0,85). |
| 14 | [`c3_text_lengths.csv`](../tables/c3_text_lengths.csv) | Tabla base por pregunta: caracteres, palabras y tokens (cl100k y Qwen2.5-VL). |
| 15 | [`c4_images_per_question.csv`](../tables/c4_images_per_question.csv) | Distribución del nº de imágenes por pregunta. |
| 16 | [`c4_image_dimensions.csv`](../tables/c4_image_dimensions.csv) | Percentiles de anchura, altura, área, aspecto y tamaño de archivo. |
| 17 | [`c4_image_summary.json`](../tables/c4_image_summary.json) | Resumen: formatos, reparto color/gris y huella en disco. |
| 18 | [`c4_image_properties.csv`](../tables/c4_image_properties.csv) | Propiedades + hashes (pHash, md5) por imagen (2.852 filas). |
| 19 | [`c4_contact_sheet_manifest.csv`](../tables/c4_contact_sheet_manifest.csv) | Manifiesto de las imágenes incluidas en la hoja de contactos. |
| 20 | [`c4_exact_duplicates.csv`](../tables/c4_exact_duplicates.csv) | Grupos de imágenes byte-idénticas (md5) dentro de MM. |
| 21 | [`c4_duplicate_question_pairs.csv`](../tables/c4_duplicate_question_pairs.csv) | Pares de preguntas que comparten imagen idéntica + su similitud textual. |
| 22 | [`c5_text_image_coupling.csv`](../tables/c5_text_image_coupling.csv) | Por pregunta: mención de figura, términos detectados y opciones que refieren figura. |

## Informes

| Documento | Descripción |
|---|---|
| [`00_integridad.md`](00_integridad.md) | Informe de verificación de integridad (Tarea B). |
| [`01_eda.md`](01_eda.md) | Informe de EDA integrado con hallazgos e implicaciones de diseño (Tarea C). |
| [`data_card.md`](data_card.md) | Data card resumida del dataset. |
| [`indice_figuras_tablas.md`](indice_figuras_tablas.md) | Este índice navegable de figuras, tablas e informes. |
| [`hallazgos_one_pager.md`](hallazgos_one_pager.md) | Resumen ejecutivo de una página con los hallazgos clave. |

