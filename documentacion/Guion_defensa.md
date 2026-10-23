# Guion de defensa — Caso 1 UCE (8–12 minutos)

**Antes de empezar:** abre el dashboard en `docs/index.html`. Lleva el informe PDF, el PowerPoint y el CSV de métricas exportado. Explica lo que hiciste, no afirmes que funciona como un banco real. Los datos son sintéticos.

## Apertura (diapositivas 1–2; 1 minuto)

> El objetivo fue comparar cuatro algoritmos supervisados para predecir incumplimiento crediticio durante los próximos 12 meses. La variable `default_12m` es binaria. La prioridad metodológica fue evitar usar el conjunto de prueba durante la selección del modelo.

## Comprensión de los datos (diapositivas 3–4; 1,5 minutos)

> El CSV contiene 50.000 registros y 16 columnas. Excluí el identificador y la etiqueta como predictores, por lo que quedan 14 características. No encontré nulos ni duplicados. La clase positiva corresponde al 5,484 %, así que está desbalanceada. Un clasificador trivial obtendría 94,516 % de accuracy sin detectar incumplimientos: por eso una accuracy alta puede engañar.

## Metodología (diapositiva 5; 1,5 minutos)

> Separé 80 % de TRAIN y 20 % de TEST mediante estratificación y semilla 42. Construí Pipeline y ColumnTransformer: imputación para numéricas, One-Hot Encoding para categóricas y StandardScaler cuando lo exige la regresión logística. Utilicé StratifiedKFold de 5 folds sobre TRAIN. Transformaciones y ajuste quedaron dentro de los folds, sin fuga de información.

## Resultados CV (diapositivas 6–7; 1,5 minutos)

> Comparé Regresión Logística, Árbol de Decisión, Random Forest y Gradient Boosting. Medí Precision, Recall, F1, ROC-AUC y PR-AUC Average Precision; Accuracy fue complementaria. Como el problema está desbalanceado, definí de antemano seleccionar por mayor PR-AUC media de CV, desempate por Recall. Gradient Boosting alcanzó aproximadamente 11,38 % de AP CV y 67,00 % de ROC-AUC CV. Los gráficos OOF muestran predicciones sobre TRAIN de modelos que no entrenaron con esas filas.

## Limitación importante (diapositivas 8–11; 3 minutos)

> Gradient Boosting se seleccionó por su ranking de probabilidades, pero al umbral fijo 0,50 no detectó positivos en TEST: 9.452 verdaderos negativos, cero falsos positivos, 548 falsos negativos y cero verdaderos positivos. Precision, Recall y F1 fueron 0 %. Su ROC-AUC en test fue aproximadamente 67,56 % y la AP aproximadamente 12,03 %. Esto no es una contradicción: AUC evalúa la clasificación a muchos umbrales, mientras Recall 0,50 evalúa una decisión concreta.

> Analicé otros umbrales **solo en predicciones out-of-fold de TRAIN** para ilustrar que reducirlo aumenta Recall y también falsos positivos. No volví a seleccionar un modelo con TEST. La diferencia entre AP interna en entrenamiento (23,87 %) y CV (11,38 %) sugiere sobreajuste; no afirmo que esa brecha prueba causalmente el fenómeno.

## Escenarios de costos — punto 9 (diapositiva 9; 1 minuto)

> El enunciado exige analizar el costo de falsos negativos y falsos positivos, pero no da costos económicos reales. Por transparencia expreso los SUPUESTOS de costo en dólares: FN = US$ 500 por error y FP = US$ 100 por error. La fórmula es Costo simulado (USD) = FN × 500 + FP × 100. Para Gradient Boosting, sobre TRAIN OOF, con umbral 0,15 hubo 2.003 FN y 812 FP, equivalentes a US$ 1.082.700 hipotéticos; con umbral 0,50 hubo 2.194 FN y 0 FP, equivalentes a US$ 1.097.000 hipotéticos. No son pérdidas reales. La pestaña «Costos y errores · P9» permite modificar importes en USD y comparar los cuatro algoritmos. No he escogido el ganador ni el umbral usando TEST: el modelo ya estaba definido por PR-AUC en CV.

## Cierre (diapositivas 12–13; 1 minuto)

> Cumplí las instrucciones del caso y documenté las limitaciones. El dataset es sintético; antes de cualquier uso real harían falta costos reales de errores, calibración, selección del umbral con datos de entrenamiento, validación externa y temporal, y supervisión humana. El dashboard publicado muestra resultados agregados, nunca aprueba créditos.

---

## Preguntas frecuentes para practicar

**1. ¿Qué es clasificación supervisada?** Aprender una relación a partir de ejemplos etiquetados (incumple/no incumple) para predecir la clase futura.

**2. ¿Por qué no es regresión?** Porque `default_12m` es binaria; la regresión logística, aunque incluya «regresión» en el nombre, es un clasificador.

**3. ¿Qué representa la clase positiva?** Incumplimiento de pagos dentro de los siguientes 12 meses, codificado como 1.

**4. ¿Por qué excluiste `applicant_id`?** Es una clave de identificación sin significado predictivo demostrado; puede inducir patrones espurios.

**5. ¿Qué es estratificar?** Mantener aproximadamente la misma proporción de clases en el 80 % de TRAIN y el 20 % de TEST, y también en los folds.

**6. ¿Qué es data leakage?** Introducir información del conjunto reservado o de validación en entrenamiento. Evité ajustando transformaciones dentro de Pipeline y folds.

**7. ¿Por qué hiciste EDA después del split?** Para no usar la relación entre variables y clase de TEST en decisiones del experimento.

**8. ¿Qué es One-Hot Encoding?** Convertir valores categóricos en columnas indicadoras para que los modelos puedan usarlos.

**9. ¿Qué es StandardScaler?** Centrar y escalar variables numéricas, útil especialmente para el ajuste de Regresión Logística.

**10. ¿Cómo funcionan los folds?** De cinco particiones, cuatro entrenan y una valida; cada partición valida una vez, y se promedian métricas.

**11. ¿Cuál métrica usaste para seleccionar?** PR-AUC, reportada como Average Precision, porque hay solo alrededor de 5,5 % de positivos. El desempate es Recall CV.

**12. ¿Qué es PR-AUC?** Resume la relación precisión–recall a través de umbrales; se reportó con Average Precision, que no es exactamente la integral trapezoidal.

**13. ¿Qué es un falso positivo?** Una solicitud que cumpliría, clasificada como incumplidora; implicaría revisión innecesaria o una decisión injusta si se automatizara.

**14. ¿Qué es un falso negativo?** Una solicitud que incumpliría, clasificada como no incumplidora; el riesgo queda sin detectar.

**15. ¿Por qué el ganador tiene Recall 0 %?** Ordena probabilidades algo mejor que otros, pero ninguna superó el umbral fijo de 0,50 en TEST. Ranking global y decisión al umbral no son equivalentes.

**16. ¿Se puede cambiar el umbral?** Sí para un experimento futuro, pero debe elegirse con TRAIN/CV y evaluarse en otro test independiente sin usar los resultados de TEST actual para optimizarlo.

**17. ¿Qué es sobreajuste?** Cuando el modelo se adapta demasiado a los ejemplos de entrenamiento y no mantiene la calidad sobre ejemplos no vistos.

**18. ¿Por qué no evaluaste los cuatro en TEST?** Porque el caso exige reservar TEST hasta seleccionar el modelo final mediante CV. La comparación de los cuatro pertenece a TRAIN/CV.

**19. ¿Qué es HistGradientBoosting?** Una implementación eficiente del algoritmo Gradient Boosting que usa histogramas para encontrar divisiones en árboles de decisión.

**20. ¿GitHub Pages ejecuta los modelos Python?** No. Muestra resultados ya calculados con Python. El dashboard es HTML/CSS/JS estático sin backend.

**21. ¿Cuál es la limitación más importante?** Dataset didáctico sintético, desbalance, umbral 0,50 con cero detecciones y posible sobreajuste. No apto para decisiones reales.

**22. ¿Qué cambiarías en una segunda versión?** Definir costos FP/FN, experimentar con ajuste/calibración y umbral dentro de CV, y reservar un test realmente nuevo para medir el sistema ya congelado.

**23. ¿De dónde salen los importes US$ 500 y US$ 100?** Son supuestos monetarios en dólares usados para ilustrar escenarios; ni el PDF ni el CSV informan costos de errores reales. No representan valores verificados.

**24. ¿Qué significa costo total US$ 1.082.700?** Es el resultado HIPOTÉTICO de 2.003 FN × US$ 500 + 812 FP × US$ 100, usando OOF de TRAIN de Gradient Boosting con umbral 0,15. No es una pérdida financiera observada.

**25. ¿Elegiste el modelo según costo?** No: el ganador se eligió únicamente por PR-AUC media de 5 folds de TRAIN. El análisis de costo es un ejercicio posterior explicativo, sin usar test.
