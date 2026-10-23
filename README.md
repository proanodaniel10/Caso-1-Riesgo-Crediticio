# Caso 1 UCE — Predicción de incumplimiento crediticio

Proyecto completo de minería de datos basado en el **caso oficial de dos páginas** y en el archivo sintético `riesgo_crediticio_bigdata.csv` (50.000 filas, 16 columnas).

## 1. Entregables

| Ruta | Contenido |
|---|---|
| `docs/index.html` | **Dashboard interactivo listo para GitHub Pages** (sin backend) |
| `dashboard/index.html` | Copia de trabajo del dashboard, abre incluso sin conexión |
| `src/entrenar.py` | Experimento Python completo y reproducible |
| `notebooks/Caso_48_Paso_a_Paso.ipynb` | Notebook didáctico con los 12 requisitos |
| `resultados/` | CSV de métricas y JSON agregado, calculados con datos oficiales |
| `reportes/Informe_Tecnico_Caso_48_UCE.docx` | Informe formal editable |
| `reportes/Informe_Tecnico_Caso_48_UCE.pdf` | Informe para entregar/imprimir |
| `reportes/Defensa_Caso_48_UCE.pptx` | Presentación editable de 13 diapositivas |
| `documentacion/Guion_defensa.md` | Guion y posibles preguntas de la defensa |

## 2. Abrir el dashboard en tu computadora (sin instalar nada)

1. Descarga y descomprime el paquete del proyecto.
2. Entra en `dashboard/` o `docs/`.
3. Haz doble clic en `index.html` para abrirlo con Chrome, Firefox o Edge. **Funciona sin internet**: usa HTML, CSS, JavaScript y datos agregados precomputados.
4. Prueba todas las pestañas, cambia entre los cuatro modelos y ajusta el umbral de clasificación. En **Costos y errores · P9**, modifica los costos hipotéticos de FN/FP expresados en dólares (USD) y compara los escenarios. **Todos los escenarios de costos/umbral son únicamente TRAIN OOF**, sin alterar el modelo seleccionado.

No contiene datos personales de solicitantes ni identificadores; solo resultados agregados. El dashboard no predice solicitudes nuevas y no puede ejecutar Python en GitHub Pages.

## 3. Reentrenar los modelos desde cero

Desde la carpeta raíz del proyecto, coloca el CSV oficial en `data/riesgo_crediticio_bigdata.csv` y ejecuta:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/entrenar.py --csv data/riesgo_crediticio_bigdata.csv --out resultados
```

El proceso entrena 4 modelos × 5 folds; al finalizar reentrena el seleccionado y evalúa una sola vez TEST. Para actualizar el dashboard con los resultados nuevos, ejecuta:

```bash
python src/publicar_resultados.py
```

Abre `notebooks/Caso_48_Paso_a_Paso.ipynb` desde JupyterLab para estudiar cada paso.

## 4. Publicación en GitHub Pages (sin servidor Python)

1. Crea un repositorio de GitHub (por ejemplo `caso-48-uce`) y sube los archivos del proyecto. **No subas el CSV original**; `.gitignore` ignora `/data/*.csv` al usar Git, pero una carga manual desde la web podría saltarse esa protección.
2. Ve a **Settings → Pages**.
3. En **Build and deployment**, elige **Deploy from a branch**.
4. Configura **Branch: main** y **Folder: /docs**, luego pulsa **Save**.
5. Cuando termine el despliegue, la dirección normalmente será `https://TU_USUARIO.github.io/caso-48-uce/` (debes sustituir el nombre y el repositorio reales). **No está publicado aún.**

El dashboard en `/docs` ya lleva `index.html`, CSS, JavaScript y métricas agregadas; no requiere Node, API ni suscripción. El contenido publicado en Pages es público: no incluyas datos que no quieras exponer.

Documentación oficial: https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

## 5. Regla crítica para el examen

El punto 9 dispone de un comparador cuantitativo en USD: Costo simulado (USD) = FN × C_FN(USD) + FP × C_FP(USD). Los importes iniciales son US$ 500 por FN y US$ 100 por FP; SON SUPUESTOS EDITABLES, no pérdidas financieras reales. Ni el PDF ni el CSV proporcionan valores de costo por error. El comparador utiliza solo matrices OOF de TRAIN.

El **holdout de prueba nunca se utiliza para selección de modelo o de hiperparámetros**. Todo ajuste y transformación aprendida ocurren dentro de CV. Se elige el modelo por mayor PR-AUC (Average Precision) media, se reentrena con el 80 % de TRAIN y se evalúa el TEST de forma única. La precisión académica exige reconocer que el modelo elegido alcanzó **Recall 0 %** con umbral fijo 0,50. No se deben maquillar cifras ni afirmar utilidad crediticia real.

## 6. Referencias

- Caso 1 UCE, enunciado académico suministrado, pp. 1–2.
- scikit-learn: https://scikit-learn.org/stable/
- GitHub Pages: https://docs.github.com/en/pages

## 7. Mantener actualizado el dashboard de GitHub Pages

La **única carpeta que se edita** es `dashboard/` (HTML, CSS, JS y recursos públicos).
La carpeta `docs/` es la copia preparada para GitHub Pages. **No se debe editar `docs/index.html` directamente** porque la siguiente sincronización lo reemplazaría.

Después de cada cambio, desde la raíz del proyecto en VS Code, ejecuta:

```bash
python sincronizar_dashboard.py
```

El script compara los archivos, copia los modificados y conserva archivos exclusivos de `docs/` (como `CNAME` para un dominio personalizado). Si se elimina un archivo del dashboard, solo se elimina de `docs/` cuando el script lo había sincronizado anteriormente. Los CSV permitidos son exclusivamente los cinco archivos de **resultados agregados**; el CSV de solicitudes `data/riesgo_crediticio_bigdata.csv` no se copia.

Opcionalmente puedes comprobar si `docs/` necesita actualización antes de publicar:

```bash
python sincronizar_dashboard.py --verificar
```

O comprobar qué cambiaría sin copiar:

```bash
python sincronizar_dashboard.py --simular
```

**Para publicar en GitHub Pages:** después de sincronizar hay que realizar **Commit + Push** al repositorio. Sin subir los cambios, la web pública NO se actualiza. GitHub Pages sirve el contenido de `docs/` de la rama configurada (`main` en este proyecto).

Si el cambio consiste en **reentrenar y actualizar resultados**:

```bash
python src/entrenar.py --csv data/riesgo_crediticio_bigdata.csv --out resultados
python src/publicar_resultados.py
```

La segunda instrucción genera los datos de `dashboard/` y ejecuta automáticamente `sincronizar_dashboard.py`, de modo que `docs/` también se actualiza.

**Aviso sobre privacidad:** antes del primer Push revisa qué archivos vas a publicar. GitHub Pages publica archivos en `docs/`. Nunca incluyas archivos de solicitudes o información privada dentro de `dashboard/` ni `docs/`.
