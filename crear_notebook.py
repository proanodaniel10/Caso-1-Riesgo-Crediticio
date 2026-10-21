from pathlib import Path
import nbformat as nbf
nb=nbf.v4.new_notebook()
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
def md(s):nb.cells.append(nbf.v4.new_markdown_cell(s))
def code(s):nb.cells.append(nbf.v4.new_code_cell(s))
md('''# UCE · Caso 48: Predicción de incumplimiento crediticio

**Proyecto completo — Notebook pedagógico en Python.**

Este notebook acompaña al script reproducible `src/entrenar.py` del repositorio. Necesita la estructura del proyecto y el CSV original. Su objetivo es que puedas **entender, ejecutar y defender** cada una de las 12 instrucciones del caso. **No contiene resultados inventados.**

**Ruta correcta para ejecutar:** sitúa `riesgo_crediticio_bigdata.csv` en la carpeta `data/` de la carpeta raíz del proyecto. Abre Jupyter o VS Code desde la raíz (o desde la subcarpeta notebooks); el notebook detecta ambas.

> Atención: si solo subes el notebook individual a Colab, debes subir también la carpeta `src` y el CSV. En una cuenta con Drive montado, abre el proyecto desde Drive. Una alternativa directa es ejecutar `python src/entrenar.py --csv data/riesgo_crediticio_bigdata.csv`.
''')
md('''## 0. Bibliotecas y rutas del proyecto

Verifica que existan las librerías indicadas en `requirements.txt`. Para Windows o Linux, desde la carpeta raíz ejecuta `pip install -r requirements.txt`.''')
code('''from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

cwd = Path.cwd()
ROOT = cwd if (cwd / 'src' / 'entrenar.py').exists() else cwd.parent
assert (ROOT / 'src' / 'entrenar.py').exists(), 'Ejecuta Jupyter desde el proyecto o su subcarpeta notebooks.'
sys.path.insert(0, str(ROOT / 'src'))
from entrenar import train, build_models, processor, PREDICTORS, TARGET, EXCLUDED, CAT, NUM, SEED, FOLDS, metrics_at
CSV = ROOT / 'data' / 'riesgo_crediticio_bigdata.csv'
assert CSV.exists(), 'Copia riesgo_crediticio_bigdata.csv a la carpeta data/ (no se distribuye en GitHub Pages).'
print('Proyecto:', ROOT)
print('CSV:', CSV)''')
md('''## 1. Cargar los datos: dimensiones, tipos, duplicados y nulos

**Instrucción 1.** Se inspecciona el dataset original **sin entrenar**. La columna `applicant_id` es un identificador y no debe entrar al modelo.''')
code('''df = pd.read_csv(CSV)
print('Dimensiones:', df.shape)
print('Duplicados completos:', df.duplicated().sum())
print('IDs repetidos:', df.applicant_id.duplicated().sum())
print('Nulos por variable:')
display(df.isna().sum().to_frame('nulos'))
display(df.dtypes.to_frame('tipo'))
display(df.head(5))''')
md('''## 2. Separar X/y y el holdout ANTES de la EDA supervisada

**Instrucciones 3–4.** Se implementan antes de la instrucción 2 para no explorar relaciones con la clase de TEST. Semilla 42 y estratificación obligatoria.
''')
code('''from sklearn.model_selection import train_test_split
X = df[PREDICTORS].copy()
y = df[TARGET].copy()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,stratify=y,random_state=SEED)
print('X:', X.shape, 'Train:', X_train.shape, 'Test:', X_test.shape)
print('Clase positiva en TRAIN:', round(y_train.mean()*100,2), '%')
assert 'applicant_id' not in X and TARGET not in X''')
md('''## 3. Análisis exploratorio — solo TRAIN

**Instrucción 2.** El gráfico de distribución, las estadísticas descriptivas, los conteos categóricos, los outliers (regla IQR) y la relación con la clase se calculan usando solo el entrenamiento.''')
code('''print('Distribución de clase en TRAIN:')
display(y_train.value_counts().sort_index().to_frame('cantidad'))
y_train.value_counts().sort_index().plot(kind='bar',title='Distribución de default_12m · TRAIN',rot=0)
plt.show()
display(X_train[NUM].describe().T)
for c in CAT:
    print(c)
    display(X_train[c].value_counts(dropna=False))''')
code('''for c in ['monthly_income_usd','debt_to_income_ratio','credit_utilization','late_payments_12m']:
    s=X_train[c].dropna()
    q1,q3=s.quantile([.25,.75]); iqr=q3-q1
    n=int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum())
    print(c, 'posibles atípicos IQR:',n)

eda_train=X_train.copy()
eda_train[TARGET]=y_train
print('Tasa de incumplimiento por tipo de empleo:')
display(eda_train.groupby('employment_type')[TARGET].agg(['size','mean']))''')
md('''## 4. ColumnTransformer + Pipeline

**Instrucción 5.** Imputación de mediana en numéricas, imputación de moda + One-Hot en categóricas. `StandardScaler` se usa para regresión logística. El pipeline se aprende de nuevo dentro de cada fold; por eso evitamos fuga de información.''')
code('''from sklearn.pipeline import Pipeline
print('Predictoras numéricas:',NUM)
print('Predictoras categóricas:',CAT)
print('Preprocesador:',processor(scale=True))''')
md('''## 5. Los cuatro modelos

**Instrucción 6.** LogisticRegression, DecisionTreeClassifier, RandomForestClassifier y **HistGradientBoostingClassifier** (algoritmo Gradient Boosting optimizado por histogramas). Los hiperparámetros están fijados antes de usar test.''')
code('''models=build_models()
for nombre,pipeline in models.items():
    print('\n',nombre)
    print(pipeline.named_steps['clasificador'])''')
md('''## 6. Entrenar y evaluar por CV de 5 folds (tarea que más tarda)

**Instrucciones 7–9.** La función `train()` implementa cinco entrenamientos por algoritmo; genera predicciones *out-of-fold*, promedia Precision, Recall, F1, ROC-AUC, PR-AUC (Average Precision) y Accuracy, selecciona exclusivamente por PR-AUC CV y finalmente evalúa **una sola vez** el modelo elegido en TEST. No vuelvas a ejecutar TEST para comparar otros algoritmos; el protocolo está escrito en `src/entrenar.py`. Este proceso podría tardar varios minutos según el ordenador.
''')
code('''# Entrenamiento reproducible desde cero en el CSV oficial (sin gráficos falsos):
resultado = train(CSV, ROOT / 'resultados', show_progress=True)
print('Modelo elegido usando SOLO TRAIN:', resultado['final']['modelo'])''')
md('''## 7. Comparación de métricas y estabilidad (CV)

**Instrucción 8.** Se presenta la media y desviación estándar por modelo para comparar la estabilidad entre folds.''')
code('''cv = pd.read_csv(ROOT/'resultados'/'comparacion_cv.csv')
display(cv)
display(cv[['modelo','pr_auc_media','roc_auc_media','recall_media','precision_media','f1_media']])
cv.plot(x='modelo',y=['pr_auc_media','roc_auc_media'],kind='bar',rot=15,title='PR-AUC y ROC-AUC · media CV',figsize=(9,4))
plt.tight_layout();plt.show()''')
md('''## 8. Matrices de confusión por modelo (solo TRAIN OOF)

**Instrucciones 8–9.** Observa las matrices OOF; recuerda que sus filas representan las clases reales y columnas las clases predichas. FN significa que un incumplidor no fue detectado.''')
code('''from sklearn.metrics import ConfusionMatrixDisplay
fig,axes=plt.subplots(2,2,figsize=(9,7))
for ax,(name,rec) in zip(axes.ravel(),resultado['modelos'].items()):
    c=rec['confusion_oof']
    mat=[[c['tn'],c['fp']],[c['fn'],c['tp']]]
    ConfusionMatrixDisplay(mat,display_labels=['No incumple','Incumple']).plot(ax=ax,colorbar=False)
    ax.set_title(name)
plt.tight_layout();plt.show()''')
md('''## 9. Comparar los escenarios de umbral en TRAIN OOF

**Instrucción 9.** La decisión no depende de test. Esta comparación pedagógica muestra cómo cambian FP, FN, Recall y Precision en función del umbral; se mantiene el umbral final fijo 0,50 en la evaluación oficial del ejercicio.''')
code('''nombre=resultado['final']['modelo']
escenarios=resultado['modelos'][nombre]['umbrales_oof']
rows=[{'umbral':r['umbral'],'precision':r['precision'],'recall':r['recall'],'f1':r['f1'],
       'FP':r['confusion']['fp'],'FN':r['confusion']['fn']} for r in escenarios]
display(pd.DataFrame(rows))''')
md('''## 9b. Análisis cuantitativo de costos hipotéticos — solo TRAIN OOF

**Instrucción 9, ampliación:** el PDF pide estudiar el costo de clasificar como bajo riesgo a alguien que incumple (FN) y el costo de revisar o rechazar a un buen pagador (FP). No especifica costos reales; por ello **solo se usan montos hipotéticos en USD**. La fórmula es:

`Costo total hipotético (USD) = FN × costo_FN_USD + FP × costo_FP_USD`

Los umbrales 0,15 y 0,50, y los cuatro algoritmos se comparan con las matrices de confusión de predicciones **fuera de fold de TRAIN**, jamás con TEST. La selección del modelo oficial continúa siendo por mayor AP media de CV.
''')
code('''# Cambia estos dos importes HIPOTÉTICOS en dólares estadounidenses (USD).
# No son pérdidas reales observadas en la institución financiera.
COSTO_FN = 500.0  # US$ 500 por incumplimiento no detectado
COSTO_FP = 100.0  # US$ 100 por buen pagador revisado o rechazado sin motivo

filas_costo = []
for nombre_modelo, res_modelo in resultado['modelos'].items():
    for u in [0.15, 0.50]:
        r = next(t for t in res_modelo['umbrales_oof'] if abs(t['umbral'] - u) < 1e-8)
        c = r['confusion']
        filas_costo.append({
            'Modelo': nombre_modelo,
            'Umbral': u,
            'FN': c['fn'],
            'FP': c['fp'],
            'Costo_FN_USD': c['fn'] * COSTO_FN,
            'Costo_FP_USD': c['fp'] * COSTO_FP,
            'Costo_total_USD_hipotetico': c['fn'] * COSTO_FN + c['fp'] * COSTO_FP
        })
costos_9 = pd.DataFrame(filas_costo)
display(costos_9)
print('¡Importante! Estos totales se expresan en USD, pero son importes HIPOTÉTICOS, no pérdidas reales.')
print('La selección oficial del modelo y el TEST permanecen intactos.')''')
code('''fig, ax = plt.subplots(figsize=(10, 4))
for nombre_modelo, subtabla in costos_9.groupby('Modelo'):
    ax.plot(subtabla['Umbral'], subtabla['Costo_total_USD_hipotetico'], marker='o',
            label=nombre_modelo)
ax.set_title('Costo hipotético vs umbral — predicciones OOF TRAIN')
ax.set_xlabel('Umbral')
ax.set_ylabel('Costo hipotético (USD)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()''')

md('''## 10. Única evaluación final en TEST (realizada dentro del protocolo)

**Instrucción 10.** No vuelvas a entrenar y escoger otro modelo usando estos resultados: el TEST quedó reservado hasta después de seleccionar el modelo por CV.''')
code('''print('Modelo seleccionado:',nombre)
test=pd.read_csv(ROOT/'resultados'/'metricas_test_FINAL.csv')
display(test)
cm=resultado['final']['confusion_test']
print('TN:',cm['tn'],'FP:',cm['fp'],'FN:',cm['fn'],'TP:',cm['tp'])''')
md('''## 11. Comparación CV versus TEST y posible sobreajuste

**Instrucción 11.** Para analizar sobreajuste observa la diferencia entre métricas internas de entrenamiento y validación; además contrasta la generalización en TEST.''')
code('''display(pd.read_csv(ROOT/'resultados'/'cv_vs_test.csv'))
f=resultado['final']
print('Brecha AP TRAIN interno - CV:', round(f['gap_ap_train_cv'],4))
print('Brecha AP CV - TEST:', round(f['gap_ap_cv_test'],4))''')
md('''## 12. Conclusiones y preparación de defensa

**Instrucción 12.** La base es sintética. No es un sistema de aprobación/rechazo automático. Es obligatorio discutir los FN/FP y los límites reales del experimento.

- **Hallazgo principal:** Gradient Boosting presenta la PR-AUC media más alta en CV, aunque a umbral 0,50 su Recall es nulo. La accuracy alta se explica por el fuerte desbalance.
- **Justificación de PR-AUC:** la clase positiva es minoritaria; es más útil que Accuracy para el objetivo académico de comparación.
- **Interpretación de errores:** FN significa riesgo no identificado; FP significa advertir innecesariamente sobre alguien que pagaría. Los costos monetarios reales no aparecen en el caso.
- **Sobreajuste:** una brecha relevante entre métricas internas de TRAIN y validación hace sospechar que el modelo no generaliza tan bien como en los ejemplos ajustados.
- **Limitaciones:** dataset artificial, ausencia de costos reales, sin test temporal/externo ni evaluación de impacto y equidad.

Consulta también el dashboard en `docs/index.html`, el informe técnico y las diapositivas. **No uses información de TEST para retroajustar el umbral y reportar nuevos resultados en este mismo conjunto.**
''')
nbf.validate(nb)
nbf.write(nb,Path('/mnt/data/Proyecto_Caso_48/notebooks/Caso_48_Paso_a_Paso.ipynb'))
print('Celdas',len(nb.cells))
