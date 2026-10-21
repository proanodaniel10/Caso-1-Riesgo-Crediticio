from __future__ import annotations
import argparse
import json
import sys
import platform
from datetime import datetime, timezone
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve)

SEED=42
TARGET='default_12m'
EXCLUDED=['applicant_id']
PREDICTORS=['age','monthly_income_usd','employment_years','employment_type',
            'housing_status','credit_amount_usd','term_months','debt_to_income_ratio',
            'credit_utilization','late_payments_12m','open_credits','savings_usd',
            'loan_purpose','has_guarantor']
CAT=['employment_type','housing_status','loan_purpose']
NUM=[c for c in PREDICTORS if c not in CAT]
THRESHOLD=.50
FOLDS=5
METRIC_NAMES=['precision','recall','f1','roc_auc','pr_auc','accuracy']


def safe_float(x):
    x=float(x)
    return round(x, 6) if np.isfinite(x) else None


def metrics_at(y_true, proba, threshold=.5):
    y_pred=(np.asarray(proba)>=threshold).astype(int)
    return {'precision':safe_float(precision_score(y_true,y_pred,zero_division=0)),
            'recall':safe_float(recall_score(y_true,y_pred,zero_division=0)),
            'f1':safe_float(f1_score(y_true,y_pred,zero_division=0)),
            'roc_auc':safe_float(roc_auc_score(y_true,proba)),
            'pr_auc':safe_float(average_precision_score(y_true,proba)),
            'accuracy':safe_float(accuracy_score(y_true,y_pred))}


def matrix_at(y_true, proba, threshold=.5):
    c=confusion_matrix(y_true,(np.asarray(proba)>=threshold).astype(int),labels=[0,1])
    tn,fp,fn,tp=c.ravel()
    return {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def subsample_curve(xs,ys, max_points=170):
    # sample monotonic curve by indices, retaining endpoints
    if len(xs)>max_points:
        ix=np.unique(np.linspace(0,len(xs)-1,max_points).astype(int))
        xs=np.asarray(xs)[ix];ys=np.asarray(ys)[ix]
    return [[safe_float(a),safe_float(b)] for a,b in zip(xs,ys)]


def curves(y, p):
    fpr,tpr,_=roc_curve(y,p)
    precision, recall,_=precision_recall_curve(y,p)
    return {'roc':subsample_curve(fpr,tpr),
            'pr':subsample_curve(recall[::-1],precision[::-1])}


def thresholds(y,p):
    records=[]
    for t in np.linspace(.05,.95,19):
        score=metrics_at(y,p,t)
        records.append({'umbral':round(float(t),2),
                        'precision':score['precision'],'recall':score['recall'],
                        'f1':score['f1'],'confusion':matrix_at(y,p,t)})
    return records


def ohe():
    # Compatible con scikit-learn moderno y versiones anteriores.
    from inspect import signature
    opt={'handle_unknown':'ignore'}
    if 'sparse_output' in signature(OneHotEncoder).parameters:
        opt['sparse_output']=False
    else:
        opt['sparse']=False
    return OneHotEncoder(**opt)


def processor(scale=False):
    numeric=[('imputar',SimpleImputer(strategy='median'))]
    if scale: numeric.append(('escalar',StandardScaler()))
    categorical=[('imputar',SimpleImputer(strategy='most_frequent')),
                 ('one_hot',ohe())]
    return ColumnTransformer([
        ('numericas',Pipeline(numeric),NUM),
        ('categoricas',Pipeline(categorical),CAT)
    ],remainder='drop',sparse_threshold=0)


def build_models():
    config={
        'Regresión Logística': (LogisticRegression(max_iter=1300,class_weight='balanced',random_state=SEED),True),
        'Árbol de Decisión':(DecisionTreeClassifier(max_depth=6,min_samples_leaf=25,class_weight='balanced',random_state=SEED),False),
        'Random Forest':(RandomForestClassifier(n_estimators=110,min_samples_leaf=5,
                            max_features='sqrt',class_weight='balanced_subsample',n_jobs=-1,random_state=SEED),False),
        'Gradient Boosting':(HistGradientBoostingClassifier(max_iter=110, max_leaf_nodes=15,
                             learning_rate=.07, l2_regularization=.1, random_state=SEED),False)
    }
    return {name:Pipeline([('procesamiento',processor(scale)),('clasificador',est)])
            for name,(est,scale) in config.items()}


def profile_train(train, y, meta):
    # EDA descriptivo SOLO TRAIN; valor inicial global de conteos se destina al control de calidad.
    eda={}
    eda['target_counts']={'0':int((y==0).sum()),'1':int((y==1).sum())}
    eda['target_rate']=safe_float(y.mean())
    eda['categories']={}
    for col in CAT+['has_guarantor']:
        tmp=pd.DataFrame({'variable':train[col].astype(str),'target':y.to_numpy()})
        g=tmp.groupby('variable')['target'].agg(['size','mean']).sort_values('size',ascending=False)
        eda['categories'][col]=[{'categoria':str(idx),'n':int(row['size']),
                                  'tasa':safe_float(row['mean'])} for idx,row in g.iterrows()]
    eda['numeric']={}; outliers=[]
    for col in NUM:
        s=pd.to_numeric(train[col],errors='coerce').dropna()
        q1,q3=s.quantile([.25,.75]); iqr=q3-q1
        n_out=int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum()) if iqr>0 else 0
        outliers.append({'variable':col,'n':n_out,'porcentaje':safe_float(n_out/len(s)*100)})
        eda['numeric'][col]={
            'min':safe_float(s.min()),'max':safe_float(s.max()),
            'mean':safe_float(s.mean()),'median':safe_float(s.median()),
            'std':safe_float(s.std()),'q1':safe_float(q1),'q3':safe_float(q3),
            'outliers_iqr':n_out}
        # Distribución agregada por clase para dashboard: hist en intervalos comunes
        bins=np.histogram_bin_edges(s.to_numpy(),bins=18)
        if not np.all(np.diff(bins)>0):
            bins=np.linspace(s.min()-.5,s.max()+.5,19)
        h0,_=np.histogram(s[y==0],bins=bins)
        h1,_=np.histogram(s[y==1],bins=bins)
        eda['numeric'][col]['hist']={
            'centros':[safe_float(v) for v in (bins[:-1]+bins[1:])/2],
            'no_incumple':h0.tolist(),'incumple':h1.tolist()}
    eda['outliers']=sorted(outliers,key=lambda row:row['porcentaje'],reverse=True)
    return eda


def train(csv_path, out_dir, show_progress=True):
    csv_path=Path(csv_path);out_dir=Path(out_dir)
    out_dir.mkdir(exist_ok=True,parents=True)
    df=pd.read_csv(csv_path,low_memory=False)
    df.columns=df.columns.str.strip()
    missing=set(PREDICTORS+[TARGET]+EXCLUDED)-set(df.columns)
    if missing:raise ValueError('Columnas del caso ausentes: '+str(sorted(missing)))
    if list(df.columns)!=['applicant_id']+PREDICTORS+[TARGET]:
        print('Nota: orden de columnas del CSV diferente del diccionario, se seleccionan por nombre')
    before=len(df)
    row_dupes=int(df.duplicated().sum());id_dupes=int(df.applicant_id.duplicated().sum())
    if row_dupes:
        df=df.drop_duplicates().reset_index(drop=True)
    if id_dupes:
        # Evitar fuga de datos por solicitantes repetidos, exigir revisión.
        raise ValueError('Hay applicant_id repetidos; definir separación por grupos antes de ejecutar CV.')
    if df[TARGET].isna().any() or not df[TARGET].isin([0,1]).all():
        raise ValueError('La variable objetivo debe ser binaria 0/1 sin faltantes.')
    if df[TARGET].nunique()!=2:raise ValueError('Se necesitan ambas clases.')
    if df[NUM].select_dtypes(exclude=np.number).shape[1]:
        raise ValueError('Las columnas numéricas contienen tipos no numéricos; revisar importación.')
    for col in CAT:
        df[col]=df[col].astype('string').str.strip().replace('',pd.NA).astype(object)
    no_columns=set(df.columns)-set([TARGET]+PREDICTORS+EXCLUDED)
    data_info={
        'rows_original':int(before),'rows_used':int(len(df)), 'columns_original':int(len(df.columns)),
        'duplicates_full':row_dupes,'duplicates_id':id_dupes,
        'missing_total':int(df.isna().sum().sum()),
        'missing_by_column':{k:int(v) for k,v in df.isna().sum().items()},
        'target':{str(k):int(v) for k,v in df[TARGET].value_counts().sort_index().items()},
        'target_rate':safe_float(df[TARGET].mean()),
        'dtypes':{k:str(v) for k,v in df.dtypes.items()},
        'predictors':PREDICTORS,'excluded':[TARGET]+EXCLUDED+sorted(no_columns),
        'variables_extra':sorted(no_columns),
        'file_name':csv_path.name,'dataset_is_synthetic':True}
    # División ANTES de cualquier exploración supervisada
    X=df[PREDICTORS].copy();y=df[TARGET].astype(int)
    Xtr,Xtest,ytr,ytest=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
    if ytr.value_counts().min()<FOLDS:
        raise ValueError('No hay suficientes registros para 5 folds estratificados.')
    data_info['train_n']=int(len(Xtr));data_info['test_n']=int(len(Xtest))
    data_info['train_positive']=int(ytr.sum());data_info['test_positive']=int(ytest.sum())
    eda=profile_train(Xtr,ytr,data_info)
    models=build_models()
    cv=StratifiedKFold(n_splits=FOLDS,shuffle=True,random_state=SEED)
    cv_splits=list(cv.split(Xtr,ytr))
    comparison=[]; model_results={}; folds_csv=[]
    if show_progress:print('CSV validado:',len(df),'registros /',len(PREDICTORS),'predictores',flush=True)
    for name,base_model in models.items():
        if show_progress:print('CV 5 folds:',name,flush=True)
        oof=np.full(len(Xtr),np.nan,dtype=float)
        fold_scores=[];fold_train_scores=[]
        for k,(idx_fit,idx_val) in enumerate(cv_splits,1):
            estimator=clone(base_model)
            estimator.fit(Xtr.iloc[idx_fit],ytr.iloc[idx_fit])
            p_val=estimator.predict_proba(Xtr.iloc[idx_val])[:,1]
            p_fit=estimator.predict_proba(Xtr.iloc[idx_fit])[:,1]
            oof[idx_val]=p_val
            scores=metrics_at(ytr.iloc[idx_val],p_val)
            fit_scores=metrics_at(ytr.iloc[idx_fit],p_fit)
            fold_scores.append(scores);fold_train_scores.append(fit_scores)
            folds_csv.append({'modelo':name,'fold':k,**scores})
            if show_progress:print(' ',k,'/ 5 · AP',round(scores['pr_auc'],4),flush=True)
        if np.isnan(oof).any():raise RuntimeError('Hay filas sin predicción OOF.')
        cv_means={key:safe_float(np.mean([m[key] for m in fold_scores])) for key in METRIC_NAMES}
        cv_stds={key:safe_float(np.std([m[key] for m in fold_scores],ddof=1)) for key in METRIC_NAMES}
        train_means={key:safe_float(np.mean([m[key] for m in fold_train_scores])) for key in METRIC_NAMES}
        comparison.append({'modelo':name,'cv_media':cv_means,'cv_desv':cv_stds,
                           'train_folds_media':train_means})
        model_results[name]={
            'cv_media':cv_means,'cv_desv':cv_stds,
            'train_folds_media':train_means,
            'folds':fold_scores,'oof_global':metrics_at(ytr,oof),
            'confusion_oof':matrix_at(ytr,oof),'curvas_oof':curves(ytr,oof),
            'umbrales_oof':thresholds(ytr,oof)
        }
    # Selección EXCLUSIVA por PR-AUC media CV; desempate Recall media CV.
    ranking=sorted(comparison,key=lambda v:(-v['cv_media']['pr_auc'],-v['cv_media']['recall'],v['modelo']))
    selected=ranking[0]['modelo']
    if show_progress:print('Modelo seleccionado SOLO CV:',selected,flush=True)
    # ÚNICA evaluación TEST: para el modelo elegido; jamás comparar modelos en TEST.
    final_model=clone(models[selected])
    final_model.fit(Xtr,ytr)
    ptest=final_model.predict_proba(Xtest)[:,1]
    test_metrics=metrics_at(ytest,ptest)
    final={'modelo':selected,'umbral_fijo':THRESHOLD,
           'metricas_test':test_metrics,'confusion_test':matrix_at(ytest,ptest),
           'curvas_test':curves(ytest,ptest),
           'cv_media':model_results[selected]['cv_media'],
           'cv_desv':model_results[selected]['cv_desv'],
           'train_folds_media':model_results[selected]['train_folds_media']}
    final['gap_ap_train_cv']=safe_float(final['train_folds_media']['pr_auc']-final['cv_media']['pr_auc'])
    final['gap_ap_cv_test']=safe_float(final['cv_media']['pr_auc']-test_metrics['pr_auc'])
    payload={
        'titulo':'CASO 48 · Predicción de incumplimiento crediticio',
        'origen':'UCE · Ingeniería en Sistemas de Información · Dataset sintético académico',
        'generado_utc':datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'versiones':{'python':platform.python_version(),'pandas':pd.__version__,'sklearn':sklearn.__version__},
        'metodologia':{'holdout':'80/20 estratificado','cv':'StratifiedKFold(5), shuffle=True, seed=42',
                       'seleccion':'Mayor PR-AUC (Average Precision) media CV; desempate Recall CV',
                       'umbral':THRESHOLD,'pr_auc':'Average Precision (AP), no área trapezoidal',
                       'evaluacion_test':'Solo una vez, exclusivamente modelo seleccionado',
                       'modelos':list(models)},
        'data':data_info,'eda_train':eda,'comparacion_cv':ranking,
        'modelos':model_results,'final':final,
        'advertencia':'Datos sintéticos; análisis didáctico. NO autoriza decisiones de crédito automáticas. Umbrales alternativos solo ilustrativos sobre TRAIN OOF.'}
    with (out_dir/'resultados_dashboard.json').open('w',encoding='utf-8') as f:
        json.dump(payload,f,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    pd.DataFrame(folds_csv).to_csv(out_dir/'metricas_por_fold.csv',index=False)
    pd.DataFrame([{'modelo':r['modelo'],**{k+'_media':r['cv_media'][k] for k in METRIC_NAMES},
                   **{k+'_std':r['cv_desv'][k] for k in METRIC_NAMES}}
                  for r in ranking]).to_csv(out_dir/'comparacion_cv.csv',index=False)
    pd.DataFrame([{'modelo':selected,**test_metrics}]).to_csv(out_dir/'metricas_test_FINAL.csv',index=False)
    pd.DataFrame([{'modelo':selected,'tipo':'CV (media)',**final['cv_media']},
                  {'modelo':selected,'tipo':'TRAIN interno (media)',**final['train_folds_media']},
                  {'modelo':selected,'tipo':'TEST final',**test_metrics}]).to_csv(
                  out_dir/'cv_vs_test.csv',index=False)
    if show_progress:
        print('TEST final —',test_metrics,flush=True)
        print('Resultados guardados en',out_dir.resolve(),flush=True)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True,help='CSV oficial de riesgo crediticio')
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'resultados')
    args=parser.parse_args()
    train(args.csv,args.out)
