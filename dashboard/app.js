(() => {
'use strict';
const D = window.CASO48;
const app = document.querySelector('#app');
if(!D || !D.modelos || !D.final){app.innerHTML='<div class="wide-note">No se encuentra disponible el conjunto de resultados agregados (datos.js). El dashboard requiere este archivo en el mismo directorio que index.html.</div>';return;}
const names=D.metodologia.modelos;
const COLORS={'Regresión Logística':'#4187c4','Árbol de Decisión':'#e3a14e','Random Forest':'#9270b7','Gradient Boosting':'#118c80'};
const LABELS={'precision':'Precision','recall':'Recall','f1':'F1','roc_auc':'ROC-AUC','pr_auc':'PR-AUC (AP)','accuracy':'Accuracy'};
const FEATURES={'age':'Edad','monthly_income_usd':'Ingreso mensual (USD)','employment_years':'Años de empleo','credit_amount_usd':'Monto solicitado (USD)','term_months':'Plazo (meses)','debt_to_income_ratio':'Deuda / ingreso','credit_utilization':'Utilización crediticia','late_payments_12m':'Atrasos 12 meses','open_credits':'Créditos abiertos','savings_usd':'Ahorros (USD)','has_guarantor':'Tiene garante'};
let tab='resumen';let model=names[0];let feature='monthly_income_usd';let categorical='employment_type';let metric='pr_auc';let thr=50;let costoFN=500;let costoFP=100;
const fmt=(n,di=1)=>Number(n).toLocaleString('es-EC',{minimumFractionDigits:di,maximumFractionDigits:di});
const pct=(n,di=2)=>fmt(n*100,di)+' %';
const usd=(n)=>'US$ '+fmt(n,2);
const esc=t=>String(t).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const title=(tag,h1,sub,badge='RESULTADOS REALES')=>`<header class="heading"><div><p class="eyebrow">${tag}</p><h1>${h1}</h1><p class="subtitle">${sub}</p></div><span class="tag">${badge}</span></header>`;
const stat=(label,value,foot,icon='◇')=>`<div class="card stat"><div class="stat-kicker"><span class="stat-label">${label}</span><span class="symbol">${icon}</span></div><div class="stat-number">${value}</div><div class="stat-foot">${foot}</div></div>`;
const box=(h,sub,body)=>`<section class="card"><div class="card-head"><div><h2>${h}</h2>${sub?`<p>${sub}</p>`:''}</div></div>${body}</section>`;
const legend=(items)=>`<div class="chart-legend">${items.map(i=>`<span><b style="background:${i.color}"></b>${esc(i.label)}</span>`).join('')}</div>`;
const mlabel=(n)=>esc(n);
const color=(name)=>COLORS[name]||'#16897e';
function lineSVG(series,mode='roc',extra={}){
  const W=660,H=290,L=53,R=18,T=18,B=38,iw=W-L-R,ih=H-T-B;
  const xx=(n)=>L+Math.min(1,Math.max(0,n))*iw;
  const yy=(n)=>T+(1-Math.min(1,Math.max(0,n)))*ih;
  let markup='';
  for(let k=0;k<=5;k++){
    let t=k/5;
    markup+=`<line x1="${L}" y1="${yy(t)}" x2="${W-R}" y2="${yy(t)}" stroke="#e9eef2" stroke-width="1"/>`;
    markup+=`<text x="${L-10}" y="${yy(t)+4}" font-size="10" text-anchor="end" fill="#7c8c97">${Math.round(t*100)}%</text>`;
    markup+=`<text x="${xx(t)}" y="${H-B+21}" font-size="10" text-anchor="middle" fill="#7c8c97">${Math.round(t*100)}%</text>`;
  }
  markup+=`<path d="M${L},${T}V${H-B}H${W-R}" stroke="#bac9d1" stroke-width="1.2" fill="none"/>`;
  if(mode==='roc')markup+=`<path d="M${xx(0)},${yy(0)}L${xx(1)},${yy(1)}" stroke="#c7d3d8" stroke-width="1.5" stroke-dasharray="5,6"/>`;
  if(extra.baseline!==undefined)markup+=`<path d="M${xx(0)},${yy(extra.baseline)}L${xx(1)},${yy(extra.baseline)}" stroke="#b1bcc3" stroke-width="1.5" stroke-dasharray="6 5"/>`;
  for(const s of series){
    const points=s.points.filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));
    if(!points.length)continue;
    const d=points.map((p,i)=>(i?'L':'M')+xx(p[0]).toFixed(2)+','+yy(p[1]).toFixed(2)).join(' ');
    markup+=`<path d="${d}" stroke="${s.color}" stroke-width="${s.width||2.4}" fill="none" stroke-linejoin="round" stroke-linecap="round" opacity="${s.opacity||1}"/>`;
  }
  markup+=`<text x="${W/2}" y="${H-5}" font-size="11" text-anchor="middle" fill="#667b88">${mode==='roc'?'Tasa de falsos positivos (FPR)':'Recall (sensibilidad)'}</text>`;
  markup+=`<text transform="translate(12,${H/2}) rotate(-90)" font-size="11" text-anchor="middle" fill="#667b88">${mode==='roc'?'Tasa de verdaderos positivos (TPR)':'Precision'}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Curva ${mode==='roc'?'ROC':'Precisión Recall'} basada en datos agregados de validación cruzada">${markup}</svg>${legend(series.map(x=>({label:x.name,color:x.color})).concat(extra.baseline!==undefined?[{label:'Prevalencia TRAIN',color:'#b1bcc3'}]:[]))}`;
}
function metricBars(key){
 const max=Math.max(...names.map(n=>D.modelos[n].cv_media[key]),.001);
 return names.map(n=>`<div class="chart-bar"><div class="bar-name">${esc(n)}</div><div class="bar-rail"><div class="bar-fill" style="width:${100*D.modelos[n].cv_media[key]/max}%;background:${color(n)}"></div></div><div class="bar-value">${pct(D.modelos[n].cv_media[key])}</div></div>`).join('');
}
function confusion(mm){
  let tot=Object.values(mm).reduce((a,b)=>a+b,0);
  return `<div class="confusion"><div></div><div class="matrix-head">Predice 0</div><div class="matrix-head">Predice 1</div>
  <div class="matrix-label">Real 0</div><div class="matrix-cell"><strong>${fmt(mm.tn,0)}</strong><span>VN · ${pct(mm.tn/tot,1)}</span></div><div class="matrix-cell bad"><strong>${fmt(mm.fp,0)}</strong><span>FP · ${pct(mm.fp/tot,1)}</span></div>
  <div class="matrix-label">Real 1</div><div class="matrix-cell bad"><strong>${fmt(mm.fn,0)}</strong><span>FN · ${pct(mm.fn/tot,1)}</span></div><div class="matrix-cell"><strong>${fmt(mm.tp,0)}</strong><span>VP · ${pct(mm.tp/tot,1)}</span></div></div>`;
}
function metricsTable(){
 let rows=D.comparacion_cv.map((r,i)=>`<tr class="${i===0?'highlight':''}"><td><span class="model-tag" style="background:${color(r.modelo)}13;color:${color(r.modelo)}">${esc(r.modelo)}</span>${i===0?'<br /><small class="display-score">★ Seleccionado por CV</small>':''}</td>${['precision','recall','f1','roc_auc','pr_auc','accuracy'].map(k=>`<td class="numeric ${k==='pr_auc'?'bold':''}">${pct(r.cv_media[k])}</td>`).join('')}</tr>`).join('');
 return `<div class="table-wrap"><table><thead><tr><th>Algoritmo</th><th class="numeric">Precision</th><th class="numeric">Recall</th><th class="numeric">F1</th><th class="numeric">ROC-AUC</th><th class="numeric">PR-AUC (AP)</th><th class="numeric">Accuracy</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}
function histSVG(col){
 let h=D.eda_train.numeric[col].hist,zero=h.no_incumple,one=h.incumple;
 let N0=D.eda_train.target_counts['0'],N1=D.eda_train.target_counts['1'];
 let a=zero.map(n=>n/N0),b=one.map(n=>n/N1),max=Math.max(...a,...b)*1.15;
 const W=620,H=262,L=48,R=10,T=15,B=44,plotW=W-L-R,plotH=H-T-B,step=plotW/zero.length;
 let html='';
 for(let k=0;k<=4;k++){let v=max*k/4, yy=T+plotH*(1-k/4);html+=`<line x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}" stroke="#ecf1f3"/><text x="${L-6}" y="${yy+3}" font-size="10" text-anchor="end" fill="#8b98a2">${(v*100).toFixed(1)}%</text>`}
 for(let i=0;i<a.length;i++){
    let y0=T+plotH*(1-a[i]/max),y1=T+plotH*(1-b[i]/max);
    html+=`<rect x="${(L+i*step+step*.08).toFixed(2)}" y="${y0.toFixed(2)}" width="${(step*.4).toFixed(2)}" height="${(T+plotH-y0).toFixed(2)}" rx="1" fill="#1f9184"><title>${esc(FEATURES[col]||col)} ${fmt(h.centros[i],2)}: ${(a[i]*100).toFixed(2)}% no incumple</title></rect>`;
    html+=`<rect x="${(L+i*step+step*.50).toFixed(2)}" y="${y1.toFixed(2)}" width="${(step*.4).toFixed(2)}" height="${(T+plotH-y1).toFixed(2)}" rx="1" fill="#d99b54"><title>${esc(FEATURES[col]||col)} ${fmt(h.centros[i],2)}: ${(b[i]*100).toFixed(2)}% incumple</title></rect>`;
 }
 for(let k=0;k<=4;k++){let i=Math.min(h.centros.length-1,Math.floor(k*(h.centros.length-1)/4));html+=`<text x="${L+(i+.5)*step}" y="${H-B+18}" font-size="10" text-anchor="middle" fill="#7d8c97">${fmt(h.centros[i],col==='credit_utilization'||col==='debt_to_income_ratio'?2:0)}</text>`;}
 return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Histograma normalizado por clase"><title>Distribuciones normalizadas de ${esc(col)}</title>${html}</svg>${legend([{label:'No incumple · proporción dentro de clase',color:'#1f9184'},{label:'Incumple · proporción dentro de clase',color:'#d99b54'}])}`;
}
function categoriesTable(col){return `<div class="table-wrap"><table><thead><tr><th>Categoría</th><th class="numeric">Solicitantes TRAIN</th><th class="numeric">Tasa incumplimiento</th></tr></thead><tbody>${D.eda_train.categories[col].map(x=>`<tr><td>${esc(x.categoria)}</td><td class="numeric">${fmt(x.n,0)}</td><td class="numeric">${pct(x.tasa)}</td></tr>`).join('')}</tbody></table></div>`;}
function makeSummary(){let fin=D.final,chosen=fin.modelo,pre=D.data.target_rate;
return title('VISIÓN GENERAL','Observatorio de Riesgo Crediticio','Resultados verificables de cuatro algoritmos entrenados con 50.000 solicitudes sintéticas.','DATASET ANALIZADO')+
 `<div class="grid4">${stat('Solicitudes registradas',fmt(D.data.rows_used,0),'Base sintética de 16 variables','▦')}${stat('Tasa de incumplimiento',pct(pre),'Clase positiva (default_12m = 1)','↗')}${stat('Modelos comparados','04','Validación cruzada de 5 folds','◇')}${stat('PR-AUC seleccionado',pct(fin.cv_media.pr_auc),'Media CV sobre TRAIN, no sobre test','◈')}</div>`+
 `<div class="section grid2">${box('Desbalance de clases','Distribución del dataset completo, revisión inicial de calidad',`<div class="two-col-label"><span><b>${fmt(D.data.target['0'],0)}</b> sin incumplimiento</span><span><b>${fmt(D.data.target['1'],0)}</b> con incumplimiento</span></div><div class="stack" style="margin-top:14px"><div style="width:${(1-pre)*100}%;background:#1a9487"></div><div style="width:${pre*100}%;background:#dfa054"></div></div>${legend([{label:'No incumple',color:'#1a9487'},{label:'Incumple',color:'#dfa054'}])}<div class="split-line"></div><div class="wide-note"><b>Efecto del desbalance:</b> un clasificador que asignara todas las solicitudes a la clase «no incumple» alcanzaría ${pct(1-pre)} de accuracy, con Recall de 0 % para la clase de incumplimiento. La accuracy aislada no refleja la capacidad de detección.</div>`)}`+
 `${box('Modelo ganador de validación cruzada','Criterio predefinido: mayor PR-AUC (AP) media',`<p class="eyebrow">MODELO FINAL</p><h2 style="font-size:24px;letter-spacing:-.7px;margin:7px 0 12px">${esc(chosen)}</h2><div class="progress-lines"><label>PR-AUC CV</label><div><strong>${pct(fin.cv_media.pr_auc)}</strong><div class="bar-rail"><div class="bar-fill" style="background:#198d80;width:${100*fin.cv_media.pr_auc}%"></div></div></div><label>ROC-AUC CV</label><div><strong>${pct(fin.cv_media.roc_auc)}</strong><div class="bar-rail"><div class="bar-fill" style="background:#4187c4;width:${100*fin.cv_media.roc_auc}%"></div></div></div></div><p class="figure-caption">No se utilizó el conjunto TEST para seleccionar el algoritmo.</p>`)}</div>`+
 `<div class="section">${box('Comparación de los cuatro modelos','Resumen de métricas calculadas sobre los 5 folds de entrenamiento',metricsTable())}</div>`+
 `<div class="section wide-note"><strong>Resultado de la evaluación independiente:</strong> con el umbral de 0,50, el modelo seleccionado registró <b>${fin.confusion_test.tp}</b> verdaderos positivos y <b>${fin.confusion_test.fn}</b> falsos negativos en TEST. Debido al desbalance de clases, una accuracy de ${pct(fin.metricas_test.accuracy)} no implica una detección eficaz de incumplimientos. El análisis de sensibilidad del umbral se presenta por separado, exclusivamente con predicciones OOF de TRAIN.</div>`;}
function makeEDA(){
 const options=Object.keys(D.eda_train.numeric).map(c=>`<option value="${c}" ${feature===c?'selected':''}>${esc(FEATURES[c]||c)}</option>`).join('');
 const catOptions=Object.keys(D.eda_train.categories).map(c=>`<option value="${c}" ${categorical===c?'selected':''}>${esc(c)}</option>`).join('');
 const num=D.eda_train.numeric[feature];
 return title('DIAGNÓSTICO DE DATOS','Exploración de datos','Descripciones y gráficos elaborados exclusivamente sobre los 40.000 registros de TRAIN.','EDA SOLO EN TRAIN')+
 `<div class="grid4">${stat('Registros TRAIN',fmt(D.data.train_n,0),'80 % estratificado','▦')}${stat('Registros TEST',fmt(D.data.test_n,0),'Aislado durante entrenamiento','▦')}${stat('Valores faltantes',fmt(D.data.missing_total,0),'Dataset original completo','◇')}${stat('Duplicados completos',fmt(D.data.duplicates_full,0),'Control de calidad inicial','✓')}</div>`+
 `<div class="section grid2">${box('Distribución numérica por clase',`Histograma porcentual normalizado · <strong>TRAIN</strong>`,`<div class="controls"><label for="feature" class="control-label">VARIABLE</label><select class="select" id="feature">${options}</select></div><div class="chart-area">${histSVG(feature)}</div><div class="figure-caption">La normalización independiente por clase evita que la mayoría oculte a la minoría.</div>`)}`+
 `${box('Estadísticas descriptivas',esc(FEATURES[feature]||feature),`<div class="metric-grid">${[['Media',num.mean],['Mediana',num.median],['Desv. estándar',num.std],['Mínimo',num.min],['Máximo',num.max],['Atípicos IQR',num.outliers_iqr]].map(([label,v])=>`<div class="metric-mini"><small>${label}</small><strong>${fmt(v,label==='Atípicos IQR'?0:2)}</strong></div>`).join('')}</div><div class="wide-note" style="margin-top:14px"><b>Análisis de valores atípicos:</b> el rango intercuartílico (IQR) identifica observaciones potencialmente extremas, sin establecer que deban excluirse. Las transformaciones del preprocesamiento se ajustaron por separado dentro de cada fold de validación cruzada.</div>`)}</div>`+
 `<div class="section grid2">${box('Variables categóricas y clase objetivo','Frecuencia y tasa de incumplimiento en TRAIN',`<div class="controls"><label for="category" class="control-label">CATEGORÍA</label><select id="category" class="select">${catOptions}</select></div>${categoriesTable(categorical)}`)}`+
 `${box('Valores atípicos (IQR)','Conteo exploratorio, sin exclusión automática',`<div class="table-wrap"><table><thead><tr><th>Variable</th><th class="numeric">Casos</th><th class="numeric">% TRAIN</th></tr></thead><tbody>${D.eda_train.outliers.map(x=>`<tr><td>${esc(FEATURES[x.variable]||x.variable)}</td><td class="numeric">${fmt(x.n,0)}</td><td class="numeric">${fmt(x.porcentaje,2)} %</td></tr>`).join('')}</tbody></table></div>`)}</div>`;
}
function makeComparison(){
 let series=names.map(n=>({name:n,points:D.modelos[n].curvas_oof.roc,color:color(n),width:2.5}));
 let pr=names.map(n=>({name:n,points:D.modelos[n].curvas_oof.pr,color:color(n),width:2.5}));
 return title('EVALUACIÓN COMPARATIVA','Cuatro modelos, una metodología','Cada punto y cada curva se obtiene sin usar el conjunto de prueba durante la selección.','5-FOLD STRATIFIED CV')+
 `<div class="section">${box('Tabla comparativa por validación cruzada','Media de los 5 folds; PR-AUC equivale a Average Precision (AP)',metricsTable())}</div>`+
 `<div class="section grid2">${box('Curvas ROC · predicciones OOF','Cada muestra TRAIN fue predicha por un modelo que no la entrenó.',`<div class="chart-area">${lineSVG(series,'roc')}</div>`)}`+
 `${box('Curvas Precision–Recall · predicciones OOF','La línea discontinua señala la prevalencia positiva en TRAIN.',`<div class="chart-area">${lineSVG(pr,'pr',{baseline:D.eda_train.target_rate})}</div>`)}</div>`+
 `<div class="section grid2">${box('Comparación según la métrica','Comparación gráfica de las métricas obtenidas mediante validación cruzada.',`<div class="controls"><label for="metric" class="control-label">MÉTRICA</label><select id="metric" class="select">${Object.keys(LABELS).map(k=>`<option value="${k}" ${metric===k?'selected':''}>${LABELS[k]}</option>`).join('')}</select></div>${metricBars(metric)}<div class="figure-caption">Valores medios por fold. Estas barras no representan el resultado en TEST.</div>`)}`+
 `${box('Estabilidad entre los folds','Media y desviación estándar de la métrica seleccionada',`<div class="table-wrap"><table><thead><tr><th>Modelo</th><th class="numeric">Media</th><th class="numeric">Desv. estándar</th></tr></thead><tbody>${D.comparacion_cv.map(x=>`<tr><td>${esc(x.modelo)}</td><td class="numeric">${pct(x.cv_media[metric])}</td><td class="numeric">${pct(x.cv_desv[metric])}</td></tr>`).join('')}</tbody></table></div><div class="callout" style="margin-top:17px"><b>Interpretación de la variabilidad:</b> una desviación estándar reducida indica menor variación entre folds. Por sí sola, no demuestra capacidad de generalización hacia poblaciones externas.</div>`)}</div>`;
}
function getThresholdRecord(name,value){return D.modelos[name].umbrales_oof.reduce((best,current)=>Math.abs(current.umbral-value)<Math.abs(best.umbral-value)?current:best,D.modelos[name].umbrales_oof[0]);}
function makeModels(){
 const r=D.modelos[model],index=D.metodologia.modelos.indexOf(model),t=getThresholdRecord(model,thr/100);
 const modelOptions=names.map(n=>`<option value="${esc(n)}" ${model===n?'selected':''}>${esc(n)}</option>`).join('');
 return title('LABORATORIO INTERACTIVO','Análisis individual de modelos','Curvas, variabilidad, errores y escenarios de umbral calculados a partir de predicciones fuera de fold de TRAIN.','TEST NO INTERVIENE')+
 `<div class="section card"><div class="controls"><label for="modelo" class="control-label">ALGORITMO</label><select id="modelo" class="select">${modelOptions}</select><span class="pill ${model===D.final.modelo?'green':'sky'}">${model===D.final.modelo?'Modelo seleccionado':'Evaluado en CV'}</span></div></div>`+
 `<div class="section grid4">${stat('Precision CV',pct(r.cv_media.precision),'Media 5 folds','◈')}${stat('Recall CV',pct(r.cv_media.recall),'Media 5 folds','↗')}${stat('F1 CV',pct(r.cv_media.f1),'Media 5 folds','◇')}${stat('PR-AUC CV',pct(r.cv_media.pr_auc),'Average Precision','◉')}</div>`+
 `<div class="section grid2">${box('Curva ROC — '+esc(model),'Predicciones fuera de fold de TRAIN',lineSVG([{name:model,points:r.curvas_oof.roc,color:color(model)}],'roc'))}`+
 `${box('Curva Precision–Recall — '+esc(model),'Predicciones fuera de fold de TRAIN',lineSVG([{name:model,points:r.curvas_oof.pr,color:color(model)}],'pr',{baseline:D.eda_train.target_rate}))}</div>`+
 `<div class="section grid2">${box('Análisis de sensibilidad del umbral · TRAIN','Comparación de escenarios con predicciones OOF de entrenamiento; la selección del algoritmo y la evaluación TEST permanecen sin modificaciones.',`<div class="range-line"><span class="control-label">UMBRAL</span><input id="threshold" type="range" min="5" max="95" step="5" value="${thr}" aria-label="Umbral de clasificación"/><span class="range-val">${(thr/100).toFixed(2)}</span></div><div class="metric-grid" style="margin:20px 0"><div class="metric-mini"><small>Precision</small><strong>${pct(t.precision)}</strong></div><div class="metric-mini"><small>Recall</small><strong>${pct(t.recall)}</strong></div><div class="metric-mini"><small>F1</small><strong>${pct(t.f1)}</strong></div></div>${confusion(t.confusion)}<p class="figure-caption">La matriz corresponde a predicciones OOF sobre TRAIN. El umbral empleado en TEST permanece fijo en 0,50.</p>`)}`+
 `${box('Desempeño por fold · '+esc(model),'Variación en PR-AUC, Recall y ROC-AUC',`<div class="table-wrap"><table><thead><tr><th>Fold</th><th class="numeric">Precision</th><th class="numeric">Recall</th><th class="numeric">F1</th><th class="numeric">PR-AUC</th><th class="numeric">ROC-AUC</th></tr></thead><tbody>${r.folds.map((f,i)=>`<tr><td>Fold ${i+1}</td>${['precision','recall','f1','pr_auc','roc_auc'].map(k=>`<td class="numeric">${pct(f[k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div><div class="callout" style="margin-top:18px"><strong>Significado de los errores:</strong> los falsos negativos corresponden a incumplimientos no detectados; los falsos positivos, a solicitantes sin incumplimiento clasificados como riesgosos. El enunciado no proporciona costos económicos observados para ninguno de los dos tipos de error.</div>`)}</div>`;
}

function pesoSeguro(valor,anterior){
 const n=Number(valor);
 return Number.isFinite(n)&&n>=0&&n<=1000000&&String(valor).trim()!==''?n:anterior;
}
function costoTotal(c){return c.fn*costoFN+c.fp*costoFP;}
function costoSVG(){
 const W=700,H=305,L=64,R=22,T=26,B=48,iw=W-L-R,ih=H-T-B;
 const rows=names.map(n=>({name:n,points:D.modelos[n].umbrales_oof.map(x=>[x.umbral,costoTotal(x.confusion)])}));
 const vmax=Math.max(1,...rows.flatMap(r=>r.points.map(p=>p[1])))*1.08;
 const xx=v=>L+(v-.05)/.90*iw,yy=v=>T+(1-v/vmax)*ih;
 let svg='';
 for(let k=0;k<=4;k++){
  const v=vmax*k/4,y=yy(v);
  svg+=`<line x1="${L}" y1="${y.toFixed(2)}" x2="${W-R}" y2="${y.toFixed(2)}" stroke="#e6edf1"/><text x="${L-9}" y="${(y+4).toFixed(2)}" fill="#718692" font-size="11" text-anchor="end">${usd(v)}</text>`;
 }
 for(const x of [.05,.20,.35,.50,.65,.80,.95])svg+=`<text x="${xx(x)}" y="${H-B+20}" font-size="11" text-anchor="middle" fill="#728492">${x.toFixed(2)}</text>`;
 svg+=`<path d="M${L} ${T}V${H-B}H${W-R}" fill="none" stroke="#b8c9d0" stroke-width="1.3"/>`;
 svg+=`<line x1="${xx(thr/100)}" x2="${xx(thr/100)}" y1="${T}" y2="${H-B}" stroke="#607486" stroke-dasharray="5 5"/>`;
 for(const row of rows){
  const pts=row.points.map((p,i)=>(i?'L':'M')+xx(p[0]).toFixed(2)+','+yy(p[1]).toFixed(2)).join(' ');
  svg+=`<path d="${pts}" fill="none" stroke="${color(row.name)}" stroke-width="2.8" stroke-linejoin="round"/>`;
  const current=getThresholdRecord(row.name,thr/100);
  svg+=`<circle cx="${xx(current.umbral)}" cy="${yy(costoTotal(current.confusion))}" r="4" fill="${color(row.name)}" stroke="white" stroke-width="1.4"/>`;
 }
 svg+=`<text x="${W/2}" y="${H-4}" text-anchor="middle" font-size="12" fill="#5b7482">Umbral aplicado a predicciones OOF · solo TRAIN</text>`;
 svg+=`<text x="${L}" y="${T-10}" font-size="11" fill="#5b7482">Costo hipotético (USD)</text>`;
 return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Costo hipotético versus umbral en validación cruzada de cuatro modelos sobre TRAIN">${svg}</svg>${legend(rows.map(r=>({label:r.name,color:color(r.name)})))}`;
}
function makeCosts(){
 const cutoff=thr/100;
 const base=D.eda_train.target_counts['1']*costoFN;
 const values=names.map(n=>{
  const t=getThresholdRecord(n,cutoff),cm=t.confusion;
  return {name:n,cm,loss:costoTotal(cm),fnLoss:cm.fn*costoFN,fpLoss:cm.fp*costoFP,precision:t.precision,recall:t.recall};
 });
 const chosen=values.find(x=>x.name===model)||values[0];
 const winner=values.find(x=>x.name===D.final.modelo);
 const bestMin=Math.min(...values.map(x=>x.loss));
 return title('INSTRUCCIÓN 9 · FALSOS POSITIVOS Y NEGATIVOS','Análisis de costos de clasificación',
  'Comparación de escenarios de costos hipotéticos en dólares estadounidenses (USD), calculados con predicciones OOF de TRAIN y sin participación de TEST.','USD HIPOTÉTICOS · SOLO TRAIN')+
 `<div class="section alert-blue"><b>Criterio de selección del algoritmo:</b> ${esc(D.final.modelo)} obtuvo la mayor PR-AUC (Average Precision) media en los cinco folds de TRAIN. La comparación económica es un análisis complementario basado en costos hipotéticos y predicciones OOF de entrenamiento. No representa pérdidas financieras observadas ni modifica la selección establecida mediante validación cruzada.</div>`+
 `<div class="section grid2">${box('Consecuencias potenciales de los errores','El modelo predice 1 = incumple y 0 = no incumple.',`<div class="cost-explain"><div><span class="cost-symbol fn">FN</span><b>Falso negativo: incumplimiento no detectado</b><p>Se etiqueta como bajo riesgo a alguien que posteriormente incumple. Podrían producirse pérdidas por impago y gastos de recuperación.</p></div><div><span class="cost-symbol fp">FP</span><b>Falso positivo: buen pagador señalado</b><p>Se clasifica como riesgosa a una persona que sí pagaría. Puede originar una revisión innecesaria, demoras o un rechazo injustificado si se utilizara la predicción indebidamente.</p></div></div>`)}`+
 `${box('Parámetros de costos hipotéticos (USD)','Importes supuestos para analizar escenarios; no corresponden a pérdidas observadas ni a valores contenidos en el CSV.',`<div class="cost-inputs"><label for="cost-fn"><b>Costo por falso negativo (FN) · USD</b><span>Riesgo que no se detectó</span><input id="cost-fn" type="number" min="0" max="1000000" step="0.01" value="${costoFN}" inputmode="decimal"/></label><label for="cost-fp"><b>Costo por falso positivo (FP) · USD</b><span>Revisión o rechazo innecesario</span><input id="cost-fp" type="number" min="0" max="1000000" step="0.01" value="${costoFP}" inputmode="decimal"/></label></div><div class="callout"><b>Fórmula:</b> Costo total (USD) = (FN × ${usd(costoFN)}) + (FP × ${usd(costoFP)}). La formulación simplificada asigna costo cero a los verdaderos positivos y verdaderos negativos.</div><p class="figure-caption"><b>Supuestos iniciales:</b> FN = US$ 500 y FP = US$ 100 por error. Los importes pueden modificarse, incluso con decimales, para comparar escenarios. El enunciado no especifica pérdidas reales ni costos de oportunidad.</p>`)}</div>`+
 `<div class="section card"><div class="controls"><label for="cost-threshold" class="control-label">UMBRAL OOF</label><input id="cost-threshold" type="range" min="5" max="95" step="5" value="${thr}" aria-label="Umbral de clasificación para comparar escenarios de costos"/><span class="range-val">${cutoff.toFixed(2)}</span><label for="cost-model" class="control-label">DESTACAR MODELO</label><select id="cost-model" class="select">${names.map(n=>`<option value="${esc(n)}" ${model===n?'selected':''}>${esc(n)}</option>`).join('')}</select></div><p class="figure-caption">Las variaciones del umbral se aplican únicamente a escenarios OOF de TRAIN. La evaluación final de TEST conserva su umbral predefinido de 0,50.</p></div>`+
 `<div class="section grid4">${stat('Costo total · '+esc(model),usd(chosen.loss),'C_FN × FN + C_FP × FP','▦')}${stat('Costo por FN',usd(chosen.fnLoss),`${fmt(chosen.cm.fn,0)} falsos negativos`,'↘')}${stat('Costo por FP',usd(chosen.fpLoss),`${fmt(chosen.cm.fp,0)} falsos positivos`,'↗')}${stat('Referencia: predecir siempre 0',usd(base),'Solo FN; comparador trivial','◇')}</div>`+
 `<div class="section grid2">${box('Comparación de los cuatro modelos','Mismo umbral y mismos pesos; 40.000 predicciones OOF de TRAIN.',`<div class="table-wrap"><table><thead><tr><th>Modelo</th><th class="numeric">FN</th><th class="numeric">FP</th><th class="numeric">FN × costo</th><th class="numeric">FP × costo</th><th class="numeric">Costo total (USD)</th></tr></thead><tbody>${values.map(x=>`<tr class="${x.name===D.final.modelo?'highlight':''}"><td>${esc(x.name)}${x.name===D.final.modelo?' ★':''}</td><td class="numeric">${fmt(x.cm.fn,0)}</td><td class="numeric">${fmt(x.cm.fp,0)}</td><td class="numeric">${usd(x.fnLoss)}</td><td class="numeric">${usd(x.fpLoss)}</td><td class="numeric bold">${usd(x.loss)}</td></tr>`).join('')}</tbody></table></div><p class="figure-caption">★ Modelo seleccionado por AP en CV (no por el costo mostrado). El mínimo del escenario es <b>${usd(bestMin)}</b>, sin que ello suponga una nueva selección oficial.</p>`)}`+
 `${box('Gráfico: costo por modelo','Cada barra representa el costo hipotético en USD calculado con matrices de TRAIN OOF.',`<div class="cost-bar-chart">${values.map(v=>`<div class="chart-bar"><div class="bar-name">${esc(v.name)}</div><div class="bar-rail"><div class="bar-fill" style="width:${100*v.loss/Math.max(1,...values.map(a=>a.loss))}%;background:${color(v.name)}"></div></div><div class="bar-value">${usd(v.loss)}</div></div>`).join('')}</div><div class="callout">El modelo destacado presenta <b>${fmt(chosen.cm.fn,0)} FN</b> y <b>${fmt(chosen.cm.fp,0)} FP</b>. La diferencia respecto al clasificador de referencia es <b>${usd(chosen.loss-base)}</b> (${chosen.loss>base?'mayor costo':chosen.loss<base?'menor costo':'igual costo'}), calculada únicamente con supuestos monetarios. No equivale a un ahorro o una pérdida financiera observada.</div>`)}</div>`+
 `<div class="section grid2">${box('Costo frente al umbral · cuatro modelos','Escenarios calculados exclusivamente sobre predicciones OOF TRAIN',costoSVG())}`+
 `${box('Matriz de confusión · '+esc(model),'Solo entrenamiento: mismas 40.000 filas predichas fuera de fold',`${confusion(chosen.cm)}<div class="metric-grid" style="margin-top:18px"><div class="metric-mini"><small>Recall al umbral</small><strong>${pct(chosen.recall)}</strong></div><div class="metric-mini"><small>Precision al umbral</small><strong>${pct(chosen.precision)}</strong></div></div>`)}</div>`+
 `<div class="section"><div class="download-row"><a href="escenarios_costo_oof.csv" download>↧ Descargar costos simulados en USD (.csv)</a></div></div>`+
 `<div class="section wide-note"><strong>Interpretación del análisis de costos (punto 9):</strong> Los valores en USD corresponden a supuestos de simulación y no a pérdidas financieras comprobadas. Los falsos negativos pueden representar incumplimientos no identificados, mientras que los falsos positivos pueden ocasionar revisiones innecesarias de solicitantes que cumplirían sus pagos. La ponderación relativa de ambos errores afecta el costo estimado y la relación entre Precision y Recall. <b>En ausencia de costos reales, no es posible establecer un modelo o umbral económicamente óptimo.</b> La selección del algoritmo se basa exclusivamente en validación cruzada sobre TRAIN y la evaluación final de TEST no interviene en el análisis de escenarios.</div>`;
}
function makeTest(){let f=D.final,c=f.confusion_test,r=f.metricas_test;
return title('EVALUACIÓN INDEPENDIENTE','Prueba final del modelo','Única evaluación TEST después de seleccionar el algoritmo con validación cruzada en TRAIN.','TEST AISLADO · 20 %')+
 `<div class="card"><div class="two-col-label"><div><span class="eyebrow">MODELO CONGELADO</span><h2 class="space-0" style="font-size:25px">${esc(f.modelo)}</h2><p class="small">Se reentrenó con todos los ${fmt(D.data.train_n,0)} ejemplos de TRAIN.</p></div><span class="pill green">Umbral fijo: ${fmt(f.umbral_fijo,2)}</span></div></div>`+
 `<div class="section grid4">${stat('PR-AUC final',pct(r.pr_auc),'Average Precision · TEST','◈')}${stat('ROC-AUC final',pct(r.roc_auc),'Discriminación · TEST','◇')}${stat('Recall final',pct(r.recall),'Detección de positivos · TEST','↗')}${stat('Accuracy final',pct(r.accuracy),'Métrica complementaria','▦')}</div>`+
 `<div class="section grid2">${box('Matriz de confusión · TEST','Valores absolutos, umbral 0,50',`${confusion(c)}<div class="figure-caption" style="text-align:center">TEST: ${fmt(D.data.test_n,0)} registros, ${fmt(D.data.test_positive,0)} incumplimientos reales.</div>`)}`+
 `${box('Comparación CV vs TEST','Se contrasta el modelo seleccionado sin reabrir la selección.',`<div class="table-wrap"><table><thead><tr><th>Métrica</th><th class="numeric">CV media</th><th class="numeric">TEST</th><th class="numeric">Brecha</th></tr></thead><tbody>${Object.keys(LABELS).map(k=>`<tr><td>${LABELS[k]}</td><td class="numeric">${pct(f.cv_media[k])}</td><td class="numeric">${pct(r[k])}</td><td class="numeric">${pct(f.cv_media[k]-r[k])}</td></tr>`).join('')}</tbody></table></div><div class="callout" style="margin-top:16px">Brecha PR-AUC train-interno menos CV: <b>${pct(f.gap_ap_train_cv)}</b>. Brecha PR-AUC CV menos TEST: <b>${pct(f.gap_ap_cv_test)}</b>. Son indicios descriptivos, no pruebas absolutas de sobreajuste.</div>`)}</div>`+
 `<div class="section grid2">${box('ROC en TEST','Una sola curva del modelo previamente elegido',lineSVG([{name:'TEST final · '+f.modelo,points:f.curvas_test.roc,color:'#168e83'}],'roc'))}`+
 `${box('Precision–Recall en TEST','Referencia de prevalencia del TRAIN',lineSVG([{name:'TEST final · '+f.modelo,points:f.curvas_test.pr,color:'#168e83'}],'pr',{baseline:D.eda_train.target_rate}))}</div>`+
 `<div class="section wide-note"><strong>Interpretación de los resultados de TEST:</strong> con umbral 0,50 se registraron <b>${c.fn} falsos negativos</b> y <b>${c.fp} falsos positivos</b>. ${c.tp===0?'No se identificaron verdaderos positivos, por lo que Recall y F1 resultaron 0 %. La accuracy elevada refleja principalmente la prevalencia de la clase mayoritaria y no evidencia una detección efectiva de los incumplimientos.':'Ambos tipos de error contribuyen a la evaluación del desempeño, más allá de la accuracy.'} En aplicaciones reales haría falta calibración, evaluación de costos, validación temporal y supervisión humana.</div>`;
}
function makeMethod(){
 let steps=[
  ['01','Control de calidad del CSV','Dimensiones, tipos, duplicados y ausentes; no se detectaron valores faltantes ni duplicados.'],
  ['02','Separación previa a EDA supervisada','80 % entrenamiento y 20 % prueba con estratificación; las gráficas exploratorias se basan en TRAIN.'],
  ['03','Variables predictoras y objetivo','14 predictores; se excluye applicant_id y la variable objetivo default_12m de X.'],
  ['04','Preprocesamiento dentro del Pipeline','Mediana para numéricas, imputación de moda y One-Hot para categóricas; StandardScaler en regresión logística.'],
  ['05','Cuatro algoritmos','Regresión Logística, Árbol de Decisión, Random Forest y Gradient Boosting optimizado por histogramas.'],
  ['06','StratifiedKFold de 5 folds','Se entrenan todas las transformaciones dentro de cada fold de TRAIN; predicciones out-of-fold para curvas y matrices.'],
  ['07','Selección por PR-AUC media CV','Desempate por Recall de CV, sin usar los resultados del conjunto TEST.'],
  ['08','Reentrenamiento y test una vez','Se vuelve a ajustar el pipeline seleccionado con TRAIN completo y se evalúa una sola vez sobre TEST.'],
  ['09','Costos hipotéticos en USD · punto 9','C = FN × C_FN(USD) + FP × C_FP(USD), con matrices OOF TRAIN; los importes son supuestos, no pérdidas reales.'],
  ['10','Interpretación y limitaciones','Precision, Recall, F1, ROC-AUC, Average Precision, falsos positivos/negativos, estabilidad y sobreajuste.']
 ];
 return title('TRAZABILIDAD ACADÉMICA','Metodología y reproducibilidad','Las decisiones del experimento siguen las 12 instrucciones del enunciado oficial.','ESTUDIO ACADÉMICO')+
 `<div class="grid2">${box('Secuencia del experimento','Diseño preventivo contra data leakage',steps.map(s=>`<div class="stage"><div class="step-index">${s[0]}</div><div><strong>${s[1]}</strong><p>${s[2]}</p></div></div>`).join(''))}`+
 `${box('Ficha técnica del experimento','Configuración y alcance',`<div class="metric-grid"><div class="metric-mini"><small>Train</small><strong>80 %</strong></div><div class="metric-mini"><small>Test</small><strong>20 %</strong></div><div class="metric-mini"><small>Folds</small><strong>5</strong></div><div class="metric-mini"><small>Semilla</small><strong>42</strong></div><div class="metric-mini"><small>Umbral</small><strong>0,50</strong></div><div class="metric-mini"><small>Modelos</small><strong>4</strong></div></div><div class="split-line"></div><h3 class="section-title">Criterios de evaluación</h3><p class="small">Se reportan Precision, Recall, F1, ROC-AUC, PR-AUC (Average Precision) y, como complemento, Accuracy. Se utiliza PR-AUC como criterio principal por el desbalance de clases.</p><h3 class="section-title">Alcance y limitaciones</h3><p class="small">Dataset sintético con fines didácticos; no hay prueba temporal o externa. La exploración de umbrales se restringe a TRAIN con predicciones OOF. Los montos en USD del simulador son hipotéticos porque los costos reales de FP y FN no son proporcionados. El dashboard no entrena modelos ni procesa nuevos solicitantes: presenta resultados agregados ya calculados.</p><div class="wide-note"><b>Ámbito de aplicación:</b> el clasificador corresponde a un ejercicio académico de minería de datos. Sus predicciones no constituyen decisiones automáticas de aprobación o rechazo de solicitudes de crédito.</div>`)}</div>`+
 `<div class="section">${box('Descargar las métricas reproducibles','Archivos agregados sin identificadores de solicitantes',`<div class="download-row"><a href="comparacion_cv.csv" download>↧ Comparación CV (.csv)</a><a href="metricas_por_fold.csv" download>↧ Métricas por fold (.csv)</a><a href="metricas_test_FINAL.csv" download>↧ Métricas finales (.csv)</a><a href="cv_vs_test.csv" download>↧ CV versus TEST (.csv)</a><a href="escenarios_costo_oof.csv" download>↧ Costos hipotéticos USD OOF (.csv)</a></div><p class="figure-caption">El proceso de entrenamiento y la configuración de algoritmos se encuentran documentados en el código Python del proyecto.</p>`)}</div>`;
}
const views={resumen:makeSummary,exploracion:makeEDA,comparacion:makeComparison,modelos:makeModels,costos:makeCosts,test:makeTest,metodo:makeMethod};
function render(){app.innerHTML=views[tab]();document.querySelectorAll('.nav-link').forEach(el=>{let is=el.dataset.tab===tab;el.classList.toggle('active',is);el.setAttribute('aria-current',is?'page':'false')});}
document.querySelector('#navigation').addEventListener('click',event=>{let btn=event.target.closest('[data-tab]');if(!btn)return;tab=btn.dataset.tab;render();window.scrollTo(0,0);});
app.addEventListener('change',event=>{
 const el=event.target;
 if(el.id==='modelo'){model=el.value;thr=50;render();}
 if(el.id==='feature'){feature=el.value;render();}
 if(el.id==='category'){categorical=el.value;render();}
 if(el.id==='metric'){metric=el.value;render();}
 if(el.id==='threshold'||el.id==='cost-threshold'){thr=Number(el.value);render();}
 if(el.id==='cost-model'){model=el.value;render();}
 if(el.id==='cost-fn'){costoFN=pesoSeguro(el.value,costoFN);render();}
 if(el.id==='cost-fp'){costoFP=pesoSeguro(el.value,costoFP);render();}
});
app.addEventListener('input',event=>{if(event.target.id==='threshold'||event.target.id==='cost-threshold'){const id=event.target.id;thr=Number(event.target.value);render();let el=document.getElementById(id);if(el)el.focus({preventScroll:true});}});
render();
})();
