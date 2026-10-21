from pathlib import Path
import shutil
import json
import csv
import subprocess
import sys
p=Path(__file__).resolve().parents[1]
folder=p/'resultados'; j=(folder/'resultados_dashboard.json').read_text(encoding='utf-8')
# Tabla de auditoría del punto 9: montos hipotéticos en USD calculados solo con TRAIN OOF.
obj=json.loads(j)
with (folder/'escenarios_costo_oof.csv').open('w',encoding='utf-8',newline='') as output:
    writer=csv.DictWriter(output,fieldnames=['modelo','umbral','fn','fp','costo_unitario_fn_usd_hipotetico','costo_unitario_fp_usd_hipotetico','costo_fn_usd_hipotetico','costo_fp_usd_hipotetico','costo_total_usd_hipotetico'])
    writer.writeheader()
    for name,rec in obj['modelos'].items():
        for entry in rec['umbrales_oof']:
            cm=entry['confusion']
            writer.writerow({'modelo':name,'umbral':entry['umbral'],'fn':cm['fn'],'fp':cm['fp'],
                'costo_unitario_fn_usd_hipotetico':500,'costo_unitario_fp_usd_hipotetico':100,
                'costo_fn_usd_hipotetico':cm['fn']*500,'costo_fp_usd_hipotetico':cm['fp']*100,
                'costo_total_usd_hipotetico':cm['fn']*500+cm['fp']*100})
for dst in [p/'dashboard']:
    dst.mkdir(parents=True,exist_ok=True)
    (dst/'datos.js').write_text('/* Estadísticas agregadas, sin filas */\nwindow.CASO48 = '+j+';\n',encoding='utf-8')
    for f in ['comparacion_cv.csv','metricas_por_fold.csv','metricas_test_FINAL.csv','cv_vs_test.csv','escenarios_costo_oof.csv']:
        shutil.copy2(folder/f,dst/f)
    (dst/'.nojekyll').touch()
subprocess.run([sys.executable, str(p/'sincronizar_dashboard.py')], check=True)
print('Datos agregados actualizados y docs/ sincronizada; verifica Git antes de publicar.')
