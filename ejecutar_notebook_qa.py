from pathlib import Path
from nbclient import NotebookClient
import nbformat
p=Path(__file__).parent
nb=nbformat.read(p/'notebooks/Caso_48_Paso_a_Paso.ipynb',as_version=4)
client=NotebookClient(nb,timeout=500,resources={'metadata':{'path':str(p/'notebooks')}},allow_errors=False)
client.execute()
print('NOTEBOOK COMPLETO: OK. Ejecutadas',len(nb.cells),'celdas')
