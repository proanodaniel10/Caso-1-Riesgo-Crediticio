from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FUENTE = ROOT / 'dashboard'
DESTINO = ROOT / 'docs'
MANIFIESTO = ROOT / '.archivos_dashboard_sincronizados.json'

# Limita lo que se copia a recursos públicos típicos de una web estática.
EXTENSIONES_PUBLICAS = {
    '.html', '.css', '.js', '.json', '.svg', '.png', '.jpg', '.jpeg',
    '.webp', '.gif', '.ico', '.woff', '.woff2', '.webmanifest', '.map',
}
# Archivos CSV AGREGADOS de resultados, no solicitudes originales.
CSVS_PUBLICOS = {
    'comparacion_cv.csv',
    'cv_vs_test.csv',
    'metricas_por_fold.csv',
    'metricas_test_FINAL.csv',
    'escenarios_costo_oof.csv',
}
ARCHIVOS_ESPECIALES = {'.nojekyll'}
CARPETAS_EXCLUIDAS = {
    '.git', '.venv', 'venv', '__pycache__', 'node_modules',
    'data', 'resultados', 'private', 'privado',
}


def archivo_permitido(relative: Path) -> bool:
    if any(p.startswith('.') and p != '.nojekyll' for p in relative.parts):
        return False
    if any(part.casefold() in CARPETAS_EXCLUIDAS for part in relative.parts[:-1]):
        return False
    if relative.name.casefold() in {'.env', '.env.local', 'credentials.json'}:
        return False
    if relative.name in ARCHIVOS_ESPECIALES:
        return True
    if relative.suffix.lower() == '.csv':
        return relative.parent == Path('.') and relative.name in CSVS_PUBLICOS
    return relative.suffix.lower() in EXTENSIONES_PUBLICAS


def huella(archivo: Path) -> str:
    digest = hashlib.sha256()
    with archivo.open('rb') as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(bloque)
    return digest.hexdigest()


def archivos_dashboard() -> dict[str, Path]:
    archivos: dict[str, Path] = {}
    for item in FUENTE.rglob('*'):
        if item.is_symlink():
            continue  # Evita copiar objetivos fuera del dashboard por enlaces simbólicos.
        if not item.is_file():
            continue
        rel = item.relative_to(FUENTE)
        if archivo_permitido(rel):
            archivos[rel.as_posix()] = item
    return archivos


def cargar_manifest() -> set[str]:
    if not MANIFIESTO.exists():
        return set()
    try:
        data = json.loads(MANIFIESTO.read_text(encoding='utf-8'))
        if data.get('version') != 1 or not isinstance(data.get('archivos'), list):
            raise ValueError('formato de manifiesto no reconocido')
        return {str(p) for p in data['archivos'] if isinstance(p, str)}
    except (json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f'Manifiesto de sincronización inválido: {error}') from error


def destino_de(relative: str) -> Path:
    rel = Path(relative)
    # Segunda barrera para manifiestos manipulados y rutas fuera de docs/.
    if rel.is_absolute() or '..' in rel.parts or not archivo_permitido(rel):
        raise ValueError(f'Ruta no permitida en manifiesto: {relative!r}')
    return DESTINO / rel


def sincronizar(modo: str) -> int:
    if not FUENTE.is_dir():
        print(f'ERROR: No existe la carpeta de origen: {FUENTE}', file=sys.stderr)
        return 2
    if not (FUENTE / 'index.html').is_file():
        print('ERROR: Falta dashboard/index.html; no se sincronizará una web incompleta.', file=sys.stderr)
        return 2

    actuales = archivos_dashboard()
    previos = cargar_manifest()
    actualizados = []
    iguales = []
    for relative, origen in sorted(actuales.items()):
        destino = destino_de(relative)
        if destino.is_symlink():
            print(f'ERROR: docs contiene un enlace simbólico: {destino}', file=sys.stderr)
            return 2
        if destino.is_file() and huella(origen) == huella(destino):
            iguales.append(relative)
        else:
            actualizados.append(relative)

    # Solo se borran archivos gestionados previamente y que desaparecieron
    # del origen. Se conservan CNAME y cualquier otro archivo añadido a docs/.
    eliminables = sorted(previos - actuales.keys())
    eliminables = [f for f in eliminables if destino_de(f).exists()]
    if not actualizados and not eliminables:
        print(f'Listo: docs/ ya coincide con dashboard/ ({len(iguales)} archivos revisados).')
        if modo == 'ejecutar' and not MANIFIESTO.exists():
            MANIFIESTO.write_text(json.dumps({
                'version': 1, 'archivos': sorted(actuales),
            }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        return 0

    print('Cambios de dashboard/ a docs/:')
    for f in actualizados:
        print('  COPIAR: ', f)
    for f in eliminables:
        print('  ELIMINAR (archivo antes sincronizado):', f)
    print(f'Resumen: {len(actualizados)} por copiar, {len(eliminables)} por eliminar, {len(iguales)} sin cambios.')

    if modo == 'verificar':
        print('docs/ está desactualizada. Ejecuta: python sincronizar_dashboard.py')
        return 1
    if modo == 'simular':
        print('Simulación: no se modificó ningún archivo.')
        return 0

    DESTINO.mkdir(parents=True, exist_ok=True)
    for relative in actualizados:
        destino = destino_de(relative)
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(actuales[relative], destino)
    for relative in eliminables:
        archivo = destino_de(relative)
        archivo.unlink()
        # Limpiar solo directorios vacíos, sin tocar contenido exclusivo de docs/.
        parent = archivo.parent
        while parent != DESTINO:
            if any(parent.iterdir()):
                break
            parent.rmdir()
            parent = parent.parent

    MANIFIESTO.write_text(json.dumps({
        'version': 1, 'archivos': sorted(actuales),
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Sincronización finalizada. El siguiente paso es Commit + Push a GitHub.')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Actualiza docs/ desde dashboard/ para GitHub Pages.')
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument('--verificar', action='store_true', help='Comprueba que docs/ esté actualizada (código 1 si difiere).')
    grupo.add_argument('--simular', action='store_true', help='Muestra cambios sin copiar archivos.')
    args = parser.parse_args()
    modo = 'verificar' if args.verificar else 'simular' if args.simular else 'ejecutar'
    try:
        return sincronizar(modo)
    except (OSError, RuntimeError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
