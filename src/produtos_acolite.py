"""Identificação sem colisões e leitura integral de produtos rhos."""
from pathlib import Path


def product_key(scene_id, known_scene_ids):
    """Mantém pastas legadas somente se a chave curta não for ambígua."""
    key = scene_id.split("_")[1]
    matches = {sid for sid in known_scene_ids if sid.split("_")[1] == key}
    if len(matches) > 1:
        raise ValueError(f"Chave ACOLITE legada ambígua: {key}; use pastas com scene_id completo")
    return key


def product_folder(root, scene_id, known_scene_ids):
    root = Path(root)
    full = root / scene_id
    if full.is_dir():
        return full, scene_id
    key = product_key(scene_id, known_scene_ids)
    return root / key, key


def validate_bands(folder, key, wavelengths, full_read=False):
    """Valida grade, banda única e, opcionalmente, todos os blocos comprimidos.

    Integridade de leitura não comprova proveniência ou qualidade radiométrica.
    Ausentes retornam False; produtos existentes inconsistentes geram erro.
    """
    import rasterio
    paths = [Path(folder) / f"{key}_rhos_{wave}.tif" for wave in wavelengths]
    if not all(p.is_file() for p in paths):
        return False
    reference = None
    for path in paths:
        with rasterio.open(path) as src:
            grid = (src.crs, src.transform, src.width, src.height)
            if src.count != 1 or src.crs is None or src.width < 1 or src.height < 1:
                raise ValueError(f"Produto sem grade/banda válida: {path}")
            if reference is not None and grid != reference:
                raise ValueError(f"Grade de banda incompatível: {path}")
            reference = grid
            if full_read:
                for _, window in src.block_windows(1):
                    src.read(1, window=window)
    return True
