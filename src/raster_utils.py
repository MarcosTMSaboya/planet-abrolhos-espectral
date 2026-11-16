"""Leitura de pixels com verificação de limites e máscaras NoData."""
import numpy as np
from rasterio.warp import transform


class OutsideRasterError(ValueError):
    """O ponto não pertence à extensão de pixels do raster."""


def sample_pixel(src, lon, lat):
    if src.crs is None:
        raise ValueError("O raster não possui CRS definido.")
    if not np.isfinite([lon, lat]).all():
        raise ValueError("Coordenadas não finitas.")
    xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
    x, y = xs[0], ys[0]
    if not np.isfinite([x, y]).all():
        raise ValueError("Não foi possível transformar as coordenadas do ponto.")
    # Na grade norte-acima, direita e base são limites exclusivos.
    # Verificar também as coordenadas evita arredondamento de index() na borda.
    if src.transform.b == 0 and src.transform.d == 0 and src.transform.a > 0 and src.transform.e < 0:
        if not (src.bounds.left <= x < src.bounds.right and src.bounds.bottom < y <= src.bounds.top):
            raise OutsideRasterError("O centroide está fora da área do raster selecionado.")
    row, col = src.index(x, y)
    if not (0 <= row < src.height and 0 <= col < src.width):
        raise OutsideRasterError("O centroide está fora da área do raster selecionado.")
    values = next(src.sample([(x, y)], masked=True))
    values = np.ma.asarray(values, dtype="float64").filled(np.nan)
    values[~np.isfinite(values) | (np.abs(values) >= 1e20)] = np.nan
    return values
