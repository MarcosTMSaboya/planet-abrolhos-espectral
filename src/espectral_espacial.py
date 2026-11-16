"""Diagnóstico e separação espacial para o piloto espectral.

Funções puras: não leem rasters, não ajustam PCA/K-means e não atribuem
identidade ecológica. As distâncias e correlações são diagnósticos; nem o
buffer nem a remoção de uma tendência linear demonstram independência.
"""
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter


_MAX_PAIRS = 10_000
_LAG_COLUMNS = [
    "band_index", "lag_pixels", "lag_m", "direction", "n_pairs_available",
    "n_pairs", "correlation_raw", "correlation_detrended",
]


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} deve ser inteiro")
    if value < minimum:
        raise ValueError(f"{name} deve ser >= {minimum}")
    return int(value)


def _valid_mask(valid):
    valid = np.asarray(valid)
    if valid.ndim != 2 or min(valid.shape) == 0 or valid.dtype != np.dtype(bool):
        raise ValueError("valid deve ser uma matriz booleana 2D não vazia")
    return valid


def _cube(cube, shape):
    cube = np.asarray(cube)
    if (cube.ndim != 3 or cube.shape[:2] != shape or cube.shape[2] < 1
            or not np.issubdtype(cube.dtype, np.number)
            or np.issubdtype(cube.dtype, np.complexfloating)):
        raise ValueError("cube deve ser matriz numérica real H × W × B compatível")
    return cube


def _correlation(first, second):
    """Pearson; uma variável constante/numericamente constante retorna NaN."""
    first = np.asarray(first, dtype="float64")
    second = np.asarray(second, dtype="float64")
    centered_first = first - first.mean()
    centered_second = second - second.mean()
    norm_first = np.linalg.norm(centered_first)
    norm_second = np.linalg.norm(centered_second)
    tolerance = np.finfo("float64").eps * np.sqrt(len(first)) * 8
    if (norm_first <= tolerance * max(1., float(np.max(np.abs(first))))
            or norm_second <= tolerance * max(1., float(np.max(np.abs(second))))):
        return np.nan
    return float(np.clip(np.dot(centered_first, centered_second)
                         / (norm_first * norm_second), -1., 1.))


def lag_diagnostics(cube, valid, pixel_size=3,
                    lags=(1, 2, 4, 8, 16, 32, 64, 96, 128), minimum_pairs=100):
    """Correlação por banda e distância horizontal/vertical dentro da ROI.

Índices das bandas começam em zero. Para cada banda, um plano linear
intercepto + linha + coluna é ajustado por mínimos quadrados aos seus pixels
finitos válidos na ROI. A correlação detrended usa os resíduos desse plano.
Isto não remove efeitos ópticos não lineares nem identifica profundidade.

Somente pares válidos e finitos entram no diagnóstico. Cada direção/lag/banda
usa no máximo 10 mil pares, selecionados por stride determinístico na ordem
das linhas. ``n_pairs_available`` registra a população e ``n_pairs`` o número
efetivamente usado. Essa subamostragem pode sofrer aliasing espacial e deve
ser interpretada junto com inspeção visual. Pares insuficientes ou variância
nula produzem NaN. Negativos finitos são preservados.

``pixel_size`` é a resolução em metros, igual nos dois eixos; verificar a
grade geográfica antes de chamar esta função.
"""
    valid = _valid_mask(valid)
    cube = _cube(cube, valid.shape)
    minimum_pairs = _integer(minimum_pairs, "minimum_pairs", 3)
    if minimum_pairs > _MAX_PAIRS:
        raise ValueError("minimum_pairs não pode exceder o limite de 10000 pares")
    if (isinstance(pixel_size, (bool, np.bool_)) or not np.isscalar(pixel_size)
            or not np.isfinite(pixel_size) or pixel_size <= 0):
        raise ValueError("pixel_size deve ser positivo e finito")
    try:
        lags = tuple(_integer(lag, "lag", 1) for lag in lags)
    except TypeError as error:
        raise ValueError("lags deve ser uma sequência de inteiros positivos") from error
    if not lags or len(set(lags)) != len(lags):
        raise ValueError("lags não pode ser vazio ou conter repetições")
    height, width = valid.shape
    rows = []
    for band_index in range(cube.shape[2]):
        band = cube[:, :, band_index]
        band_valid = valid & np.isfinite(band)
        rr, cc = np.nonzero(band_valid)
        # Coordenadas locais normalizadas evitam mau condicionamento em ROIs
        # afastadas da origem do raster. Um eixo de comprimento 1 é constante.
        plane = np.column_stack((np.ones(len(rr)), rr / max(height - 1, 1),
                                 cc / max(width - 1, 1)))
        coefficients = (np.linalg.lstsq(plane, band[rr, cc], rcond=None)[0]
                        if len(rr) >= 3 else np.zeros(3))
        for lag in lags:
            for direction in ("horizontal", "vertical"):
                dr, dc = (0, lag) if direction == "horizontal" else (lag, 0)
                if dr >= height or dc >= width:
                    pair_r = pair_c = np.empty(0, dtype="int64")
                else:
                    pairs = (band_valid[:height - dr, :width - dc]
                             & band_valid[dr:, dc:])
                    pair_r, pair_c = np.nonzero(pairs)
                available = len(pair_r)
                stride = max(1, (available + _MAX_PAIRS - 1) // _MAX_PAIRS)
                pair_r, pair_c = pair_r[::stride], pair_c[::stride]
                count = len(pair_r)
                correlation_raw = correlation_detrended = np.nan
                if count >= minimum_pairs:
                    first = band[pair_r, pair_c].astype("float64")
                    second = band[pair_r + dr, pair_c + dc].astype("float64")
                    trend_first = (coefficients[0]
                                   + coefficients[1] * pair_r / max(height - 1, 1)
                                   + coefficients[2] * pair_c / max(width - 1, 1))
                    trend_second = (coefficients[0]
                                    + coefficients[1] * (pair_r + dr) / max(height - 1, 1)
                                    + coefficients[2] * (pair_c + dc) / max(width - 1, 1))
                    correlation_raw = _correlation(first, second)
                    correlation_detrended = _correlation(
                        first - trend_first, second - trend_second)
                rows.append({
                    "band_index": band_index, "lag_pixels": lag,
                    "lag_m": float(lag * pixel_size), "direction": direction,
                    "n_pairs_available": available, "n_pairs": count,
                    "correlation_raw": correlation_raw,
                    "correlation_detrended": correlation_detrended,
                })
    return pd.DataFrame(rows, columns=_LAG_COLUMNS)


def spatial_partition(valid, block_pixels, buffer_pixels, validation_fraction=.2,
                      seed=42, min_train_blocks=5, min_validation_blocks=2):
    """Sorteia blocos de avaliação e exclui uma vizinhança do desenvolvimento.

Retorna ``(split, blocks, metadata)``. ``split`` é uint8: 0 inválido,
1 desenvolvimento, 2 avaliação, 3 buffer. Blocos são quadrados com origem
local (0, 0) da ROI; as bordas parciais usam sua área efetiva na ROI.

Blocos com pelo menos 50% dessa área contendo pixels válidos são elegíveis
para avaliação. Entre eles, uma única seleção aleatória, sem reposição,
escolhe ceil(fração × elegíveis), respeitando o mínimo de avaliação. Não há
busca por uma semente ou arranjo mais conveniente. Os demais pixels válidos
integram desenvolvimento, exceto os situados a distância Chebyshev <= buffer
de qualquer pixel válido de avaliação. Não há pixel de desenvolvimento em
bloco selecionado para avaliação. Após o buffer, exigem-se pelo menos
``min_train_blocks`` blocos elegíveis com desenvolvimento remanescente.

O buffer reduz proximidade espacial; não constitui prova de independência.
O teste de sensibilidade deve variar tamanhos de bloco e buffer.
"""
    valid = _valid_mask(valid)
    block_pixels = _integer(block_pixels, "block_pixels", 1)
    buffer_pixels = _integer(buffer_pixels, "buffer_pixels")
    min_train_blocks = _integer(min_train_blocks, "min_train_blocks", 1)
    min_validation_blocks = _integer(min_validation_blocks, "min_validation_blocks", 1)
    seed = _integer(seed, "seed")
    if (isinstance(validation_fraction, (bool, np.bool_))
            or not np.isscalar(validation_fraction)
            or not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1):
        raise ValueError("validation_fraction deve estar entre 0 e 1")
    height, width = valid.shape
    records = []
    for block_row, row_start in enumerate(range(0, height, block_pixels)):
        for block_col, col_start in enumerate(range(0, width, block_pixels)):
            row_stop = min(height, row_start + block_pixels)
            col_stop = min(width, col_start + block_pixels)
            block_area = (row_stop - row_start) * (col_stop - col_start)
            valid_pixels = int(valid[row_start:row_stop, col_start:col_stop].sum())
            records.append({
                "block_id": len(records), "block_row": block_row,
                "block_col": block_col, "row_start": row_start,
                "row_stop": row_stop, "col_start": col_start,
                "col_stop": col_stop, "area_pixels": block_area,
                "valid_pixels": valid_pixels,
                "valid_fraction": valid_pixels / block_area,
                "validation_eligible": valid_pixels / block_area >= .5,
            })
    blocks = pd.DataFrame(records)
    eligible = blocks.loc[blocks["validation_eligible"], "block_id"].to_numpy()
    n_validation = max(min_validation_blocks,
                       int(np.ceil(len(eligible) * validation_fraction)))
    if len(eligible) - n_validation < min_train_blocks:
        raise ValueError("Blocos elegíveis insuficientes para desenvolvimento e avaliação")
    chosen = np.random.default_rng(seed).choice(eligible, size=n_validation, replace=False)
    blocks["selected_validation"] = blocks["block_id"].isin(chosen)
    split = valid.astype("uint8")
    for block in blocks.loc[blocks["selected_validation"]].itertuples():
        view = split[block.row_start:block.row_stop, block.col_start:block.col_stop]
        view[view == 1] = 2
    if buffer_pixels:
        near_validation = maximum_filter(
            (split == 2).astype("uint8"), size=2 * buffer_pixels + 1,
            mode="constant", cval=0).astype(bool)
        split[(split == 1) & near_validation] = 3
    for label, column in ((1, "development_pixels"), (2, "evaluation_pixels"),
                          (3, "buffer_pixels")):
        blocks[column] = [
            int((split[block.row_start:block.row_stop, block.col_start:block.col_stop]
                 == label).sum()) for block in blocks.itertuples()
        ]
    train_blocks = int((blocks["validation_eligible"]
                        & (blocks["development_pixels"] > 0)).sum())
    if train_blocks < min_train_blocks:
        raise ValueError("Buffer deixou blocos de desenvolvimento insuficientes; revisar geometria")
    metadata = {
        "scheme": "roi_block_holdout_chebyshev_buffer_v1",
        "shape": [height, width], "origin": "ROI row=0, col=0",
        "block_pixels": block_pixels, "buffer_pixels": buffer_pixels,
        "validation_fraction_requested": float(validation_fraction), "seed": seed,
        "validation_eligibility_minimum_fraction": .5,
        "partial_edge_area": "intersection_with_roi",
        "eligible_blocks": int(len(eligible)),
        "validation_blocks": n_validation, "development_eligible_blocks": train_blocks,
        "min_train_blocks": min_train_blocks,
        "min_validation_blocks": min_validation_blocks,
        "counts": {str(label): int((split == label).sum()) for label in range(4)},
        "classes": {"0": "inválido", "1": "desenvolvimento", "2": "avaliação", "3": "buffer"},
        "limitation": "Separação espacial e buffer não demonstram independência estatística.",
    }
    return split, blocks, metadata


def spatial_sample(cube, split, train_size=20000, validation_size=5000, seed=42,
                   row_offset=0, col_offset=0):
    """Amostras uniformes sem reposição nas duas populações já fixadas.

Retorna ``{"train": {"values", "positions"}, "validation": {...}}``.
``positions`` contém (linha, coluna) na grade original, somando offsets à
grade local da ROI. Alterar offsets não muda os valores sorteados. Cada
amostra tem min(tamanho solicitado, população) pixels, com mínimos de 20
para desenvolvimento e 10 para avaliação. Buffer e inválidos são excluídos.
Não há balanceamento por espectro, nomes de pontos ou categorias ecológicas.
Todos os pixels designados para amostragem precisam ser finitos nas bandas.
"""
    split = np.asarray(split)
    if (split.ndim != 2 or min(split.shape) == 0
            or not np.issubdtype(split.dtype, np.integer)
            or not np.isin(split, (0, 1, 2, 3)).all()):
        raise ValueError("split deve ser uma matriz inteira 2D com valores 0, 1, 2 ou 3")
    cube = _cube(cube, split.shape)
    train_size = _integer(train_size, "train_size", 20)
    validation_size = _integer(validation_size, "validation_size", 10)
    seed = _integer(seed, "seed")
    row_offset = _integer(row_offset, "row_offset")
    col_offset = _integer(col_offset, "col_offset")
    generators = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)]
    result = {}
    for label, name, size, minimum, rng in (
        (1, "train", train_size, 20, generators[0]),
        (2, "validation", validation_size, 10, generators[1]),
    ):
        population = np.flatnonzero(split.ravel() == label)
        if len(population) < minimum:
            raise ValueError(f"População {name} insuficiente: mínimo {minimum} pixels")
        rr, cc = np.unravel_index(population, split.shape)
        if not np.isfinite(cube[rr, cc]).all():
            raise ValueError(f"População {name} contém bandas não finitas; corrigir valid antes da separação")
        chosen = rng.choice(population, size=min(size, len(population)), replace=False)
        rr, cc = np.unravel_index(chosen, split.shape)
        result[name] = {
            "values": cube[rr, cc].copy(),
            "positions": np.column_stack((rr + row_offset, cc + col_offset)).astype("int64"),
        }
    return result
