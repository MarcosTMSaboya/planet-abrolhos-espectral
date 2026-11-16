import os
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium
from src.painel_qualidade import render_quality
from src.pontos_campo import render_field_points, add_field_points
from src.raster_utils import OutsideRasterError, sample_pixel
from src.analise_classificacao import render_classification_analysis
from src.produtos_acolite import product_folder
from src.reclassificacao_app import render as render_full_scene
from src.pontos_validacao_app import render as render_validation_points


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROOT = Path(__file__).resolve().parent

PLANET_DIRS = [ROOT / "PSScene"]
SECONDARY_SCENE_DIR = os.getenv("PLANET_SECONDARY_SCENE_DIR")
if SECONDARY_SCENE_DIR:
    PLANET_DIRS.append(Path(SECONDARY_SCENE_DIR))

CENTROID_PATH = Path(
    os.getenv("PLANET_CENTROID_PATH", str(ROOT / "points" / "centroids.gpkg"))
)

ACOLITE_GEOTIFF_DIR = ROOT / "data" / "acolite" / "geotiff"

BAND_NAMES = {
    1: "Coastal Blue",
    2: "Blue",
    3: "Green I",
    4: "Green",
    5: "Yellow",
    6: "Red",
    7: "Red Edge",
    8: "NIR",
}

BAND_WAVELENGTHS = {
    1: 442,
    2: 490,
    3: 531,
    4: 565,
    5: 610,
    6: 665,
    7: 705,
    8: 865,
}

ACOLITE_BANDS = {
    444: "Coastal Blue",
    492: "Blue",
    533: "Green I",
    566: "Green",
    612: "Yellow",
    666: "Red",
    707: "Red Edge",
    866: "NIR",
}

COMPOSITES = {
    "Cor natural — RGB 6 / 4 / 2": (6, 4, 2),
    "Águas rasas / bentos — RGB 4 / 3 / 2": (4, 3, 2),
    "Coastal / azul — RGB 3 / 2 / 1": (3, 2, 1),
    "Falsa cor — RGB 8 / 6 / 4": (8, 6, 4),
}

ACOLITE_COMPOSITES = {
    "Cor natural — 666 / 566 / 492": (666, 566, 492),
    "Águas rasas / bentos — 566 / 533 / 492": (566, 533, 492),
    "Coastal / azul — 533 / 492 / 444": (533, 492, 444),
    "Falsa cor — 866 / 666 / 566": (866, 666, 566),
}

UDM2_NAMES = {
    1: "Clear",
    2: "Snow / Ice",
    3: "Shadow",
    4: "Haze (Light Haze no nome legado)",
    5: "Heavy Haze (legado; zero no UDM2.1)",
    6: "Cloud",
    7: "Confidence",
    8: "Unusable / bits UDM1",
}


# ============================================================
# ARQUIVOS PLANET
# ============================================================

def find_scenes():
    scenes = []

    for folder in PLANET_DIRS:
        if folder.exists():
            scenes.extend(
                folder.rglob(
                    "*_3B_AnalyticMS_8b_clip.tif"
                )
            )

    return sorted(
        set(scenes)
    )


def scene_id(path):
    return Path(path).name.split(
        "_3B_"
    )[0]


def scene_date_label(path):
    """
    Converte o prefixo YYYYMMDD da cena para DD/MM/YYYY.
    """
    sid = scene_id(path)
    raw_date = sid.split("_")[0]

    if (
        len(raw_date) == 8
        and raw_date.isdigit()
    ):
        return (
            f"{raw_date[6:8]}/"
            f"{raw_date[4:6]}/"
            f"{raw_date[0:4]}"
        )

    return raw_date


def find_asset(
    scene_path,
    filename
):
    scene_path = Path(
        scene_path
    )

    local_file = (
        scene_path.parent
        / filename
    )

    if local_file.exists():
        return local_file

    for folder in PLANET_DIRS:

        if not folder.exists():
            continue

        matches = list(
            folder.rglob(
                filename
            )
        )

        if matches:
            return matches[0]

    return None


def find_metadata(path):
    sid = scene_id(
        path
    )

    return find_asset(
        path,
        f"{sid}_metadata.json"
    )


def find_xml_metadata(path):
    sid = scene_id(
        path
    )

    return find_asset(
        path,
        f"{sid}_3B_AnalyticMS_8b_metadata_clip.xml"
    )


def find_udm2(path):
    sid = scene_id(
        path
    )

    return find_asset(
        path,
        f"{sid}_3B_udm2_clip.tif"
    )


def load_json(path):

    if path is None:
        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

    except Exception:

        return {}


@st.cache_data(
    show_spinner=False
)
def read_reflectance_coefficients(
    xml_path
):

    if not xml_path:
        return {}

    root = ET.parse(
        xml_path
    ).getroot()

    coefficients = {}

    for element in root.iter():

        if not element.tag.endswith(
            "bandSpecificMetadata"
        ):
            continue

        band_number = None
        coefficient = None

        for child in element:

            if child.tag.endswith(
                "bandNumber"
            ):

                band_number = int(
                    child.text
                )

            elif child.tag.endswith(
                "reflectanceCoefficient"
            ):

                coefficient = float(
                    child.text
                )

        if (
            band_number is not None
            and coefficient is not None
        ):

            coefficients[
                band_number
            ] = coefficient

    return coefficients


# ============================================================
# RASTER / PREVIEWS
# ============================================================

def preview_dimensions(
    src,
    max_size
):

    factor = min(
        1.0,
        max_size
        / max(
            src.width,
            src.height
        )
    )

    width = max(
        1,
        int(
            src.width
            * factor
        )
    )

    height = max(
        1,
        int(
            src.height
            * factor
        )
    )

    return height, width


def stretch(
    array,
    low_percentile,
    high_percentile
):

    array = array.astype(
        "float32"
    )

    valid = (
        np.isfinite(
            array
        )
        & (
            array > 0
        )
    )

    if not valid.any():

        return np.zeros_like(
            array,
            dtype="float32"
        )

    low, high = np.percentile(
        array[
            valid
        ],
        [
            low_percentile,
            high_percentile
        ]
    )

    if high <= low:

        return np.zeros_like(
            array,
            dtype="float32"
        )

    result = (
        array - low
    ) / (
        high - low
    )

    result = np.clip(
        result,
        0,
        1
    )

    result[
        ~valid
    ] = 0

    return result


@st.cache_data(
    show_spinner=False
)
def make_rgb(
    path,
    bands,
    p_low,
    p_high,
    max_size
):

    with rasterio.open(
        path
    ) as src:

        height, width = (
            preview_dimensions(
                src,
                max_size
            )
        )

        data = src.read(
            list(
                bands
            ),
            out_shape=(
                3,
                height,
                width
            ),
            resampling=(
                Resampling.bilinear
            )
        )

    output = np.zeros_like(
        data,
        dtype="float32"
    )

    for i in range(
        3
    ):

        output[
            i
        ] = stretch(
            data[
                i
            ],
            p_low,
            p_high
        )

    rgb = np.moveaxis(
        output,
        0,
        2
    )

    return (
        rgb * 255
    ).astype(
        "uint8"
    )


@st.cache_data(
    show_spinner=False
)
def make_single_band(
    path,
    band,
    p_low,
    p_high,
    max_size
):

    with rasterio.open(
        path
    ) as src:

        height, width = (
            preview_dimensions(
                src,
                max_size
            )
        )

        data = src.read(
            band,
            out_shape=(
                height,
                width
            ),
            resampling=(
                Resampling.bilinear
            )
        )

    data = stretch(
        data,
        p_low,
        p_high
    )

    return (
        data * 255
    ).astype(
        "uint8"
    )


@st.cache_data(
    show_spinner=False
)
def make_udm2(
    path,
    band,
    max_size
):

    with rasterio.open(
        path
    ) as src:

        height, width = (
            preview_dimensions(
                src,
                max_size
            )
        )

        data = src.read(
            band,
            out_shape=(
                height,
                width
            ),
            resampling=(
                Resampling.nearest
            )
        )

    if band == 7:

        data = np.clip(
            data.astype(
                "float32"
            ) / 100,
            0,
            1
        )

    else:

        data = (
            data > 0
        ).astype(
            "float32"
        )

    return (
        data * 255
    ).astype(
        "uint8"
    )


def raster_bounds_4326(
    path
):

    with rasterio.open(
        path
    ) as src:

        west, south, east, north = (
            transform_bounds(
                src.crs,
                "EPSG:4326",
                *src.bounds,
                densify_pts=21
            )
        )

    return [
        [
            south,
            west
        ],
        [
            north,
            east
        ]
    ]


def raster_info(
    path
):

    with rasterio.open(
        path
    ) as src:

        return {
            "CRS":
                str(
                    src.crs
                ),

            "Bandas":
                src.count,

            "Colunas":
                src.width,

            "Linhas":
                src.height,

            "Resolução X":
                src.res[
                    0
                ],

            "Resolução Y":
                abs(
                    src.res[
                        1
                    ]
                ),

            "Tipo":
                src.dtypes[
                    0
                ],

            "NoData":
                src.nodata,

            "Oeste":
                src.bounds.left,

            "Sul":
                src.bounds.bottom,

            "Leste":
                src.bounds.right,

            "Norte":
                src.bounds.top,
        }


# ============================================================
# CENTROIDES
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_centroids(
    path
):

    gdf = gpd.read_file(
        path
    )

    if gdf.crs is None:

        raise ValueError(
            "O GPKG não possui CRS definido."
        )

    gdf = gdf.to_crs(
        "EPSG:4326"
    )

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & (
            gdf.geometry.geom_type
            == "Point"
        )
    ].copy()

    return gdf


# ============================================================
# PLANET TOA — EXTRAÇÃO ESPECTRAL
# ============================================================

def sample_spectrum(
    raster_path,
    lon,
    lat,
    coefficients
):

    with rasterio.open(
        raster_path
    ) as src:

        values = sample_pixel(src, lon, lat)

    records = []

    for band_number, dn in enumerate(
        values,
        start=1
    ):

        dn_value = float(
            dn
        )

        coefficient = (
            coefficients.get(
                band_number
            )
        )

        toa = np.nan

        if coefficient is not None:

            toa = (
                dn_value
                * coefficient
            )

        records.append(
            {
                "Banda":
                    f"B{band_number}",

                "Nome":
                    BAND_NAMES[
                        band_number
                    ],

                "Comprimento de onda (nm)":
                    BAND_WAVELENGTHS[
                        band_number
                    ],

                "DN":
                    dn_value,

                "Reflectância TOA":
                    toa,
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# ACOLITE RHOS
# ============================================================

def acolite_scene_key(
    scene_path
):

    sid = scene_id(
        scene_path
    )

    parts = sid.split(
        "_"
    )

    if len(
        parts
    ) >= 2:

        return parts[
            1
        ]

    return sid


def find_acolite_rhos(
    scene_path
):

    folder, key = product_folder(
        ACOLITE_GEOTIFF_DIR, scene_id(scene_path),
        [scene_id(path) for path in scenes],
    )

    files = {}

    for wavelength in ACOLITE_BANDS:

        path = (
            folder
            / f"{key}_rhos_{wavelength}.tif"
        )

        if path.exists():

            files[
                wavelength
            ] = path

    return folder, files


@st.cache_data(
    show_spinner=False,
    persist="disk"
)
def make_acolite_rgb(
    scene_path,
    wavelengths,
    p_low,
    p_high,
    max_size,
    source_signature=None
):

    folder, files = (
        find_acolite_rhos(
            scene_path
        )
    )

    missing = [
        wavelength
        for wavelength
        in wavelengths
        if wavelength not in files
    ]

    if missing:

        raise FileNotFoundError(
            "Bandas ACOLITE ausentes: "
            + ", ".join(
                str(
                    value
                )
                for value
                in missing
            )
            + f" nm. Pasta esperada: {folder}"
        )

    first_path = files[
        wavelengths[
            0
        ]
    ]

    with rasterio.open(
        first_path
    ) as src:

        height, width = (
            preview_dimensions(
                src,
                max_size
            )
        )

    channels = []
    valid_masks = []

    for wavelength in wavelengths:

        raster_path = files[
            wavelength
        ]

        with rasterio.open(
            raster_path
        ) as src:

            data = (
                src.read(
                    1,
                    out_shape=(
                        height,
                        width
                    ),
                    resampling=(
                        Resampling.bilinear
                    ),
                    masked=True
                )
                .filled(
                    np.nan
                )
                .astype(
                    "float32"
                )
            )

        valid = (
            np.isfinite(
                data
            )
            & (
                np.abs(
                    data
                ) < 1e20
            )
        )

        clean = data.copy()

        clean[
            ~valid
        ] = np.nan

        channels.append(
            stretch(
                clean,
                p_low,
                p_high
            )
        )

        valid_masks.append(
            valid
        )

    rgb = np.dstack(
        channels
    )

    alpha = (
        np.logical_and.reduce(
            valid_masks
        )
        .astype(
            "uint8"
        )
        * 255
    )

    rgba = np.dstack(
        [
            (
                rgb * 255
            ).astype(
                "uint8"
            ),
            alpha
        ]
    )

    return rgba


def sample_acolite_spectrum(
    scene_path,
    lon,
    lat
):

    folder, files = (
        find_acolite_rhos(
            scene_path
        )
    )

    missing = [
        wavelength
        for wavelength
        in ACOLITE_BANDS
        if wavelength not in files
    ]

    if missing:

        raise FileNotFoundError(
            "Bandas ACOLITE ausentes: "
            + ", ".join(
                str(
                    value
                )
                for value
                in missing
            )
            + f" nm. Pasta esperada: {folder}"
        )

    records = []

    for wavelength, band_name in (
        ACOLITE_BANDS.items()
    ):

        raster_path = files[
            wavelength
        ]

        with rasterio.open(
            raster_path
        ) as src:

            value = float(sample_pixel(src, lon, lat)[0])

        records.append(
            {
                "Banda":
                    f"rhos_{wavelength}",

                "Nome":
                    band_name,

                "Comprimento de onda (nm)":
                    wavelength,

                "Reflectância ACOLITE rhos":
                    value,
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Abrolhos Macroalgas",
    page_icon="🛰️",
    layout="wide"
)

st.title(
    "Abrolhos — PlanetScope SuperDove"
)

st.caption(
    "Ambiente de visualização e análise "
    "espectral para mapeamento bentônico."
)


# ============================================================
# CENA
# ============================================================

module = st.sidebar.radio(
    "Módulo",
    ["Qualidade e assinaturas", "Análise da classificação", "Pontos de campo", "Mapa GIS", "Espectro", "RGB", "Bandas", "UDM2", "Metadados", "Pontos de verificação", "Classificação da cena"],
    key="app_module",
)

# O painel usa apenas CSVs: não procura cenas nem abre rasters.
if module == "Qualidade e assinaturas":
    render_quality(ROOT)
    st.stop()

if module == "Análise da classificação":
    render_classification_analysis(ROOT)
    st.stop()

if module == "Pontos de campo":
    render_field_points(ROOT)
    st.stop()

if module == "Classificação da cena":
    render_full_scene(ROOT)
    st.stop()

if module == "Pontos de verificação":
    render_validation_points(ROOT)
    st.stop()

scenes = find_scenes()

if not scenes:

    st.error(
        "Nenhuma cena SuperDove "
        "8 bandas foi encontrada."
    )

    st.code(
        "\n".join(
            str(
                folder
            )
            for folder
            in PLANET_DIRS
        )
    )

    st.stop()


SCENE_LABELS = {
    scene: (
        f"Cena {index} "
        f"({scene_date_label(scene)})"
    )
    for index, scene
    in enumerate(
        scenes,
        start=1
    )
}

SCENE_LABEL_BY_ID = {
    scene_id(scene):
        SCENE_LABELS[scene]
    for scene
    in scenes
}


selected_scene = (
    st.sidebar.selectbox(
        "Cena Planet",
        scenes,
        format_func=lambda x:
            SCENE_LABELS.get(
                x,
                Path(x).name
            )
    )
)

sid = scene_id(
    selected_scene
)

metadata_path = (
    find_metadata(
        selected_scene
    )
)

xml_metadata_path = (
    find_xml_metadata(
        selected_scene
    )
)

udm2_path = (
    find_udm2(
        selected_scene
    )
)

metadata = load_json(
    metadata_path
)

properties = metadata.get(
    "properties",
    {}
)

reflectance_coefficients = (
    read_reflectance_coefficients(
        str(
            xml_metadata_path
        )
        if xml_metadata_path
        else None
    )
)

acolite_folder, acolite_files = (
    find_acolite_rhos(
        selected_scene
    )
)

acolite_ready = (
    len(
        acolite_files
    )
    == len(
        ACOLITE_BANDS
    )
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.divider()

st.sidebar.divider()

st.sidebar.header(
    "Visualização"
)

max_size = (
    st.sidebar.slider(
        "Tamanho máximo do preview",
        min_value=400,
        max_value=1800,
        value=800,
        step=200
    )
)

p_low, p_high = (
    st.sidebar.slider(
        "Stretch percentil",
        min_value=0,
        max_value=100,
        value=(
            2,
            98
        )
    )
)

st.sidebar.caption(
    f"Scene ID: {sid}"
)

if xml_metadata_path:

    st.sidebar.success(
        "XML radiométrico encontrado"
    )

else:

    st.sidebar.warning(
        "XML radiométrico não encontrado"
    )

if acolite_ready:

    st.sidebar.success(
        "ACOLITE rhos disponível"
    )

else:

    st.sidebar.caption(
        "ACOLITE rhos não disponível "
        "para esta cena"
    )


# ============================================================
# RESUMO
# ============================================================

c1, c2, c3, c4 = (
    st.columns(
        4
    )
)

c1.metric(
    "Instrumento",
    properties.get(
        "instrument",
        "—"
    )
)

c2.metric(
    "Nuvens",
    f"{properties.get('cloud_percent', '—')} %"
)

c3.metric(
    "Sun elevation",
    f"{properties.get('sun_elevation', '—')}°"
)

c4.metric(
    "View angle",
    f"{properties.get('view_angle', '—')}°"
)


# ============================================================
# MÓDULO: MAPA GIS
# ============================================================

if module == "Mapa GIS":

    st.subheader(
        "SuperDove + centroides das algas"
    )

    st.markdown(
        "#### Cenas visíveis"
    )

    scenes_to_draw = st.multiselect(
        "Selecione as cenas que devem aparecer no mapa",
        options=scenes,
        default=[
            selected_scene
        ],
        format_func=lambda x:
            SCENE_LABELS.get(
                x,
                scene_id(x)
            ),
        help=(
            "Carregar apenas as cenas necessárias "
            "deixa o mapa muito mais rápido."
        ),
    )

    show_centroids = st.checkbox(
        "Mostrar centroides de algas",
        value=True,
    )

    with st.expander(
        "Identificação das cenas",
        expanded=False
    ):

        scene_table = pd.DataFrame(
            [
                {
                    "Cena":
                        SCENE_LABELS[scene],

                    "Scene ID Planet":
                        scene_id(scene),

                    "Data":
                        scene_date_label(scene),
                }

                for scene
                in scenes
            ]
        )

        st.dataframe(
            scene_table,
            width="stretch",
            hide_index=True,
        )

    if not scenes_to_draw:

        st.info(
            "Selecione pelo menos uma cena "
            "para visualizar o mapa."
        )

        st.stop()

    map_source = st.radio(
        "Fonte da imagem",
        [
            "ACOLITE rhos",
            "Planet TOA",
        ],
        horizontal=True,
        index=0 if acolite_ready else 1,
    )

    if map_source == "ACOLITE rhos":

        composite = st.selectbox(
            "Composição do mapa",
            list(
                ACOLITE_COMPOSITES.keys()
            )
        )

        wavelengths = (
            ACOLITE_COMPOSITES[
                composite
            ]
        )

        st.caption(
            f"R = rhos_{wavelengths[0]} | "
            f"G = rhos_{wavelengths[1]} | "
            f"B = rhos_{wavelengths[2]}"
        )

    else:

        composite = st.selectbox(
            "Composição do mapa",
            list(
                COMPOSITES.keys()
            )
        )

        bands = COMPOSITES[
            composite
        ]

        st.caption(
            f"R = B{bands[0]} | "
            f"G = B{bands[1]} | "
            f"B = B{bands[2]}"
        )

    opacity = st.slider(
        "Opacidade das cenas",
        min_value=0.20,
        max_value=1.00,
        value=0.90,
        step=0.05,
    )

    map_preview_size = min(
        max_size,
        650
    )

    st.caption(
        "Mapa em preview otimizado "
        f"(máx. {map_preview_size}px por cena). "
        "Os dados originais não são reduzidos."
    )

    prepared_layers = []
    skipped_scenes = []

    with st.spinner(
        "Preparando cenas selecionadas..."
    ):

        for scene_path in scenes_to_draw:

            current_sid = scene_id(
                scene_path
            )

            current_label = (
                SCENE_LABEL_BY_ID.get(
                    current_sid,
                    current_sid
                )
            )

            if map_source == "ACOLITE rhos":

                (
                    scene_acolite_folder,
                    scene_acolite_files
                ) = find_acolite_rhos(
                    scene_path
                )

                scene_acolite_ready = (
                    len(
                        scene_acolite_files
                    )
                    == len(
                        ACOLITE_BANDS
                    )
                )

                if not scene_acolite_ready:

                    skipped_scenes.append(
                        current_label
                    )

                    continue

                source_signature = tuple(
                    (
                        wavelength,
                        scene_acolite_files[
                            wavelength
                        ].stat().st_size,
                        scene_acolite_files[
                            wavelength
                        ].stat().st_mtime_ns,
                    )
                    for wavelength
                    in wavelengths
                )

                try:

                    rgb_layer = make_acolite_rgb(
                        scene_path,
                        wavelengths,
                        p_low,
                        p_high,
                        map_preview_size,
                        source_signature,
                    )

                    reference_raster = (
                        scene_acolite_files[
                            wavelengths[0]
                        ]
                    )

                    layer_bounds = (
                        raster_bounds_4326(
                            reference_raster
                        )
                    )

                except Exception as error:

                    skipped_scenes.append(
                        current_label
                    )

                    st.warning(
                        f"{current_label} foi ignorada "
                        "porque não pôde ser lida."
                    )

                    st.code(
                        str(
                            error
                        )
                    )

                    continue

            else:

                try:

                    rgb_layer = make_rgb(
                        str(
                            scene_path
                        ),
                        bands,
                        p_low,
                        p_high,
                        map_preview_size
                    )

                    layer_bounds = (
                        raster_bounds_4326(
                            scene_path
                        )
                    )

                except Exception as error:

                    skipped_scenes.append(
                        current_label
                    )

                    st.warning(
                        f"{current_label} foi ignorada "
                        "porque não pôde ser lida."
                    )

                    st.code(
                        str(
                            error
                        )
                    )

                    continue

            prepared_layers.append(
                (
                    current_label,
                    current_sid,
                    rgb_layer,
                    layer_bounds,
                )
            )

    if not prepared_layers:

        st.error(
            "Nenhuma das cenas selecionadas "
            "pôde ser carregada."
        )

        st.stop()

    south = min(
        layer_bounds[0][0]
        for _, _, _, layer_bounds
        in prepared_layers
    )

    west = min(
        layer_bounds[0][1]
        for _, _, _, layer_bounds
        in prepared_layers
    )

    north = max(
        layer_bounds[1][0]
        for _, _, _, layer_bounds
        in prepared_layers
    )

    east = max(
        layer_bounds[1][1]
        for _, _, _, layer_bounds
        in prepared_layers
    )

    total_bounds = [
        [
            south,
            west,
        ],
        [
            north,
            east,
        ],
    ]

    center = [
        (
            south
            + north
        ) / 2,
        (
            west
            + east
        ) / 2,
    ]

    m = folium.Map(
        location=center,
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    for (
        current_label,
        current_sid,
        rgb_layer,
        layer_bounds
    ) in prepared_layers:

        folium.raster_layers.ImageOverlay(
            image=rgb_layer,
            bounds=layer_bounds,
            opacity=opacity,
            name=current_label,
            show=True,
        ).add_to(
            m
        )

    field_count = add_field_points(m, ROOT, [item[1] for item in prepared_layers])
    st.caption(f"GPX: {field_count} pontos nas cenas exibidas. Verde: clear UDM2; laranja: revisar qualidade. Classes pendentes.")

    if show_centroids:

        if CENTROID_PATH.exists():

            try:

                points = load_centroids(
                    str(
                        CENTROID_PATH
                    )
                )

                for idx, row in (
                    points.iterrows()
                ):

                    geom = row.geometry

                    popup_lines = []

                    for column in (
                        points.columns
                    ):

                        if column == "geometry":
                            continue

                        value = row[
                            column
                        ]

                        try:

                            if pd.isna(
                                value
                            ):
                                continue

                        except Exception:
                            pass

                        popup_lines.append(
                            f"<b>{column}</b>: "
                            f"{value}"
                        )

                    popup = "<br>".join(
                        popup_lines
                    )

                    folium.CircleMarker(
                        location=[
                            geom.y,
                            geom.x,
                        ],
                        radius=5,
                        weight=2,
                        fill=True,
                        fill_opacity=0.8,
                        tooltip=(
                            f"Centroide {idx}"
                        ),
                        popup=folium.Popup(
                            popup,
                            max_width=400,
                        ),
                    ).add_to(
                        m
                    )

            except Exception as error:

                st.error(
                    "Erro ao carregar centroides."
                )

                st.code(
                    str(
                        error
                    )
                )

        else:

            st.warning(
                "GPKG dos centroides "
                "não encontrado."
            )

    loaded_names = [
        item[0]
        for item
        in prepared_layers
    ]

    st.success(
        "Cenas carregadas: "
        + ", ".join(
            loaded_names
        )
    )

    if skipped_scenes:

        st.warning(
            "Cenas ignoradas: "
            + ", ".join(
                skipped_scenes
            )
        )

    m.fit_bounds(
        total_bounds
    )

    selection_key = "_".join(
        item[1]
        for item
        in prepared_layers
    )

    st_folium(
        m,
        height=720,
        width=1200,
        key=(
            f"map_"
            f"{map_source}_"
            f"{composite}_"
            f"{opacity}_"
            f"{selection_key}"
        )
    )


# ============================================================
# MÓDULO: ESPECTRO
# ============================================================

elif module == "Espectro":

    st.info("Este módulo consulta o pixel central da cena original, sem filtro UDM2. "
            "Para medianas 3×3 filtradas, comparação entre cenas e auditoria, selecione ‘Qualidade e assinaturas’ no menu.")

    st.subheader(
        "Assinatura espectral "
        "dos centroides"
    )

    st.caption(
        "Compare a reflectância TOA original "
        "da Planet com a reflectância rhos "
        "corrigida pelo ACOLITE."
    )

    spectral_source = (
        st.radio(
            "Fonte espectral",
            [
                "ACOLITE rhos",
                "Planet TOA",
            ],
            horizontal=True,
            index=(
                0
                if acolite_ready
                else 1
            )
        )
    )

    if acolite_ready:

        st.success(
            "ACOLITE rhos disponível "
            "para esta cena."
        )

    else:

        st.caption(
            "ACOLITE rhos ainda não está "
            "completo para esta cena."
        )

    if not CENTROID_PATH.exists():

        st.error(
            "Arquivo de centroides "
            "não encontrado."
        )

        st.code(
            str(
                CENTROID_PATH
            )
        )

    else:

        try:

            points = (
                load_centroids(
                    str(
                        CENTROID_PATH
                    )
                )
            )

            if points.empty:

                st.warning(
                    "Nenhum ponto válido foi "
                    "encontrado no GPKG."
                )

            else:

                attribute_columns = [
                    col
                    for col
                    in points.columns
                    if col != "geometry"
                ]

                label_column = None

                if attribute_columns:

                    label_column = (
                        st.selectbox(
                            "Campo usado para "
                            "identificar os pontos",
                            attribute_columns
                        )
                    )

                if label_column:

                    selected_index = (
                        st.selectbox(
                            "Centroide",
                            points.index.tolist(),
                            format_func=lambda idx:
                                (
                                    f"{idx} — "
                                    f"{points.loc[idx, label_column]}"
                                )
                        )
                    )

                else:

                    selected_index = (
                        st.selectbox(
                            "Centroide",
                            points.index.tolist(),
                            format_func=lambda idx:
                                f"Centroide {idx}"
                        )
                    )

                row = points.loc[
                    selected_index
                ]

                geom = row.geometry

                lon = float(
                    geom.x
                )

                lat = float(
                    geom.y
                )

                ec1, ec2 = (
                    st.columns(
                        2
                    )
                )

                ec1.metric(
                    "Latitude",
                    f"{lat:.7f}"
                )

                ec2.metric(
                    "Longitude",
                    f"{lon:.7f}"
                )

                attributes = {}

                for column in (
                    points.columns
                ):

                    if column == "geometry":
                        continue

                    value = row[
                        column
                    ]

                    try:

                        if pd.isna(
                            value
                        ):
                            continue

                    except Exception:
                        pass

                    attributes[
                        column
                    ] = str(
                        value
                    )

                if attributes:

                    with st.expander(
                        "Atributos do centroide"
                    ):

                        st.json(
                            attributes
                        )

                # --------------------------------------------
                # ACOLITE
                # --------------------------------------------

                if (
                    spectral_source
                    == "ACOLITE rhos"
                ):

                    if not acolite_ready:

                        missing = [
                            wavelength
                            for wavelength
                            in ACOLITE_BANDS
                            if wavelength
                            not in acolite_files
                        ]

                        st.warning(
                            "Ainda não existem as oito "
                            "bandas ACOLITE para esta cena."
                        )

                        st.code(
                            "Pasta esperada:\n"
                            f"{acolite_folder}\n\n"
                            "Bandas ausentes:\n"
                            + ", ".join(
                                str(
                                    value
                                )
                                for value
                                in missing
                            )
                        )

                    else:

                        spectrum = (
                            sample_acolite_spectrum(
                                selected_scene,
                                lon,
                                lat
                            )
                        )

                        values = (
                            spectrum[
                                "Reflectância ACOLITE rhos"
                            ]
                        )

                        if (
                            values
                            .isna()
                            .all()
                        ):

                            st.warning(
                                "O centroide corresponde "
                                "a NoData na saída ACOLITE."
                            )

                        else:

                            table = (
                                spectrum.copy()
                            )

                            table[
                                "Reflectância ACOLITE rhos"
                            ] = (
                                table[
                                    "Reflectância ACOLITE rhos"
                                ]
                                .round(
                                    6
                                )
                            )

                            st.subheader(
                                "Valores espectrais corrigidos"
                            )

                            st.dataframe(
                                table,
                                width="stretch",
                                hide_index=True
                            )

                            st.subheader(
                                "Curva espectral ACOLITE"
                            )

                            chart = (
                                spectrum[
                                    [
                                        "Comprimento de onda (nm)",
                                        "Reflectância ACOLITE rhos",
                                    ]
                                ]
                                .dropna()
                                .set_index(
                                    "Comprimento de onda (nm)"
                                )
                            )

                            st.line_chart(
                                chart
                            )

                            st.info(
                                "rhos é a reflectância "
                                "de superfície produzida "
                                "no L2R do ACOLITE."
                            )

                # --------------------------------------------
                # PLANET TOA
                # --------------------------------------------

                else:

                    spectrum = (
                        sample_spectrum(
                            str(
                                selected_scene
                            ),
                            lon,
                            lat,
                            reflectance_coefficients
                        )
                    )

                    if (
                        spectrum[
                            "DN"
                        ]
                        .fillna(
                            0
                        )
                        .eq(
                            0
                        )
                        .all()
                    ):

                        st.warning(
                            "O centroide corresponde "
                            "a NoData ou está fora "
                            "da área válida."
                        )

                    else:

                        table = (
                            spectrum.copy()
                        )

                        table[
                            "DN"
                        ] = (
                            table[
                                "DN"
                            ]
                            .round(
                                0
                            )
                            .astype(
                                "Int64"
                            )
                        )

                        table[
                            "Reflectância TOA"
                        ] = (
                            table[
                                "Reflectância TOA"
                            ]
                            .round(
                                6
                            )
                        )

                        st.subheader(
                            "Valores espectrais TOA"
                        )

                        st.dataframe(
                            table,
                            width="stretch",
                            hide_index=True
                        )

                        st.subheader(
                            "Curva espectral TOA"
                        )

                        chart = (
                            spectrum[
                                [
                                    "Comprimento de onda (nm)",
                                    "Reflectância TOA",
                                ]
                            ]
                            .dropna()
                            .set_index(
                                "Comprimento de onda (nm)"
                            )
                        )

                        st.line_chart(
                            chart
                        )

                        st.info(
                            "Estes valores são "
                            "reflectância TOA da Planet. "
                            "Para análise bentônica, "
                            "prefira ACOLITE rhos "
                            "quando disponível."
                        )

        except OutsideRasterError:
            st.warning(
                "O centroide selecionado está fora da área desta cena. "
                "Selecione outro centroide ou outra cena Planet."
            )

        except Exception as error:

            st.error(
                "Erro na extração espectral."
            )

            st.code(
                str(
                    error
                )
            )


# ============================================================
# MÓDULO: RGB
# ============================================================

elif module == "RGB":

    st.subheader(
        "Composição RGB"
    )

    composite = (
        st.selectbox(
            "Composição",
            list(
                COMPOSITES.keys()
            )
        )
    )

    bands = COMPOSITES[
        composite
    ]

    st.caption(
        f"R = B{bands[0]} | "
        f"G = B{bands[1]} | "
        f"B = B{bands[2]}"
    )

    rgb = make_rgb(
        str(
            selected_scene
        ),
        bands,
        p_low,
        p_high,
        max_size
    )

    st.image(
        rgb,
        width="stretch"
    )


# ============================================================
# MÓDULO: BANDAS
# ============================================================

elif module == "Bandas":

    st.subheader(
        "Bandas individuais SuperDove"
    )

    band = (
        st.selectbox(
            "Banda",
            list(
                BAND_NAMES.keys()
            ),
            format_func=lambda x:
                (
                    f"B{x} — "
                    f"{BAND_NAMES[x]} — "
                    f"{BAND_WAVELENGTHS[x]} nm"
                )
        )
    )

    image = make_single_band(
        str(
            selected_scene
        ),
        band,
        p_low,
        p_high,
        max_size
    )

    st.image(
        image,
        caption=(
            f"B{band} — "
            f"{BAND_NAMES[band]} — "
            f"{BAND_WAVELENGTHS[band]} nm"
        ),
        width="stretch"
    )


# ============================================================
# MÓDULO: UDM2
# ============================================================

elif module == "UDM2":

    st.caption("Este módulo mostra a máscara da cena. Para decisões por observação, assinaturas filtradas e downloads, use ‘Qualidade e assinaturas’. "
               "Na banda 8, branco indica qualquer bit UDM1 acionado; a decodificação dos bits está na auditoria.")

    st.subheader(
        "Máscara de qualidade UDM2"
    )

    if udm2_path is None:

        st.warning(
            "UDM2 não encontrado."
        )

    else:

        udm_band = (
            st.selectbox(
                "Camada",
                list(
                    UDM2_NAMES.keys()
                ),
                format_func=lambda x:
                    (
                        f"B{x} — "
                        f"{UDM2_NAMES[x]}"
                    )
            )
        )

        image = make_udm2(
            str(
                udm2_path
            ),
            udm_band,
            max_size
        )

        st.image(
            image,
            caption=(
                f"UDM2 — "
                f"{UDM2_NAMES[udm_band]}"
            ),
            width="stretch"
        )


# ============================================================
# MÓDULO: METADADOS
# ============================================================

elif module == "Metadados":

    st.subheader(
        "Raster"
    )

    st.json(
        raster_info(
            selected_scene
        )
    )

    st.subheader(
        "Aquisição Planet"
    )

    st.json(
        {
            "Scene ID":
                sid,

            "Data":
                properties.get(
                    "acquired"
                ),

            "Instrumento":
                properties.get(
                    "instrument"
                ),

            "Satélite":
                properties.get(
                    "satellite_id"
                ),

            "Qualidade":
                properties.get(
                    "quality_category"
                ),

            "Cloud %":
                properties.get(
                    "cloud_percent"
                ),

            "Clear %":
                properties.get(
                    "clear_percent"
                ),

            "Light haze %":
                properties.get(
                    "light_haze_percent"
                ),

            "Heavy haze %":
                properties.get(
                    "heavy_haze_percent"
                ),

            "Shadow %":
                properties.get(
                    "shadow_percent"
                ),

            "Sun elevation":
                properties.get(
                    "sun_elevation"
                ),

            "Sun azimuth":
                properties.get(
                    "sun_azimuth"
                ),

            "View angle":
                properties.get(
                    "view_angle"
                ),

            "Pixel resolution":
                properties.get(
                    "pixel_resolution"
                ),
        }
    )

    st.subheader(
        "Coeficientes DN → Reflectância TOA"
    )

    if reflectance_coefficients:

        coefficient_table = (
            pd.DataFrame(
                [
                    {
                        "Banda":
                            f"B{band}",

                        "Nome":
                            BAND_NAMES[
                                band
                            ],

                        "Comprimento de onda (nm)":
                            BAND_WAVELENGTHS[
                                band
                            ],

                        "Coeficiente":
                            coefficient,
                    }

                    for band, coefficient
                    in sorted(
                        reflectance_coefficients.items()
                    )
                ]
            )
        )

        st.dataframe(
            coefficient_table,
            width="stretch",
            hide_index=True
        )

    else:

        st.warning(
            "Coeficientes não encontrados "
            "no XML."
        )

    st.subheader(
        "ACOLITE rhos"
    )

    if acolite_ready:

        st.success(
            "As 8 bandas ACOLITE rhos "
            "foram encontradas para esta cena."
        )

        st.code(
            str(
                acolite_folder
            )
        )

    else:

        st.info(
            "Esta cena ainda não possui "
            "as 8 bandas ACOLITE rhos exportadas."
        )

    with st.expander(
        "Metadata JSON completo"
    ):

        st.json(
            metadata
        )


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    f"Projeto: {ROOT}"
)
