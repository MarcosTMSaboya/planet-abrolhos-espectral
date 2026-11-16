# PlanetScope: análise exploratória de assinaturas espectrais

Projeto em andamento para inspecionar e comparar assinaturas espectrais em imagens PlanetScope, com apoio do ACOLITE, Python e Streamlit. A aplicação reúne ferramentas de inspeção de cenas, qualidade de pixels, análise exploratória de agrupamentos, revisão espacial e preparação de pontos para verificação.

O foco é apoiar a investigação de padrões espectrais em ambientes costeiros. Os resultados e rótulos gerados são exploratórios: este repositório não apresenta uma classificação ecológica validada nem uma estimativa independente de acurácia.

## Tecnologias

- Python, Streamlit, NumPy e Pandas
- Rasterio, GeoPandas e Folium
- ACOLITE para processamento de imagens, configurado e executado à parte

## Conteúdo

- `app.py`: interface de inspeção e análise
- `src/`: módulos de leitura, qualidade, análise espectral e espacial, classificação exploratória e apoio à verificação
- `requirements*.txt`: dependências Python

## Dados e execução

Este repositório contém código e dependências, sem cenas PlanetScope, arquivos ACOLITE, coordenadas, camadas de campo, produtos derivados ou arquivos do QGIS. As fontes de dados têm condições de uso e redistribuição próprias e devem permanecer em armazenamento local autorizado.

Para iniciar a aplicação, instale as dependências adequadas ao seu ambiente geoespacial e execute:

```powershell
python -m streamlit run app.py
```

O aplicativo precisa de cenas e camadas compatíveis configuradas localmente. Por padrão, procura cenas em `PSScene/` e centroides em `points/centroids.gpkg`. Para usar outros locais, configure `PLANET_CENTROID_PATH` e, opcionalmente, `PLANET_SECONDARY_SCENE_DIR` no ambiente antes da execução. Não adicione arquivos `.env`, credenciais ou dados pessoais ao repositório.

## Limitações

A análise está em desenvolvimento. Agrupamentos, curvas e resultados cartográficos servem à inspeção e formulação de hipóteses; não devem ser interpretados como identificação confirmada de habitats ou substratos. A validação independente depende de referências de campo adequadas e ainda não é demonstrada neste repositório.

## Publicação

Esta cópia foi preparada como repositório privado. Antes de qualquer abertura pública, revisar as condições de redistribuição das imagens e camadas, créditos, dependências e a remoção de informação interna que não faça parte do portfólio.
