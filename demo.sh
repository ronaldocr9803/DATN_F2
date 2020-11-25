python3 -m pip install geopandas
python3 -m pip install rasterio
python3 -m pip install slidingwindow
export PYTHONPATH=.
python3 demo/predict_tiff_img.py \
    --image_path=../data/W05_202003281250_RI_RSK_RSKA003603_RGB/W05_202003281250_RI_RSK_RSKA003603_RGB_0.tif \
    --shapefile_path=../data/W05_202003281250_RI_RSK_RSKA003603_RGB/W05_202003281250_RI_RSK_RSKA003603_RGB.shp \
    --evaluate=True
