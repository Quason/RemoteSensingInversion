# coding=UTF-8
'''
@Author: cuiqiyuan
@Date: 2020-06-21 19:17:26
@Description: aquatic plants detecttion based on Sentinel-2/MSI
'''
import os
import sys
import argparse

from osgeo import gdal, ogr
import numpy as np
import skimage.io
import cv2 as cv

def array2tif(array_srs, mask, raster_fn_dst, geo_trans, pro_jref, type='float'):
    driver = gdal.GetDriverByName('GTiff')
    if len(np.shape(array_srs)) == 2:
        nbands = 1
    else:
        nbands = np.shape(array_srs)[2]
    if type == 'uint8':
        target_ds = driver.Create(
            raster_fn_dst, np.shape(array_srs)[1], np.shape(array_srs)[0], nbands,
            gdal.GDT_Byte
        )
        mask_value = 255
    elif type == 'int':
        target_ds = driver.Create(
            raster_fn_dst, np.shape(array_srs)[1], np.shape(array_srs)[0], nbands,
            gdal.GDT_Int16
        )
        mask_value = -999
    else:
        target_ds = driver.Create(
            raster_fn_dst, np.shape(array_srs)[1], np.shape(array_srs)[0], nbands,
            gdal.GDT_Float32
        )
        mask_value = -999
    target_ds.SetGeoTransform(geo_trans)
    target_ds.SetProjection(pro_jref)
    if nbands == 1:
        array_srs[mask] = mask_value
        target_ds.GetRasterBand(1).WriteArray(array_srs)
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(mask_value)
    else:
        for i in range(nbands):
            target_ds.GetRasterBand(i+1).WriteArray(array_srs[:,:,i])
    target_ds = None


def vector2mask(raster_fn, vector_fn):
    """resterize vector layer

    Args:
        raster_fn (str): source raster file
        vector_fn (str): source vector file
    """    
    raster = gdal.Open(raster_fn)
    data = raster.GetRasterBand(1).ReadAsArray()
    raster_fn_dst = raster_fn.replace('.tif', '') + '_mask.tif'
    raster_dst = gdal.GetDriverByName('GTiff').Create(
        raster_fn_dst, np.shape(data)[1], np.shape(data)[0], 1, gdal.GDT_Byte)
    raster_dst.SetGeoTransform(raster.GetGeoTransform())
    raster_dst.SetProjection(raster.GetProjectionRef())
    raster_dst.GetRasterBand(1).SetNoDataValue(255)
    vector = ogr.Open(vector_fn)
    layer = vector.GetLayer()
    gdal.RasterizeLayer(raster_dst, [1], layer, burn_values=[0])
    raster_dst = None
    raster = None
    mask = skimage.io.imread(raster_fn_dst) != 0
    os.remove(raster_fn_dst)
    return(mask)


def main(raster_fn, path_dst, vector_fn=None):
    """main function
    """    
    print('data loading ...')
    srs_ds = gdal.Open(raster_fn)
    band_green = (srs_ds.GetRasterBand(2).ReadAsArray().astype(float))/10000.0 # 560nm
    band_red = (srs_ds.GetRasterBand(3).ReadAsArray().astype(float))/10000.0 # 665nm
    band_705 = (srs_ds.GetRasterBand(4).ReadAsArray().astype(float))/10000.0
    band_740 = (srs_ds.GetRasterBand(5).ReadAsArray().astype(float))/10000.0
    band_nir = (srs_ds.GetRasterBand(7).ReadAsArray().astype(float))/10000.0 # 842nm
    band_swir = (srs_ds.GetRasterBand(9).ReadAsArray().astype(float))/10000.0 # 1610nm
    mci = (band_705-band_red) - (band_740-band_red) * (705-665)/(740-665)
    ndvi = (band_nir-band_red) / (band_nir+band_red)
    fai = (band_nir-band_red) - (band_swir-band_red) * (842-665)/(1610-665)
    mndwi = (band_green-band_swir) / (band_green+band_swir)
    if vector_fn is None:
        mask = mndwi < 0
    else:
        mask = vector2mask(raster_fn, vector_fn)
    print('decision tree ...')
    thresh_ndvi = -0.1
    thresh_mci = 0.005
    thresh_fai = 0.01
    mask = mask + (ndvi<thresh_ndvi)
    print('opening operator ...')
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    opening = cv.morphologyEx(
        (mask<0.5).astype(np.uint8), cv.MORPH_OPEN, kernel, iterations=1)
    mask = opening == 0
    fuye = (ndvi>thresh_ndvi) * (mci<thresh_mci) * (fai>=thresh_fai)
    chenshui = (ndvi>thresh_ndvi) * (mci<thresh_mci) * (fai<thresh_fai)
    algae = (ndvi>thresh_ndvi) * (mci>=thresh_mci)
    plants = np.zeros((np.shape(band_red)), np.uint8)
    plants[fuye] = 1
    plants[chenshui] = 2
    plants[algae] = 3
    print('deal with spatial relationship ...')
    # labels[0]: count; labels[1]: label;
    # labels[2]: [minY,minX,block_width,block_height,cnt]
    labels_struc = cv.connectedComponentsWithStats((mask<0.5).astype(np.uint8), connectivity=8)
    labels = [[i, labels_struc[2][i][4]] for i in range(labels_struc[0])]
    labels.sort(key=lambda x: x[1], reverse=True)
    for item in labels[1:20]:
        cnt_fuye = len(plants[(labels_struc[1] == item[0]) * (plants == 1)])
        cnt_chenshui = len(plants[(labels_struc[1] == item[0]) * (plants == 2)])
        cnt_algae = len(plants[(labels_struc[1] == item[0]) * (plants == 3)])
        if cnt_fuye + cnt_chenshui > 0:
            if cnt_algae / (cnt_fuye + cnt_chenshui + cnt_algae) > 0.2:
                plants[labels_struc[1] == item[0]] = 3
    print('export results ...')
    geo_trans = srs_ds.GetGeoTransform()
    pro_jref = srs_ds.GetProjectionRef()
    raster_fn_dst = os.path.join(
        path_dst,
        os.path.split(raster_fn)[1].replace('.tif', '')+'_aquaticPlants.tif'
    )
    array2tif(plants, mask, raster_fn_dst, geo_trans, pro_jref, 'uint8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Satellite Data Preprocess')
    parser.add_argument('--ifile', type=str, help='original albedo file (after preprocess)')
    parser.add_argument('--opath', type=str, default='', help='export path')
    parser.add_argument('--vector', type=str, default='', help='vector file for masking')
    args = parser.parse_args()
    opath = args.opath if args.opath else os.path.split(args.ifile)[0]
    vector = args.vector if args.vector else None
    main(args.ifile, opath, vector)
