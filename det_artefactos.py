# -*- coding: utf-8 -*-
"""
Proyecto deteccion artefactos
"""


from PIL import Image
from numpy import *
import os
import matplotlib.pyplot as plt
from skimage.feature import match_template
from skimage.feature import peak_local_max
#import numpy as np
import scipy.stats as st

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = diff(st.norm.cdf(x))
    kernel_raw = sqrt(outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

# Lectura imagen 1
im1_gray = array(Image.open(os.path.abspath("prueba2\\recorteRGB1.png")).convert('L'))
im1_rgb = array(Image.open(os.path.abspath("prueba2\\recorteRGB1.png")))

# Lectura imagen 2
im2_gray = array(Image.open(os.path.abspath("prueba2\\recorteRGB2.png")).convert('L'))
im2_rgb = array(Image.open(os.path.abspath("prueba2\\recorteRGB2.png")))

# plantilla gaussiana (similar a artefacto)
template = gkern(21,7)
template = template.max() - template
template =  15 * template/template.max()
plt.figure()
gr = cm = plt.get_cmap('gray')
plt.imshow(template,cmap=gr)


# Correlacion cruzada normalizada
ncc1 = match_template(im1_gray, template, pad_input=True)
ncc2 = match_template(im2_gray, template, pad_input=True)

# threshold
th = 0.65
belowth1 = ncc1 < th
ncc1_th = ncc1
ncc1_th[belowth1] = 0

belowth2 = ncc2 < th
ncc2_th = ncc2
ncc2_th[belowth2] = 0

# Non-maximum suppression
nms_im1 = peak_local_max(ncc1_th,min_distance=10,indices=False)
nms_im2 = peak_local_max(ncc2_th,min_distance=10,indices=False)

ind_nms_im1 = peak_local_max(ncc1_th,min_distance=10,indices=True)
ind_nms_im2 = peak_local_max(ncc2_th,min_distance=10,indices=True)

# Detecciones finales: artefactos detectado en ambas imagenes
detecciones = zeros(im1_gray.shape)

max_dist = 20 
for i in range(len(ind_nms_im1)):
    for j in range(len(ind_nms_im2)):
        if linalg.norm(ind_nms_im1[i]-ind_nms_im2[j]) <= max_dist:
            detecciones[ind_nms_im1[i,0],ind_nms_im1[i,1]] = 1

ind_detecciones = where(detecciones)

# Marcar detecciones en imagenes de entrada
y_det = ind_detecciones[0]
x_det = ind_detecciones[1]

fig_rgb1,ax1 = plt.subplots(1)
ax1.imshow(im1_rgb)
for i in range(len(ind_detecciones[0])):
    rect = plt.Rectangle((x_det[i]-12, y_det[i]-12), 25, 25, edgecolor='b', facecolor='none')
    ax1.add_patch(rect)


fig_rgb2,ax2 = plt.subplots(1)
ax2.imshow(im2_rgb)
for i in range(len(ind_detecciones[0])):
    rect = plt.Rectangle((x_det[i]-12, y_det[i]-12), 25, 25, edgecolor='b', facecolor='none')
    ax2.add_patch(rect)









