import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
from PIL import Image


def create_cloth_mask(original_name, source, dest):
    original_image = cv.imread(osp.join(source, 'images', original_name))
    original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
    # plt.imshow(original_image), plt.colorbar(),plt.show()

    # il grabcut funziona meglio con il blurring, i contorni sono più smooth
    blurred_image = cv.GaussianBlur(original_image, (5, 5), 0)

    # SEPARO COLORE BACKGROUND
    # estraggo il colore del pixel in alto a sx
    r0 = blurred_image[2, 2, 0]
    g0 = blurred_image[2, 2, 1]
    b0 = blurred_image[2, 2, 2]
    # estraggo i 3 piani di colore
    r = blurred_image[:, :, 0]
    g = blurred_image[:, :, 1]
    b = blurred_image[:, :, 2]
    # segmentazione: metto a 1 i pixel con colore uguale al primo (valori rgb uguali)
    segmented_image = np.where(r != r0, 0, 1)
    segmented_image = np.where(g != g0, 0, segmented_image)
    segmented_image = np.where(b != b0, 0, segmented_image)
    # plt.imshow(segmented_image), plt.colorbar(), plt.show()       # stampa immagine segmentata

    # CREAZIONE MASK
    # inizializzo la mask a BGD (valore 0)
    mask = np.zeros((original_image.shape[0], original_image.shape[1]), np.uint8)
    mask[:] = cv.GC_BGD

    # GESTIONE IMMAGINI PROBLEMATICHE
    # calcolo la percentuale di pixel di background
    back_perc = (segmented_image == 1).sum() / (original_image.shape[0] * original_image.shape[1])
    # print(back_perc)

    # se la % è abbastanza alta significa che la segmentazione funziona bene,
    # setto i pixel segmentati a 0 come probabile foreground PR_FGD (valore 3)
    if back_perc >= 0.15:
        mask[segmented_image == 0] = cv.GC_PR_FGD

    # se la % è bassa significa che la segmentazione non funziona bene
    # allora applico OTSU, i pixel a 0 vengono segnati come foregroung FGD (valore 1)
    else:
        ret, th = cv.threshold(cv.cvtColor(original_image, cv.COLOR_RGB2GRAY), 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # l'if serve per impostare il background a 0 e il foreground a 1, in alcune threshold venivano scambiati
        if th[0, 0] == 0:
            th = np.where(th == 0, 255, 0)
        mask[th == 0] = cv.GC_FGD
    # plt.imshow(mask), plt.colorbar(),plt.show()    # stampa maschera

    # GRAB CUT: definisco elementi di base per il grabcut
    # Definisco il boundary rectangle che contiene il foreground object
    height, width, _ = original_image.shape
    left_margin_proportion = 0.2
    right_margin_proportion = 0.2
    up_margin_proportion = 0.1
    down_margin_proportion = 0.1

    boundary_rectangle = (
        int(width * left_margin_proportion),
        int(height * up_margin_proportion),
        int(width * (1 - right_margin_proportion)),
        int(height * (1 - down_margin_proportion)),
    )

    # Arrays usati internamente dall'algoritmo
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    cv.grabCut(
        blurred_image,
        mask,
        boundary_rectangle,
        background_model,
        foreground_model,
        5,
        cv.GC_INIT_WITH_MASK+cv.GC_INIT_WITH_RECT   # non capisco se cambia qualcosa togliendo RECT
    )

    # grabcut_mask è la mashcera binaria finale, la ottengo dalla mask modificata
    grabcut_mask = np.where((mask == cv.GC_PR_BGD) | (mask == cv.GC_BGD), 0, 1).astype("uint8")
    #plt.imshow(grabcut_mask), plt.colorbar(), plt.show()    # stampa maschera grabcut
    im_grabcut_mask = Image.fromarray(grabcut_mask)
    im_grabcut_mask.save(osp.join(dest, 'cloth_masks', original_name.replace('.jpg', '.png')))