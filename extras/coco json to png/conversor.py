from pycocotools.coco import COCO
import numpy as np
import cv2
import os

# Cargar el archivo JSON de COCO
coco = COCO("D:/Bibliotecas/Proyectos/codes/Python/CraftzAPI/Resourses/Dataset v3/train/_annotations.coco.json")

# Directorio para guardar las máscaras
output_dir = "D:/Bibliotecas/Proyectos/codes/Python/CraftzAPI/Resourses/Dataset v3/train_mask"
os.makedirs(output_dir, exist_ok=True)

# Obtener todas las imágenes anotadas
image_ids = coco.getImgIds()

for img_id in image_ids:
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # Crear una máscara vacía
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

    # Agregar las máscaras de cada objeto en la imagen
    for ann in anns:
        rle = coco.annToRLE(ann)
        m = coco.annToMask(ann)
        mask[m == 1] = 255  # Puedes usar el valor 1 o el valor de la clase si tienes varias

    # Guardar la máscara como una imagen
    #mask_path = os.path.join(output_dir, img_info['file_name'].replace(".jpg", ".png"))
    mask_path = os.path.join(output_dir, img_info['file_name'])
    cv2.imwrite(mask_path, mask)
