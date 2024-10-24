# data_loader.py
import os
import cv2
import numpy as np

def load_data(image_dir, mask_dir):
    images = []
    masks = []

    image_files = os.listdir(image_dir)
    mask_files = os.listdir(mask_dir)

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        #mask_file = img_file.replace('.jpg', '.png')  # Cambia la extensión de .jpg a .png si las máscaras son PNG
        mask_file = img_file
        mask_path = os.path.join(mask_dir, mask_file)

        print(f"Procesando imagen: {img_path}, máscara: {mask_path}")  # Para depuración

        # Cargar y procesar la imagen
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: No se pudo cargar la imagen: {img_path}")
            continue
        img = cv2.resize(img, (256, 256))
        img = img / 255.0  # Normalizar
        images.append(img)

        # Cargar y procesar la máscara
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: No se pudo cargar la máscara: {mask_path}")
            continue
        mask = cv2.resize(mask, (256, 256))
        mask = mask / 255.0  # Normalizar
        mask = np.expand_dims(mask, axis=-1)  # Expandir dimensiones para que tenga un canal
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    return images, masks
