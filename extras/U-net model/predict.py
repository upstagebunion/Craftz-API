# predict.py
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Cargar el modelo entrenado
model = load_model('D:/Bibliotecas/Proyectos/codes/Python/CraftzAPI/unet_model_final.h5')

# Cargar la imagen a predecir
image = cv2.imread('D:/Bibliotecas/Proyectos/codes/Python/CraftzAPI/Resourses/Dataset/test/n03595614_2694_JPEG_jpg.rf.f8319deb9162fc775f17001d95ed58b2.jpg')
cv2.imshow('Imagen Original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
image_resized = cv2.resize(image, (256, 256))  # Redimensionar si es necesario

# Normalizar la imagen si es necesario
image_resized = image_resized / 255.0

# Añadir la dimensión del batch si es necesario
image_resized = np.expand_dims(image_resized, axis=0)

# Predecir usando el modelo
prediction = model.predict(image_resized)

# Si es una máscara de una sola clase, podrías usar esto para convertir la predicción en una imagen binaria:
predicted_mask = (prediction > 0.5).astype(np.uint8)

print(f"Valor mínimo de la predicción: {prediction.min()}")
print(f"Valor máximo de la predicción: {prediction.max()}")


# Visualizar la máscara predicha o guardarla
cv2.imshow('Predicted Mask', predicted_mask[0])
cv2.waitKey(0)