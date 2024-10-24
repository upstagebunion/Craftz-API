# train.py
from model import unet_model
from data_loader import load_data
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Rutas de los datos
image_dir = 'D:/Bibliotecas/Proyectos/codes/Python/CraftzAPI/Resourses/Dataset v3/train'
mask_dir = 'D:/Bibliotecas/Proyectos/codes/Python/CraftzAPI/Resourses/Dataset v3/train_mask'

# Cargar los datos
X, y = load_data(image_dir, mask_dir)

# Dividir en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Crear el modelo
model = unet_model()
#model = load_model('D:/Bibliotecas/Proyectos/codes/Python/CraftzAPI/unet_model_v2.keras') #para cuando el modelo tenga que cargarse y no sobreeescribirse

# Definir callbacks
checkpoint = ModelCheckpoint('unet_model_v2.keras', monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Entrenar el modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=8,
    epochs=1,
    callbacks=[checkpoint, early_stop]
)

# Guardar el modelo final
model.save('unet_model_final.keras')
