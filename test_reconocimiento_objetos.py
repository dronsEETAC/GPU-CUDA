# ============================================================================
# benchmark_yolo_batch.py
#
# Compara CPU y GPU procesando lotes de imágenes con YOLOv5.
# Instalar opencv-python
# ============================================================================

import time
import cv2
import torch


import os

# ---------------------------------------------------------------
# Dibujar detecciones y guardar imagen
# ---------------------------------------------------------------

def guardar_resultado(imagen, results, nombre_fichero):

    salida = imagen.copy()

    for *box, conf, cls in results.xyxy[0]:

        x1, y1, x2, y2 = map(int, box)

        etiqueta = f"{results.names[int(cls)]} {conf:.2f}"

        cv2.rectangle(salida,
                      (x1, y1),
                      (x2, y2),
                      (0,255,0),
                      2)

        cv2.putText(salida,
                    etiqueta,
                    (x1, max(20,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

    cv2.imwrite(nombre_fichero, salida)

# ---------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------

IMAGE_FILE = "imagen.png"      # Cambia por una imagen cualquiera

BATCH_SIZES = [1, 2, 4, 8, 16, 32]

# ---------------------------------------------------------------
# Cargar imagen
# ---------------------------------------------------------------

img = cv2.imread(IMAGE_FILE)

if img is None:
    raise Exception(f"No se puede abrir {IMAGE_FILE}")

print(f"Imagen: {img.shape[1]}x{img.shape[0]}")

# ---------------------------------------------------------------
# Probar CPU y GPU
# ---------------------------------------------------------------

for device in ["cpu", "cuda"]:

    if device == "cuda" and not torch.cuda.is_available():
        continue

    print("\n")
    print("=" * 70)
    print(f"DISPOSITIVO: {device.upper()}")

    model = torch.hub.load(
        "ultralytics/yolov5",
        "yolov5s",
        pretrained=True
    )

    model.to(device)
    model.eval()



    for batch in BATCH_SIZES:

        # Crear un lote con la misma imagen repetida
        images = [img] * batch

        #
        # Calentamiento (warm-up)
        #
        '''_ = model(images, size=640)

        if device == "cuda":
            torch.cuda.synchronize()'''

        #
        # Medir tiempo
        #
        t0 = time.perf_counter()

        results = model(images, size=640)

        if device == "cuda":
            torch.cuda.synchronize()

        t1 = time.perf_counter()

        tiempo = t1 - t0

        ips = batch / tiempo

        print(
            f"Batch {batch:2d}   "
            f"Tiempo: {tiempo:6.3f} s   "
            f"Imágenes/s: {ips:6.1f}"
        )




guardar_resultado(
    img,
    results,
    f"resultado.jpg"
)