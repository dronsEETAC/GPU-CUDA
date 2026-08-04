import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin")

import cv2
import time

# ==========================================================
# Cargar imagen
# ==========================================================

imagen = cv2.imread("imagen.png")

if imagen is None:
    raise Exception("No se pudo abrir la imagen")

print("Resolución:", imagen.shape)

# ==========================================================
# CPU
# ==========================================================

inicio = time.perf_counter()

gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

bordes_cpu = cv2.Canny(
    gris,
    100,
    200
)

fin = time.perf_counter()

print(f"CPU : {(fin-inicio)*1000:.1f} ms")

# ==========================================================
# GPU
# ==========================================================

gpu = cv2.cuda_GpuMat()

gpu.upload(imagen)

# Conversión a gris
gpu_gris = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2GRAY)

# Crear detector Canny
canny = cv2.cuda.createCannyEdgeDetector(
    100,
    200
)

# ----------------------------
# Calentamiento
# ----------------------------

_ = canny.detect(gpu_gris)
_.download()

# ----------------------------
# Medición
# ----------------------------

inicio = time.perf_counter()

gpu_bordes = canny.detect(gpu_gris)

bordes_gpu = gpu_bordes.download()

fin = time.perf_counter()

print(f"GPU : {(fin-inicio)*1000:.1f} ms")


# ==========================================================
# Guardar resultados
# ==========================================================

cv2.imwrite("bordes_cpu2.jpg", bordes_cpu)
cv2.imwrite("bordes_gpu2.jpg", bordes_gpu)



