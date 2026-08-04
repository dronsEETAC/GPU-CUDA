#
import cv2

info = cv2.getBuildInformation()

if "CUDA" in info:

    # Leer una imagen (CPU)
    imagen = cv2.imread("imagen.jpg")

    # Crear un objeto GpuMat
    gpu = cv2.cuda_GpuMat()

    # Copiar la imagen a la GPU
    gpu.upload(imagen)

    # Convertir a escala de grises en la GPU
    gris_gpu = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2GRAY)

    # Recuperar la imagen
    gris = gris_gpu.download()

    cv2.imshow("Original", imagen)
    cv2.imshow("Gris", gris)

    cv2.waitKey(0)
else:
    print("Parece que esta versión de OpenCV no tiene soporte para CUDA.")


