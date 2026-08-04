# Tutorial sobre GPU y CUDA    

## 1. Presentación   
Todos los ordenadores disponen de una CPU (Central Processing Unit), que es el componente encargado de ejecutar los programas y coordinar el funcionamiento del sistema. La CPU está diseñada para realizar una gran variedad de tareas diferentes y resolverlas de forma muy eficiente. Aunque los procesadores actuales disponen de varios núcleos, normalmente su número es reducido (entre 4 y 16 en la mayoría de los ordenadores personales), ya que cada núcleo es muy potente y está preparado para ejecutar instrucciones complejas.    
 
Además de la CPU, muchos ordenadores incorporan una GPU (Graphics Processing Unit). Originalmente, las GPU se diseñaron para acelerar operaciones con gráficos y videojuegos, que requieren realizar millones de operaciones matemáticas sobre píxeles y objetos en tres dimensiones. Para conseguirlo, una GPU dispone de cientos o incluso miles de núcleos de procesamiento mucho más simples que los de una CPU. Cada uno de estos núcleos puede realizar una pequeña parte del trabajo de manera simultánea, permitiendo ejecutar el mismo cálculo sobre una gran cantidad de datos al mismo tiempo.    

<img width="443" height="309" alt="image" src="https://github.com/user-attachments/assets/23f08eeb-b79a-4cfb-8db2-25d4a196922e" />    
  
Esta capacidad de procesamiento paralelo hace que las GPU sean extraordinariamente rápidas para determinadas aplicaciones. Hoy en día se utilizan no solo para gráficos, sino también para el entrenamiento de modelos de inteligencia artificial, el reconocimiento de imágenes, el procesamiento de vídeo, las simulaciones científicas, el cálculo numérico, la minería de datos y muchas otras aplicaciones donde es necesario realizar millones de operaciones similares. En este tipo de problemas, una GPU puede acelerar la ejecución entre varias veces y varios cientos de veces respecto a una CPU, dependiendo del algoritmo utilizado y del modelo de GPU.    
 
Los programas escritos en Python también pueden aprovechar la potencia de una GPU. Para ello existen diferentes bibliotecas capaces de enviar parte del trabajo a la tarjeta gráfica en lugar de ejecutarlo en la CPU. En el caso de las tarjetas gráficas NVIDIA, la tecnología que hace posible este proceso se denomina CUDA (Compute Unified Device Architecture). CUDA proporciona un conjunto de herramientas y bibliotecas que permiten escribir programas capaces de utilizar la GPU como un procesador de propósito general. Bibliotecas muy conocidas como PyTorch, TensorFlow, CuPy o algunas funciones de OpenCV pueden utilizar CUDA de forma transparente, permitiendo que un mismo programa se ejecute mucho más rápido cuando dispone de una GPU compatible.     
 
En este tutorial aprenderemos cómo comprobar si nuestro ordenador dispone de una GPU compatible con CUDA, cómo preparar el entorno de programación y cómo modificar algunos programas de Python para que puedan aprovechar este recurso. Veremos que, con pequeños cambios en el código, tareas de cálculo, de reconocimiento de imágenes mediante redes neuronales o determinados algoritmos de visión artificial pueden ejecutarse de forma considerablemente más rápida que utilizando únicamente la CPU.    

## 2. Comprobar si nuestro portátil dispone de una GPU NVIDIA    

Antes de escribir programas que aprovechen la GPU, lo primero que debemos hacer es comprobar si nuestro ordenador dispone de una tarjeta gráfica NVIDIA. Si el equipo únicamente incorpora una GPU de Intel o AMD, los ejemplos de este tutorial basados en CUDA no podrán utilizar la aceleración por GPU (aunque existen otras tecnologías para esos fabricantes).    

_Método 1_. Comprobar la GPU desde el Administrador de dispositivos   

En Windows, la forma más sencilla de averiguar qué tarjeta gráfica tiene el ordenador es abrir el Administrador de dispositivos.
1.	Pulsa las teclas Windows + X.
2.	Selecciona Administrador de dispositivos.
3.	Despliega el apartado Adaptadores de pantalla.
Si el ordenador dispone de una GPU NVIDIA aparecerá un nombre similar a alguno de estos:

•	NVIDIA GeForce MX350    
•	NVIDIA GeForce RTX 3050 Laptop GPU    
•	NVIDIA GeForce RTX 4060 Laptop GPU     

Si únicamente aparecen dispositivos Intel o AMD, el ordenador no dispone de una GPU NVIDIA dedicada.    

_Método 2_. Utilizar el Administrador de tareas    

Otra forma muy cómoda consiste en abrir el Administrador de tareas (Ctrl + Mayús + Esc) y seleccionar la pestaña Rendimiento.    
 
En la parte izquierda aparecerán todos los dispositivos disponibles. Normalmente veremos algo parecido a:    

•	GPU 0 → Intel UHD Graphics    
•	GPU 1 → NVIDIA GeForce RTX 4060    

Este método también permite comprobar la memoria de vídeo (VRAM) disponible y observar la carga de trabajo de la GPU mientras se ejecutan programas.   

En el caso de que el ordenador si que tenga una GPU NVIDIA es muy probable que tenga instalado el driver de NVIDIA correspondiente. En todo caso, veamos cómo instalar la versión más actual de ese driver.    

## 3. Instalar el controlador de NVIDIA   

Seguiremos los pasos siguientes (ver las figuras de soporte):    

<img width="1618" height="531" alt="image" src="https://github.com/user-attachments/assets/0c8d1848-166c-41bd-acee-757c3c807d05" />




_Paso 1_. Comprobar el modelo de la GPU    

Antes de instalar el controlador, debemos conocer el modelo de nuestra tarjeta gráfica. Podemos averiguarlo desde el Administrador de dispositivos o desde el Administrador de tareas, tal como se explicó en el apartado anterior.    

_Paso 2_. Descargar el controlador    

Accede a la página oficial de descarga de controladores de NVIDIA y selecciona el modelo de tu tarjeta gráfica, el sistema operativo (Windows 10 o Windows 11) y la arquitectura correspondiente.     

Una vez seleccionado el modelo, descarga el controlador más reciente recomendado para tu GPU.    

_Paso 3_. Instalar el controlador     

Ejecuta el programa descargado y sigue el asistente de instalación.   

En la mayoría de los casos basta con aceptar las opciones predeterminadas. Durante el proceso la pantalla puede parpadear varias veces, ya que Windows reiniciará el controlador gráfico.    

Al finalizar la instalación es recomendable reiniciar el ordenador.    

_Paso 4_. Verificar la instalación    

Después de reiniciar, abre una ventana de PowerShell o del Símbolo del sistema y ejecuta:   
```
nvidia-smi
```

Aparecerá una tabla similar a esta:   
+-----------------------------------------------------------+   
| NVIDIA-SMI 582.xx            Driver Version: 582.xx          
| CUDA Version: 13.0                                           
+-----------------------------------------------------------+   
| GPU  Name                  Memory-Usage                      
| 0    GeForce RTX 4060      320 MiB / 8188 MiB                
+-----------------------------------------------------------+   

Esta información nos indica:   
•	Modelo de la GPU (GeForce RTX 4060)    
•	Versión del controlador instalada (582.xx)   
•	Versión de CUDA soportada por el controlador (13.0)   
•	Memoria de vídeo (VRAM) disponible (8GB de los cuales están ocupados en ese momento 320MB)   

## 4. Más información sobre la GPU   

Las operaciones anteriores ya nos habrán proporcionado información importante sobre la GPU:  

Modelo (por ejemplo, GeForce RTX 4065)   
La memoria disponible en la GPU   

Mas información relevante puede obtenerse consultando los datos de la GPU en internet. Por ejemplo, para el caso de GeForce RTX 4065 es fácil encontrar algo como muestra la tabla.    
<img width="393" height="294" alt="image" src="https://github.com/user-attachments/assets/35a2297c-d5b7-4ade-867d-14b4252ae779" />

 
El número de cores nos indica la potencia de cálculo de la GPU. También es importante la capacidad de la VRAM porque limita el tamaño de los datos que se pueden llevar y traer de la memoria del ordenador a la GPU para hacer allí los cálculos.    

También es importante verificar si la GPU es compatible con CUDA. Con toda probabilidad lo será, pero se puede verificar consultando en Google “Compatibilidad CUDA".

 <img width="600" height="900" alt="image" src="https://github.com/user-attachments/assets/d75a17c4-83e2-4e48-8937-cfcae8f2da73" />

Para el caso de GeForce RTC 4060 el nivel de compatibilidad es 8.9. Cuanto mayor sea ese número, más modernas son las características CUDA que soporta. Esto permite utilizar funciones y optimizaciones que no están disponibles en GPU de generaciones anteriores.   

## 5. Comprobar que Python puede utilizar la GPU   

Una vez instalado el controlador de NVIDIA, el siguiente paso consiste en comprobar que Python puede acceder a la GPU. Para ello utilizaremos PyTorch, una de las bibliotecas más utilizadas en Inteligencia Artificial y Visión por Computador.    

En el entorno virtual de nuestro proyecto realizaremos la siguiente instalación:   
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Después ejecutaremos el siguiente programa (el código está en el repositorio con el nombre _test_GPU.py_):   

```
import torch

print("-" * 50)
print("Comprobación de la GPU")
print("-" * 50)

print("Versión de PyTorch :", torch.__version__)
print("CUDA disponible    :", torch.cuda.is_available())

if torch.cuda.is_available():

    print("Nombre de la GPU  :", torch.cuda.get_device_name(0))

    propiedades = torch.cuda.get_device_properties(0)

    print("Memoria           : {:.2f} GB".format(
        propiedades.total_memory / (1024**3)))

    print("Multiprocesadores :", propiedades.multi_processor_count)

else:

    print("\nPython utilizará únicamente la CPU.")
```
Si todo está correctamente instalado, el resultado será parecido al siguiente:    

--------------------------------------------------    
Comprobación de la GPU    

--------------------------------------------------    
Versión de PyTorch : 2.x.x    
CUDA disponible    : True    
Nombre de la GPU   : NVIDIA GeForce RTX 4060 Laptop GPU    
Memoria            : 8.00 GB    
Multiprocesadores  : 24    

Si, por el contrario, aparece:    

CUDA disponible : False    

significa que Python no puede utilizar la GPU. Las causas más habituales son:   

•	El ordenador no dispone de una GPU NVIDIA.   
•	El controlador de NVIDIA no está instalado correctamente.   
•	Se ha instalado una versión de PyTorch sin soporte para CUDA.   

El programa muestra varios datos interesantes sobre la GPU:   

•	Versión de PyTorch instalada.   
•	Disponibilidad de CUDA, que indica si Python puede utilizar la GPU.    
•	Nombre del dispositivo gráfico.    
•	Memoria de vídeo (VRAM) disponible.    
•	Número de multiprocesadores (Streaming Multiprocessors o SM) de la GPU.   

Aunque PyTorch no muestra directamente el número de núcleos CUDA, esta información es suficiente para comprobar que el ordenador está preparado para ejecutar programas acelerados por GPU.   

A partir de este momento ya podemos comenzar a ejecutar programas que realicen cálculos sobre la GPU en lugar de utilizar únicamente la CPU.    

## 6. Primer programa utilizando la GPU     

En el apartado anterior comprobamos que Python puede acceder a la GPU. Ahora vamos a ejecutar un programa muy sencillo que nos permitirá observar la diferencia entre realizar un cálculo en la CPU y realizar el mismo cálculo en la GPU.    

Como ejemplo utilizaremos la multiplicación de dos matrices de gran tamaño. Este tipo de operación aparece continuamente en Inteligencia Artificial, Visión Artificial, Aprendizaje Automático y cálculo científico, por lo que constituye una buena prueba del rendimiento de una GPU. El código es este (también disponible en el repositorio con el nombre _test_multiplicacion_matrices.py_) :   

```
import torch
import time

# ==========================================================
# Comparación CPU - GPU
# ==========================================================

N = 5000

print("-" * 60)
print("Comparación CPU - GPU")
print("-" * 60)

# ----------------------------------------------------------
# Crear matrices en la CPU
# ----------------------------------------------------------

A = torch.rand(N, N)
B = torch.rand(N, N)

# ==========================================================
# CÁLCULO EN LA CPU
# ==========================================================

inicio = time.perf_counter()

C = torch.matmul(A, B)

fin = time.perf_counter()

tiempo_cpu = fin - inicio

print(f"Tiempo de cálculo en CPU : {tiempo_cpu:.3f} s")

# ==========================================================
# CÁLCULO EN LA GPU
# ==========================================================

if torch.cuda.is_available():

    print("\nUtilizando GPU:", torch.cuda.get_device_name(0))

    # ------------------------------------------------------
    # 1. Copiar datos a la GPU
    # ------------------------------------------------------

    inicio = time.perf_counter()

    A_gpu = A.to("cuda")
    B_gpu = B.to("cuda")

    torch.cuda.synchronize()

    fin = time.perf_counter()

    tiempo_copia = fin - inicio

    # ------------------------------------------------------
    # 2. Multiplicar matrices en la GPU
    # ------------------------------------------------------

    inicio = time.perf_counter()

    C_gpu = torch.matmul(A_gpu, B_gpu)

    torch.cuda.synchronize()

    fin = time.perf_counter()

    tiempo_gpu = fin - inicio

    # ------------------------------------------------------
    # 3. Copiar resultado a la CPU
    # ------------------------------------------------------

    inicio = time.perf_counter()

    C = C_gpu.to("cpu")

    torch.cuda.synchronize()

    fin = time.perf_counter()

    tiempo_retorno = fin - inicio

    # ------------------------------------------------------
    # Resumen
    # ------------------------------------------------------

    print("\n--------------- RESULTADOS ----------------")

    print(f"Copia CPU → GPU      : {tiempo_copia:.3f} s")
    print(f"Cálculo en GPU       : {tiempo_gpu:.3f} s")
    print(f"Copia GPU → CPU      : {tiempo_retorno:.3f} s")

    tiempo_total = tiempo_copia + tiempo_gpu + tiempo_retorno

    print(f"\nTiempo total GPU     : {tiempo_total:.3f} s")
    print(f"Tiempo total CPU     : {tiempo_cpu:.3f} s")

    print(f"\nAceleración          : {tiempo_cpu/tiempo_total:.2f}x")

else:

    print("\nNo hay ninguna GPU CUDA disponible.")
```

Primero se crean dos matrices aleatorias de 5000 × 5000 elementos. A continuación se multiplican utilizando la CPU y se mide el tiempo necesario para realizar la operación. Después, las dos matrices se copian a la memoria de la GPU mediante las instrucciones:  

```
A_gpu = A.to("cuda")
B_gpu = B.to("cuda")
```

A partir de ese momento, ambas matrices ya estan en la memoria de la tarjeta gráfica. La multiplicación se realiza completamente en la GPU mediante la instrucción:   
```
C_gpu = torch.matmul(A_gpu, B_gpu)
```
Antes de detener el cronómetro se ejecuta:
```
torch.cuda.synchronize()
```

Esta instrucción obliga a Python a esperar a que la GPU haya terminado todos los cálculos pendientes. Si no se utilizara, el tiempo medido sería incorrecto, ya que la GPU trabaja de forma asíncrona y Python continuaría ejecutando instrucciones sin esperar a que el cálculo hubiera finalizado.    

El programa también toma datos del tiempo que se necesita para trasladar las matrices de la memoria de la GPU y recuperar el resultado.   

Dependiendo de la potencia de la GPU, es posible que para matrices no muy grandes la CPU sea aun más rápida. Pero usando ese mismo programa y aumentando el valor de N se puede verificar que a partir de cierto tamaño la GPU es mucho más rápida que la CPU, incluso teniendo en cuenta los tiempos necesarios para mover las matrices de una memoria a otra.   

Por otra parte es interesante hacer la prueba con el portátil conectado a la red electrica y repetirla con el portatil desconectado (solo bateria). Muy probablemente se observará que las conclusiones son muy diferenes porque al trabajar solo con bateria Windows aplica políticas muy agresivas de ahorro de energía, como por ejemplo, reducir mucho la frecuencia de trabajo tanto de la CPU como, sobre todo, de la GPU.    

Este ejemplo pone de manifiesto un aspecto fundamental de la programación con GPU: la GPU no acelera automáticamente todos los programas de Python.
Un programa solo utilizará la GPU cuando los datos se encuentren en su memoria y las operaciones sean realizadas por bibliotecas capaces de ejecutar código CUDA, como PyTorch. En los próximos apartados veremos que este mismo mecanismo es el que utilizan las redes neuronales y los programas de reconocimiento de imágenes para acelerar su ejecución.    

## 7. Acelerar el reconocimiento de objetos   

El reconocimiento de objetos mediante una red neuronal es otro ejemplo de tarea que puede beneficiarse mucho del uso de la GPU. Para comprobarlo usaremos el siguiente programa en Python (se encuentra en el repositorio con el nombre _test_reconocimiento_objetos.py_), que requiere la instalación de las siguientes librerías:
```
pip install opencv-python ultralytics pandas tqdm seaborn scipy
```
```
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

IMAGE_FILE = "imagen.png"      # Cambia por una imagen cualquiera
BATCH_SIZES = [1, 2, 4, 8, 16, 32]

# Cargar imagen
img = cv2.imread(IMAGE_FILE)
if img is None:
    raise Exception(f"No se puede abrir {IMAGE_FILE}")
print(f"Imagen: {img.shape[1]}x{img.shape[0]}")


# Probar CPU y GPU

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
 
        # Medir tiempo
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
```

El programa usa una red neuronal pre-entrenada capaz de reconocer hasta 80 objetos diferentes. Toma una imagen y genera batches de diferentes tamaños con esa misma imagen repetida. Se la entrega a la red neuronal para que en esas imágenes reconozca los objetos. Por tanto, hay que hacer mucho cálculo que puede hacerse en paralelo en la GPU. El programa primero procesa las imágenes en la CPU y después en la GPU y va indicando el tiempo que necesita en cada caso para procesar cada uno de los batches. Los datos mostrarán una abrumadora aceleración en el caso de la GPU. También se genera al final una imagen que muestra el resultado del reconocimiento de objetos simplemente para verificar que la red neuronal ha hecho su trabajo (encontrar el libro en la playa).    

Es importante comprender en este ejemplo que lo que enviamos a la GPU es la red neuronal, mediante la operación:    
```
model.to(device)
```
Cuando después hacemos las inferencias para reconocer objetos, con la operación:
```
results = model(images, size=640)
```
el sistema envía a la memoria de la GPU la imagen, la procesa y recupera el resultado.

## 8. OpenCV y CUDA   

Los ejemplos que hemos visto antes muestran mejoras notables de velocidad porque estamos usando la librería PyTorch con soporte para CUDA para hacer operaciones de cálculo y de inferencia con redes neuronales. Otras librerías también tienen soporte para CUDA para acelerar sus operaciones. Ese es el caso de OpenCV que tiene muchas funciones de procesado de imagen que se benefician mucho del soporte para CUDA.  

Es importante observar que ya hemos usado OpenCV en el ejemplo anterior para procesar imágenes (cargar/salvar imágenes de/en ficheros o añadir a la imagen el recuadro que identifica el objeto reconocido). Pero esas operaciones no usan CUDA porque la librería estándar de OpenCV para Python (que es la que se ha usado en el ejemplo) no tiene soporte para CUDA.    

Lamentablemente instalar una versión de OpenCV con soporte para CUDA no es tan fácil como en el caso de PyTorch. De hecho, hoy por hoy es necesario hacer un proceso complejo de compilación de las fuentes de OpenCV que está fuera del alcance de este tutorial (aunque pueden encontrarse fácilmente videos tutoriales de cómo hacerlo).   

Una vez conseguida esa instalación de OpenCV con soporte para CUDA es posible acelerar operaciones de procesado de imagen como muestra el código siguiente para convertir una imagen a escala de grises (en el repositorio este código tiene el nombre _test_opencv_cuda.py_):    

```
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
```








 

