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

<img width="960" height="314" alt="Imagen6" src="https://github.com/user-attachments/assets/13640185-5c73-4f33-98f1-f7bf3783a3c2" />


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



