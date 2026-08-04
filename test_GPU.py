# Instalar pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

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
