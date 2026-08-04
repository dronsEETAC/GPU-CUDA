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
