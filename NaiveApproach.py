import time
import random
import sys
import logging
import numpy as np
from numba import jit


N = 1024

# Configurazione del Logging per HPC
# Usiamo sys.stdout per far confluire tutto nel file .out di Slurm in tempo reale
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Matrix Initialization

def matrix_number_init(matrix, N):
    for i in range(N):
        for j in range(N):
            matrix[i][j] = random.random()

def matrixes_init():

    logging.info(f"Allocazione e inizializzazione delle matrici (N = {N})...")

    A = np.random.rand(N,N).astype(np.float64)
    B = np.random.rand(N,N).astype(np.float64)

    C = np.zeros((N,N), dtype = np.float64) 

    return A, B, C


@jit(nopython=True)
def naive_approach(A, B, C):

    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]

@jit(nopython=True)
def exploiting_locality(A, B, C):

    for i in range(N):
        for k in range(N):
            for j in range(N):
                C[i][j] += A[i][k] * B[k][j]

@jit(nopython=True)
def transposed_approach(A, B, C):

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    C[i][j] += A[i][k] * B[j][k]


@jit(nopython=True)
def tiling_approach(A, B, C, block_size):

    for ih in range(0, N, block_size):
        for kh in range(0, N, block_size):
            for jh in range(0, N, block_size):

                # min in necessary in case the matrix is smaller than the sum of the 
                # the block and the already reached position
                for i in range(ih, min(ih + block_size, N)):
                               for k in range(kh, min(kh + block_size, N)):
                                              for j in range(jh, min(jh + block_size, N)):
                                                             C[i][j] += A[i][k] * B[k][j]


def main():
    
    logging.info(f"Allocazione e inizializzazione delle matrici (N = {N})...")

    A, B, C = matrixes_init()
    logging.info("------------------------------")
    logging.info("Starting Naive Computation")
    logging.info("------------------------------")

    start_time_naive = time.perf_counter()

    naive_approach(A, B, C)

    end_time_naive = time.perf_counter()  # Corretto refuso (perf_counter)

    timex_naive = (end_time_naive - start_time_naive) * 1000

    logging.info("------------------------------")
    logging.info("Finished Naive Computation")
    logging.info("------------------------------")
    logging.info(f"Time Naive: {timex_naive:.2f} ms")
    logging.info(f"Controllo elemento C[0][0]: {C[0][0]:.4f}")


    logging.info("                              ")
    logging.info("                              ")
    
    C = np.zeros((N,N), dtype=np.float64)

    logging.info("------------------------------")
    logging.info("Starting Locality Exploitation Computation")
    logging.info("------------------------------")

    start_time_local = time.perf_counter()
    exploiting_locality(A, B, C)
    end_time_local = time.perf_counter()
    timex_local = (end_time_local - start_time_local) * 1000

    logging.info("------------------------------")
    logging.info("Finished Locality Computation")
    logging.info("------------------------------")
    logging.info(f"Time Locality: {timex_local:.2f} ms")
    logging.info(f"Controllo elemento C[0][0]: {C[0][0]:.4f}")

    logging.info("                              ")
    logging.info("                              ")

    

    logging.info("------------------------------")
    logging.info("Starting Traspose Naive Computation")
    logging.info("------------------------------")
    
    C = np.zeros((N,N), dtype=np.float64)

    B_transposed = np.transpose(B).copy()

    start_time_transpose_naive = time.perf_counter()

    transposed_approach(A, B_transposed, C)

    end_time_transpose_naive = time.perf_counter()  # Corretto refuso (perf_counter)

    timex_transpose_naive = (end_time_transpose_naive - start_time_transpose_naive) * 1000

    logging.info("------------------------------")
    logging.info("Finished Transpose Naive Computation")
    logging.info("------------------------------")
    logging.info(f"Time Transpose: {timex_transpose_naive:.2f} ms")
    logging.info(f"Controllo elemento C[0][0]: {C[0][0]:.4f}")

    logging.info("                              ")
    logging.info("                              ")

    logging.info("------------------------------")
    logging.info("Starting Tiling Computation")
    logging.info("------------------------------")
   
    # Computing with multiple block size in order to understand the effect of changes on the tile parameter
    b_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    for b in b_sizes:

        C = np.zeros((N,N), dtype=np.float64)
        
        logging.info(f"Testing with tile block of {b}")
        
        start_time_b = time.perf_counter()

        tiling_approach(A, B, C, b)

        end_time_b = time.perf_counter()
        time_b = (end_time_b - start_time_b) * 1000 

        logging.info(f"Time Tiling with tile of {b}: {time_b:.4f} ms")
        logging.info(f"Controllo elemento C[0][0]: {C[0][0]:.4f}")

    logging.info("------------------------------")
    logging.info("Finished Tiling Computation")
    logging.info("------------------------------")

    
    logging.info("                              ")
    logging.info("                              ")


    


if __name__ == "__main__":
    main()
