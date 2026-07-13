import time
import sys
import logging 
import numpy as np
import cupy as cp

N = 2048

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s', handlers = [logging.StreamHandler(sys.stdout)])



def main():
    
    logging.info(f"-- Bencmarking Matrix Multiplication (N = {N}) --")

    t_transfer = time.perf_counter()


    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)

    C = np.zeros((N, N), dtype=np.float64)


    time_transfer = (time.perf_counter() - t_transfer) * 1000

    logging.info(f"Tempo di trasferimento {time_transfer:.4f} ms")

    logging.info("Esecuzione GPU")

    C = cp.dot(A, B)
    cp.cuda.Stream.null.synchronize()


    t1 = time.perf_counter()

    C = cp.dot(A, B)
    cp.cuda.Stream.null.synchronize()

    t2 = time.perf_counter()

    time_gpu = (t2-t1) * 1000

    logging.info(f"Tempo GPU {time_gpu:.2f} ms")
    logging.info(f"Controllo Elemento C[0,0]: {C[0][0]:.4f}")

    logging.info("Esecuzione completata!")



if __name__ == "__main__":
    main()
