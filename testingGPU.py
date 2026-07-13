import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

try:
    import cupy as cp
    logging.info(f"CuPy importato con successo!")
    
    # Controlliamo i dettagli del driver e quanti dispositivi vede
    devices = cp.cuda.runtime.getDeviceCount()
    logging.info(f"Numero di GPU CUDA rilevate da CuPy: {devices}")
    
    if devices > 0:
        with cp.cuda.Device(0):
            logging.info(f"GPU in uso: {cp.cuda.Device(0).attributes}")
            # Facciamo un piccolo calcolo di prova
            x = cp.array([1, 2, 3])
            print("Risultato test GPU:", x * 2)
            
except Exception as e:
    logging.error(f"Errore durante l'inizializzazione di CuPy: {e}", exc_info=True)