from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

def run_jobs(dataset, numThreads, funName, debug=False):
    splits = np.array_split(dataset, numThreads)
    futures = []
    with ThreadPoolExecutor(numThreads) as executor:        
        for dslist in splits:
            a_result =  executor.submit(funName, dslist)
            futures.append(a_result)

    if debug is True:
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
            pass

