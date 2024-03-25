import pandas as pd
import numpy as np
# from sklearn.mixture import GaussianMixture
import multiprocessing
from multiprocessing.pool import MapResult
from Utils import remove_splchars,clean
from pyarrow.parquet import ParquetFile
import pyarrow as pa


def get_data():
    data = ParquetFile("paraquet/train.parquet")
    batch = next(data.iter_batches(batch_size = 1000)) 
    batch = pa.Table.from_batches([batch]).to_pandas()
    batch.drop(["id"], axis=1, inplace=True)

    #pool = multiprocessing.Pool()
    with multiprocessing.Pool(5) as p:
        batch["article"] = p.map(remove_splchars,batch["article"])
    
    
    return batch