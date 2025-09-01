from utils.Compression import batchprocessor
from utils.Compression import compress
from pyteomics import mgf
import os
import torch
import numpy as np
import pandas as pd
import pyarrow as pa

DEFAULTMGFPARAMSKEYS = ['title','rtinseconds','pepmass','charge','scans','seq','modifications']
class MGFDecompressor():
    def __init__(self,model,inputFileName,outputfilemgf = None, batch_size = None):
        self.model = model
        if outputfilemgf is None:
            self.OutputFileMGF = 'output.mgf'
        else:
            self.OutputFileMGF = outputfilemgf
            assert self.OutputFileMGF.endswith('.mgf'), 'Output file must be in MGF format'
        self.inputFileName = inputFileName
        self.device = next(self.model.parameters()).device
        self.depacker = compress.BitUnpacker(10, open(self.inputFileName + '.vqms2', 'rb'))
        self.metadataFileName = self.inputFileName + '.parquet' 
        self.MaxValList = []
        if batch_size is None:
            Warning('Batch size is not defined; using default batch size of 128')
            self.batch_size = 128
        else:
            self.batch_size = batch_size

    def Decompress(self):
        self.CodesList = []
        for _, metaDataBatch in enumerate(pd.read_parquet(self.metadataFileName, engine='fastparquet', chunksize = self.batch_size)):
            metaDataBatch = metaDataBatch.to_dict(orient='records')
            for j in range(len(metaDataBatch)):
                self.CodesList.append(self.depacker.pull())
            self.CodesList = np.array(self.CodesList)
            self.CodesList = np.transpose(self.CodesList,(1,0,2))
            bp = batchprocessor.SpectrumBatchDecoder(self.CodesList,self.model,MaxValList=metaDataBatch['MaxVal'].values)
            output_mz, output_intensity = bp.getReconstructSpectrum()
            metaDataBatch.pop('MaxVal')
            for k in range(len(output_mz)):
                spectrum = {'m/z array':output_mz[k],'intensity array':output_intensity[k]}
                spectrum.update({'params':{key:metaDataBatch[key].values[k] for key in metaDataBatch.keys()}})
                mgf.write([spectrum],self.OutputFileMGF)