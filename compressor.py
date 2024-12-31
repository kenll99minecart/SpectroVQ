from utils.Compression import batchprocessor
from utils.Compression import compress
from pyteomics import mgf
import os
import torch
import numpy as np
import pandas as pd
import pyarrow as pa

DEFAULTMGFPARAMSKEYS = ['title','rtinseconds','pepmass','charge','scans','seq','modifications']
class MGFCompressor():
    def __init__(self,model,mgfFilePath,OutputFileName = None,batch_size = None,quantizer = 6):
        self.model = model
        self.mgfFilePath = mgfFilePath
        self.OutputFileName = OutputFileName
        if self.OutputFileName is None:
            self.OutputFileName = os.path.basename(mgfFilePath).split('.')[0] 
            Warning(f"Output file name is not defined; using input name {self.OutputFileName} as output name")
        self.mgfFile = mgf.read(mgfFilePath)
        self.device = next(self.model.parameters()).device
        self.packer = compress.BitPacker(10, open(self.OutputFileName + '.vqms2', 'wb'))
        self.quantizer = quantizer
        self.metadataFileName = self.OutputFileName + '.parquet' 
        self.MaxValList = []

        if batch_size is None:
            Warning('Batch size is not defined; using default batch size of 128')
            self.batch_size = 128
        else:
            self.batch_size = batch_size

    def LoadAll(self):
        mzList, intensityList,paramList = [],[],[]
        for spectrum in self.mgfFile:
            mzList.append(spectrum['m/z array'])
            intensityList.append(spectrum['intensity array'])
            paramList.append(spectrum['params'])
        self.mzList = mzList
        self.intensityList = intensityList
        self.paramList = paramList
        self.MaxValList = []

    def CompressAll(self):
        self.LoadAll()
        for i in range(0,len(self.mzList),self.batch_size):
            bp = batchprocessor.SpectrumBatchEncoder(self.mzList[i:i+self.batch_size],self.intensityList[i:i+self.batch_size],self.model)
            output_codes = bp.getReconstructIndices(outputOutOfRange = False,quantizer = self.quantizer)
            self.MaxValList.extend(bp.MaxValList)
            output_codes = np.transpose(np.array(output_codes.cpu().numpy(),dtype=np.int16),(1,0,2))
            for j in range(len(output_codes)):
                self.packer.push(output_codes[j])
        if len(self.mzList) % self.batch_size != 0:
            bp = batchprocessor.SpectrumBatchEncoder(self.mzList[i:],self.intensityList[i:],self.model)
            output_codes = bp.getReconstructIndices(outputOutOfRange = False,quantizer = self.quantizer)
            output_codes = np.transpose(np.array(output_codes.cpu().numpy(),dtype=np.int16),(1,0,2))
            self.MaxValList.extend(bp.MaxValList)
            for j in range(len(output_codes)):
                self.packer.push(output_codes[j])
        self.packer.flush()
    
    def Compress(self):
        mzList, intensityList,paramsList = [],[],[]
        for _ , spectrum in enumerate(self.mgfFile):
            mzList.append(spectrum['m/z array'])
            intensityList.append(spectrum['intensity array'])
            paramsList.append(spectrum['params'])
            if len(mzList) == self.batch_size:
                bp = batchprocessor.SpectrumBatchEncoder(mzList,intensityList,self.model)
                output_codes = bp.getReconstructIndices(outputOutOfRange = False,quantizer = self.quantizer)
                self.MaxValList.extend(bp.MaxValList)
                output_codes = np.transpose(np.array(output_codes.cpu().numpy(),dtype=np.int16),(1,0,2)) # N B D -> B N D
                for j in range(self.batch_size):
                    self.packer.push(output_codes[j])
                df_meta = pd.DataFrame.from_dict(paramsList)
                df_meta['pepmass'] = df_meta['pepmass'].astype(str)
                df_meta['charge'] = df_meta['charge'].apply(lambda x: x[0])
                df_meta['MaxVal'] = self.MaxValList
                self.MaxValList.clear()
                if os.path.exists(self.metadataFileName):
                    df_meta.to_parquet(self.metadataFileName, engine='fastparquet', append=True)
                else:
                    df_meta.to_parquet(self.metadataFileName, engine='fastparquet')
                mzList.clear(), intensityList.clear(),paramsList.clear()
        if len(mzList) > 0:
            bp = batchprocessor.SpectrumBatchEncoder(mzList,intensityList,self.model)
            output_codes = bp.getReconstructIndices(outputOutOfRange = False,quantizer = self.quantizer)
            self.MaxValList.extend(bp.MaxValList)
            output_codes = np.transpose(np.array(output_codes.cpu().numpy(),dtype=np.int16),(1,0,2))
            for j in range(len(mzList)):
                self.packer.push(output_codes[j])
            df_meta = pd.DataFrame.from_dict(paramsList)
            df_meta['pepmass'] = df_meta['pepmass'].astype(str)
            df_meta['charge'] = df_meta['charge'].apply(lambda x: x[0])
            df_meta['MaxVal'] = self.MaxValList
            self.MaxValList.clear()
        self.packer.flush()