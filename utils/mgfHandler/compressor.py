from ..Compression import batchprocessor
from ..Compression import compress
from ..Compression import torchprocessor
from pyteomics import mgf
import os
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pickle


DEFAULTMGFPARAMSKEYS = ['title','rtinseconds','pepmass','charge','scans','seq','modifications']
class MGFCompressor():
    def __init__(self,model,mgfFilePath,OutputFileName = None,batch_size = None,quantizer = 6, store_chimeric = True,
                 gzip_compression_level = 6):
        self.model = model
        self.mgfFilePath = mgfFilePath
        self.OutputFileName = OutputFileName
        if self.OutputFileName is None:
            self.OutputFileName = os.path.basename(mgfFilePath).split('.')[0] 
            Warning(f"Output file name is not defined; using input name {self.OutputFileName} as output name")
        self.mgfFile = mgf.read(mgfFilePath)
        self.device = next(self.model.parameters()).device
        self.packer = compress.BitPacker(10, open(self.OutputFileName + '.vqms2', 'wb'), gzip_compression=gzip_compression_level)
        self.gzip_compression_level = gzip_compression_level
        self.quantizer = quantizer
        self.metadataFileName = self.OutputFileName + '.parquet' 
        self.MaxValList = []

        if batch_size is None:
            Warning('Batch size is not defined; using default batch size of 128')
            self.batch_size = 128
        else:
            self.batch_size = batch_size
        if store_chimeric:
            self.store_chimeric = True
            self.chimeric_codes_list = []
            self.chimeric_idx_list = []

    def ProcessMetaData(self):
        df_meta = pd.DataFrame.from_dict(self.paramList)
        # df_meta['pepmass'] = df_meta['pepmass'].astype(str)
        df_meta['pepmz'] = df_meta['pepmass'].apply(lambda x: x[0])
        df_meta['pepit'] = df_meta['pepmass'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
        df_meta['charge'] = df_meta['charge'].apply(lambda x: x[0])
        df_meta['MaxVal'] = self.MaxValList
        df_meta.pop('pepmass')
        df_meta.to_parquet(self.metadataFileName,engine='fastparquet',compression = {'method': 'gzip', 'compresslevel': self.gzip_compression_level})

    # Mode 1: Load all spectra into MEMORY, then perform compression
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

    def CompressAll(self,num_workers = 32):
        self.LoadAll()
        tb = torchprocessor.SpectrumTorchBatchEncoder(self.mzList,self.intensityList,self.model,chimeric = self.store_chimeric)
        output_codes = tb.getReconstructIndices(outputOutOfRange = False,quantizer = self.quantizer,batch_size = self.batch_size, num_workers = num_workers)
        if self.store_chimeric:
            raw_codes,chimeric_codes,chimericIdx = output_codes
            self.chimeric_codes_list.extend(chimeric_codes)
            self.chimeric_idx_list.extend(chimericIdx)
            output_codes = raw_codes
        self.MaxValList.extend(tb.MaxValList)
        output_codes = np.transpose(np.array(output_codes.cpu().numpy(),dtype=np.int32),(1,0,2)) # N B D -> B N D
        for j in range(output_codes.shape[0]): # Data Size
            for k in range(self.quantizer):
                for l in range(18): # Number of integers in each quantizer
                    self.packer.push(output_codes[j,k,l]) # First in First out
        if self.store_chimeric:
            for j in range(len(self.chimeric_codes_list)):
                for k in range(self.quantizer): # Number of quantizers
                    for l in range(18): # Number of integers in each quantizer
                        self.packer.push(self.chimeric_codes_list[j][k,l])
        self.packer.flush()
        self.ProcessMetaData()
        if self.store_chimeric:
            with open(self.OutputFileName + '_chimeric.pkl', 'wb') as f:
                pickle.dump(self.chimeric_idx_list, f)
    

    def ProcessSpectra(self,mz,intensity,idx = 0):
        bp = batchprocessor.SpectrumBatchEncoder(mz,intensity,self.model,chimeric = self.store_chimeric)
        output_codes = bp.getReconstructIndices(outputOutOfRange = False,quantizer = self.quantizer)
        if self.store_chimeric:
            raw_codes,chimeric_codes,chimericIdx = output_codes
            self.chimeric_codes_list.extend(chimeric_codes)
            arrChimericIdx = np.array(chimericIdx,dtype=np.int32)
            self.chimeric_idx_list.extend(arrChimericIdx  + (idx * self.batch_size))
            output_codes = raw_codes
        self.MaxValList.extend(bp.MaxValList)
        output_codes = np.transpose(np.array(output_codes.cpu().numpy(),dtype=np.int32),(1,0,2)) # N B D -> B N D 
        for j in range(output_codes.shape[0]): # Batch Number
            for k in range(self.quantizer): # Number of quantizers
                for l in range(18): # Number of integers in each quantizer
                    self.packer.push(output_codes[j,k,l]) # First in First out

    # Mode 2: Batch-wise compression
    def Compress(self,verbose = 0,debug = False):
        mzList, intensityList,paramsList = [],[],[]
        for spectrumidx,spectrum in enumerate(self.mgfFile):
            mzList.append(spectrum['m/z array'])
            intensityList.append(spectrum['intensity array'])
            paramsList.append(spectrum['params'])
            if verbose >= 1:
                if (spectrumidx % 1000) == 0:
                    print(f'Processing Spectrum {spectrumidx}, Title: {spectrum["params"]["title"]}')
            if len(mzList) == self.batch_size:
                self.ProcessSpectra(mzList,intensityList,spectrumidx)
                df_meta = pd.DataFrame.from_dict(paramsList)
                # df_meta['pepmass'] = df_meta['pepmass'].astype(str)
                df_meta['pepmz'] = df_meta['pepmass'].apply(lambda x: x[0])
                df_meta['pepit'] = df_meta['pepmass'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
                df_meta['charge'] = df_meta['charge'].apply(lambda x: x[0])
                df_meta['MaxVal'] = self.MaxValList
                df_meta.pop('pepmass')
                
                if os.path.exists(self.metadataFileName) & ((spectrumidx // self.batch_size) > 0):
                    df_meta.to_parquet(self.metadataFileName, engine='fastparquet', compression = {'method': 'gzip', 'compresslevel': self.gzip_compression_level}, append=True)#, 'mtime': 1
                else:
                    df_meta.to_parquet(self.metadataFileName, engine='fastparquet', compression = {'method': 'gzip', 'compresslevel': self.gzip_compression_level},append = False)#, 'mtime': 1
                mzList.clear(), intensityList.clear(),paramsList.clear()
                self.MaxValList.clear()
            if debug & (spectrumidx >= 3):
                break
        # Process Last Batch
        if len(mzList) > 0:
            if verbose >=1:
                print(f'Processing Last Batch of size {len(mzList)}')
            self.ProcessSpectra(mzList,intensityList,spectrumidx)
            df_meta = pd.DataFrame.from_dict(paramsList)
            # df_meta['pepmass'] = df_meta['pepmass'].astype(str)
            df_meta['pepmz'] = df_meta['pepmass'].apply(lambda x: x[0])
            df_meta['pepit'] = df_meta['pepmass'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
            df_meta['charge'] = df_meta['charge'].apply(lambda x: x[0])
            df_meta['MaxVal'] = self.MaxValList
            df_meta.pop('pepmass')
            
            if os.path.exists(self.metadataFileName) & ((spectrumidx // self.batch_size) > 0):
                df_meta.to_parquet(self.metadataFileName, engine='fastparquet',compression = {'method': 'gzip', 'compresslevel': self.gzip_compression_level},append=True)
            else:
                df_meta.to_parquet(self.metadataFileName, engine='fastparquet',compression = {'method': 'gzip', 'compresslevel': self.gzip_compression_level},append = False)
            self.MaxValList.clear()
        if self.store_chimeric:
            for j in range(len(self.chimeric_codes_list)):
                for k in range(self.quantizer): # Number of quantizers
                    for l in range(18): # Number of integers in each quantizer
                        self.packer.push(self.chimeric_codes_list[j][k,l])
        if verbose >=1:
            print(f'Flushing Packer')
        self.packer.flush()
        
        if self.store_chimeric:
            if verbose >=1:
                print(f'Processing Chimeric MetaData')
            # stored_dict = {
            #     'chimeric_codes':self.chimeric_codes_list,
            #     'chimeric_idx':self.chimeric_idx_list
            # }
            # pd.DataFrame.from_dict(stored_dict).to_parquet(self.OutputFileName + 'chimeric.parquet', engine='fastparquet')
            with open(self.OutputFileName + '_chimeric.pkl', 'wb') as f:
                pickle.dump(self.chimeric_idx_list, f)