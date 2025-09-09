from utils.Compression import batchprocessor
from utils.Compression import compress
from pyteomics import mgf
import os
import torch
import numpy as np
import pandas as pd
import pickle

DEFAULTMGFPARAMSKEYS = ['title','rtinseconds','pepmass','charge','scans','seq','modifications']
class MGFDecompressor():
    def __init__(self,model,inputFileName,outputfilemgf = None, batch_size = None,quantizer = 6, store_chimeric = True,
                 gzip_compression_level = None,zlib_compression_level = 6):
        self.model = model
        if outputfilemgf is None:
            self.OutputFileMGF = 'output.mgf'
        else:
            self.OutputFileMGF = outputfilemgf
            assert self.OutputFileMGF.endswith('.mgf'), 'Output file must be in MGF format'
        self.inputFileName = inputFileName
        self.device = next(self.model.parameters()).device
        assert os.path.exists(self.inputFileName), 'Input file does not exist'
        self.inputFileNameWithoutExt = inputFileName.split('.')[0]
        self.depacker = compress.BitUnpacker(10, open(self.inputFileName, 'rb'),gzip_compression=gzip_compression_level, zlib_compression=zlib_compression_level)
        self.metadataFileName = self.inputFileNameWithoutExt + '.parquet' 
        self.MaxValList = []
        if batch_size is None:
            Warning('Batch size is not defined; using default batch size of 128')
            self.batch_size = 128
        else:
            self.batch_size = batch_size
        self.quantizer = quantizer
        if store_chimeric:
            self.store_chimeric = True
            self.chimeric_codes_list = []
            self.chimeric_idx_list = []
            if os.path.exists(self.inputFileNameWithoutExt + '_chimeric.pkl'):
                with open(self.inputFileNameWithoutExt + '_chimeric.pkl','rb') as f:
                    chimericIdx = pickle.load(f)
            else:
                Warning('Chimeric file not found, chimeric spectra will not be processed')
                self.store_chimeric = False
    
    @staticmethod
    def changeSpectrumName(name):
        name_chunks = name.split('.')
        name_chunks[0] += '_chimeric'
        name = '.'.join(name_chunks)
        return name

    # Mode 1: Load All Spectra, then perform decompression
    def DecompressAll(self):
        self.CodesList = []
        df_metadata = pd.read_parquet(self.metadataFileName,engine='fastparquet')
        for metaDataBatch in range(df_metadata.shape[0]):
            metaDataBatch = metaDataBatch.to_dict(orient='records')
            for j in range(len(metaDataBatch)):
                self.CodesList.append(self.depacker.pull())
            self.CodesList = np.array(self.CodesList)
            self.CodesList = np.transpose(self.CodesList,(1,0,2))
            bp = batchprocessor.SpectrumBatchDecoder(self.CodesList,self.model,MaxValList=metaDataBatch['MaxVal'].values)
            output_mz, output_intensity = bp.getReconstructSpectrum()
            metaDataBatch.pop('MaxVal')
            for k in range(len(output_mz)):
                # Round m/z values to 2 decimal places
                formatted_mz = np.round(output_mz[k], 2)
                spectrum = {'m/z array':formatted_mz,'intensity array':output_intensity[k]}
                spectrum.update({'params':{key:metaDataBatch[key].values[k] for key in metaDataBatch.keys()}})
                mgf.write([spectrum],self.OutputFileMGF)


    # Mode 2: Load Spectra in Batches, then perform decompression
    def Decompress(self,verbose = 0):
        # self.CodesList = []
        df_metadata = pd.read_parquet(self.metadataFileName,engine='fastparquet')
        for idx in range(0,df_metadata.shape[0],self.batch_size):
            metaDataBatch = df_metadata.iloc[idx:idx+self.batch_size]
            output_codes = np.zeros((metaDataBatch.shape[0],self.quantizer,18),dtype=np.int32) #Fixed quantizers and 18 dimensions
            if verbose >= 1:
                if (idx % 1000) == 0:
                    print(f'Processing Spectrum Batch {idx}, Title: {metaDataBatch["params"]["title"].values[0]}')
            for j in range(metaDataBatch.shape[0]): # Batch Number
                for k in range(self.quantizer): # Number of quantizers
                    for l in range(18): # Number of integers in each quantizer
                        output_codes[j,k,l] = self.depacker.pull()
            output_codes = np.transpose(output_codes,(1,0,2)) # B N D -> N B D
            bp = batchprocessor.SpectrumBatchDecoder(output_codes,self.model,MaxValList=metaDataBatch['MaxVal'].values)
            output_mz, output_intensity = bp.getReconstructSpectrum()
            metaDataBatch.pop('MaxVal')
            ListOfSpectrum = []
            for k in range(len(output_mz)):
                # Round m/z values to 2 decimal places
                spectrum = {'m/z array':output_mz[k],'intensity array':output_intensity[k]}
                spectrum.update({'params':{key:metaDataBatch[key].values[k] for key in metaDataBatch.keys()}})
                spectrum['params'].update({'pepmass':(spectrum['params']['pepmz'], spectrum['params']['pepit'])})

                spectrum['params'].pop('pepmz')
                spectrum['params'].pop('pepit')
                ListOfSpectrum.append(spectrum)
            if idx > 0:
                mgf.write(ListOfSpectrum,self.OutputFileMGF, file_mode ='a')
            else:
                mgf.write(ListOfSpectrum,self.OutputFileMGF)
            # self.CodesList.clear()
        # Process chimeric spectra
        if self.store_chimeric:
            if len(self.chimeric_idx_list) == 0:
                Warning('No chimeric spectra found')
                return
            chimeric_indices = np.array(self.chimeric_idx_list)
            chimeric_codes = np.zeros((chimeric_indices.shape[0],self.quantizer,18),dtype=np.int32) #Fixed quantizers and 18 dimensions
            for j in range(chimeric_indices.shape[0]): # Batch Number
                for k in range(self.quantizer): # Number of quantizers
                    for l in range(18): # Number of integers in each quantizer
                        chimeric_codes[j,k,l] = self.depacker.pull()
            chimeric_codes = np.transpose(chimeric_codes,(1,0,2))
            chimeric_max_value_list = df_metadata['MaxVal'].values[chimeric_indices]
            bp = batchprocessor.SpectrumBatchDecoder(chimeric_codes,self.model,MaxValList=chimeric_max_value_list)
            output_mz, output_intensity = bp.getReconstructSpectrum()
            metaDataBatch = df_metadata.iloc[chimeric_indices]
            metaDataBatch = metaDataBatch.reset_index(drop=True)
            metaDataBatch.pop('MaxVal')
            ListOfSpectrum = []
            for k in range(len(output_mz)):
                spectrum = {'m/z array':output_mz[k],'intensity array':output_intensity[k]}
                spectrum.update({'params':{key:metaDataBatch[key].values[k] for key in metaDataBatch.keys()}})
                spectrum.update({'params':{'title':self.changeSpectrumName(metaDataBatch['params']['title'].values[k])}})
                spectrum['params'].update({'pepmass':(spectrum['params']['pepmz'], spectrum['params']['pepit'])})
                spectrum['params'].pop('pepmz')
                spectrum['params'].pop('pepit')
                ListOfSpectrum.append(spectrum)
            mgf.write(ListOfSpectrum,self.OutputFileMGF, file_mode ='a')
        self.depacker.close()
