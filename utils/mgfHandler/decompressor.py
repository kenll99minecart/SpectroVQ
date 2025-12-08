from utils.Compression import batchprocessor
from utils.Compression import compress
from utils.Compression import postprocessing
from pyteomics import mgf
import os
import torch
import numpy as np
import pandas as pd
import pickle

DEFAULTMGFPARAMSKEYS = ['title','rtinseconds','pepmass','charge','scans','seq','modifications']
class MGFDecompressor():
    def __init__(self,model,inputFileName,outputfilemgf = None, batch_size = None,quantizer = 6, store_compounded = True,
                 gzip_compression_level = None,zlib_compression_level = 6):
        """
        Initialize the MGFDecompressor for decompressing mass spectrometry data.

        Args:
            model: Neural network model for decoding spectra
            inputFileName (str): Path to the compressed input file (.vqms2 format)
            outputfilemgf (str, optional): Path for output MGF file. Defaults to 'output.mgf'.
            batch_size (int, optional): Number of spectra to process in each batch. Defaults to 128.
            quantizer (int): Number of quantization levels used during compression. Defaults to 6.
            store_compounded (bool): Whether to process chimeric spectra. Defaults to True.
            gzip_compression_level (int, optional): Compression level for gzip. Defaults to None.
            zlib_compression_level (int): Compression level for zlib. Defaults to 6.
        """
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
        if store_compounded:
            self.store_compounded = True
            self.compounded_codes_list = []
            if os.path.exists(self.inputFileNameWithoutExt + '_compounded.pkl'):
                with open(self.inputFileNameWithoutExt + '_compounded.pkl','rb') as f:
                    self.compounded_idx_list = pickle.load(f)
            else:
                Warning('Compounded file not found, compounded spectra will not be processed')
                self.store_compounded = False
        self.compounded_original_codes_dict = {}

    @staticmethod
    def changeSpectrumName(name):
        """
        Modify spectrum name to indicate it's a compounded spectrum.

        Args:
            name (str): Original spectrum name

        Returns:
            str: Modified spectrum name with '_compounded' suffix added to the first part
        """
        name_chunks = name.split('.')
        name_chunks[0] += '_compounded'
        name = '.'.join(name_chunks)
        return name

    # Mode 1: Load All Spectra, then perform decompression
    def DecompressAll(self):
        """
        Decompress all spectra from compressed format to MGF.

        This method loads all compressed codes and metadata, then decodes them
        back to m/z and intensity arrays using the neural network model.
        All spectra are processed in memory at once.
        """
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
        """
        Decompress compressed spectra to MGF format using batch-wise processing.

        This method processes spectra in batches to manage memory usage efficiently.
        It reads compressed codes and metadata, decodes them back to m/z and intensity
        arrays, and writes the reconstructed spectra to an MGF file. Also handles
        chimeric spectra if they were stored during compression.

        Args:
            verbose (int): Verbosity level for progress reporting. Defaults to 0.
        """
        # self.CodesList = []
        df_metadata = pd.read_parquet(self.metadataFileName,engine='fastparquet')
        for idx in range(0,df_metadata.shape[0],self.batch_size):
            metaDataBatch = df_metadata.iloc[idx:idx+self.batch_size]
            output_codes = np.zeros((metaDataBatch.shape[0],self.quantizer,self.model.quantizedlen),dtype=np.int32)
            if verbose >= 1:
                if (idx % 1000) == 0:
                    print(f'Processing Spectrum Batch {idx}, Title: {metaDataBatch["title"].values[0]}')
            for j in range(metaDataBatch.shape[0]): # Batch Number
                for k in range(self.quantizer): # Number of quantizers
                    for l in range(self.model.quantizedlen): # Number of integers in each quantizer
                        output_codes[j,k,l] = self.depacker.pull()
                if (j + idx* self.batch_size) in self.compounded_idx_list:
                    self.compounded_original_codes_dict[str(j + idx* self.batch_size)] = output_codes[j,:]
            output_codes = np.transpose(output_codes,(1,0,2)) # B N D -> N B D
            if 'ExtendedMZ' in metaDataBatch.columns:
                bp = batchprocessor.SpectrumBatchDecoder(output_codes,self.model,MaxValList=metaDataBatch['MaxVal'].values,
                leftovermzList=metaDataBatch['ExtendedMZ'],leftoverintensityList=metaDataBatch['ExtendedIT'])
            else:
                bp = batchprocessor.SpectrumBatchDecoder(output_codes,self.model,MaxValList=metaDataBatch['MaxVal'].values)
            output_mz, output_intensity = bp.getReconstructSpectrum()
            metaDataBatch.pop('MaxVal')
            ListOfSpectrum = []
            for k in range(len(output_mz)):
                # Round m/z values to 2 decimal places
                spectrum = {'m/z array':postprocessing.changeTofloat(list(output_mz[k])),'intensity array':np.array(list(output_intensity[k]),np.float64)}
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
        if self.store_compounded:
            if verbose >=1:
                print('Processing compounded spectra')
            if len(self.compounded_idx_list) == 0:
                Warning('No compounded spectra found')
                return
            compounded_indices = np.array(self.compounded_idx_list)
            compounded_codes = np.zeros((compounded_indices.shape[0],self.quantizer,self.model.quantizedlen),dtype=np.int32) #Fixed quantizers
            for j in range(compounded_indices.shape[0]): # Batch Number
                for k in range(self.quantizer): # Number of quantizers
                    for l in range(self.model.quantizedlen): # Number of integers in each quantizer
                        compounded_codes[j,k,l] = self.depacker.pull()
            
            compounded_codes = np.transpose(compounded_codes,(1,0,2))
            compounded_max_value_list = df_metadata['MaxVal'].values[compounded_indices]
            compounded_df = df_metadata.iloc[compounded_indices]
            for j in range(0,compounded_codes.shape[0],self.batch_size): # Batched Spectrum Decompression
                if verbose >=1:
                    print(f'Processing compounded spectrum batch {j}')
                batched_compounded_codes = compounded_codes[j:j+self.batch_size]
                batched_compounded_max_value_list = compounded_max_value_list[j:j+self.batch_size]
                batched_original_codes_list = []
                for k in range(len(batched_compounded_max_value_list)):
                    batched_original_codes_list.append(self.compounded_original_codes_dict[str(compounded_indices[j+k])])
                batch_original_codes_vector = np.transpose(np.array(batched_original_codes_list,dtype=np.int32),(1,0,2))
                if 'ExtendedMZ' in metaDataBatch.columns:
                    bp = batchprocessor.SpectrumBatchDecoder(batched_compounded_codes,self.model,MaxValList=batched_compounded_max_value_list,
                    leftovermzList=compounded_df['ExtendedMZ'],leftoverintensityList=compounded_df['ExtendedIT'])
                else:
                    bp = batchprocessor.SpectrumBatchDecoder(batched_compounded_codes,self.model,MaxValList=batched_compounded_max_value_list)
                output_mz, output_intensity = bp.getReconstructSpectrum(batch_original_codes_vector)
                metaDataBatch = compounded_df.iloc[j:j+self.batch_size]
                metaDataBatch = metaDataBatch.reset_index(drop=True)
                metaDataBatch.pop('MaxVal')
                # Add the spectrum into the a dictionary
                ListOfSpectrum = []
                for k in range(len(batched_compounded_max_value_list)):
                    spectrum = {'m/z array':postprocessing.changeTofloat(list(output_mz[k])),'intensity array':np.array(list(output_intensity[k]),np.float64)}
                    spectrum.update({'params':{key:metaDataBatch[key].values[k] for key in metaDataBatch.keys()}})
                    spectrum['params'].update({'title':self.changeSpectrumName(metaDataBatch['title'].values[k])})
                    spectrum['params'].update({'pepmass':(spectrum['params']['pepmz'], spectrum['params']['pepit'])})
                    spectrum['params'].pop('pepmz')
                    spectrum['params'].pop('pepit')
                    ListOfSpectrum.append(spectrum)
                mgf.write(ListOfSpectrum,self.OutputFileMGF, file_mode ='a')
