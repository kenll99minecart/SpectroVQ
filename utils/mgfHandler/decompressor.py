from utils.Compression import batchprocessor
from utils.Compression import compress
from utils.Compression import postprocessing
from pyteomics import mgf
import os
import torch
import numpy as np
import pandas as pd
import pickle
import traceback

DEFAULTMGFPARAMSKEYS = ['title','rtinseconds','pepmass','charge','scans','seq','modifications']
class MGFDecompressor():
    def __init__(self,model,inputFileName,outputfilemgf = None, batch_size = None,quantizer = 6, store_compounded = True,
                 gzip_compression_level = None,zlib_compression_level = 6, zstd_compression_level = None):
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
        self.inputFileNameWithoutExt = '.'.join(inputFileName.split('.')[:-1])
        self.depacker = compress.BitUnpacker(10, open(self.inputFileName, 'rb'),gzip_compression=gzip_compression_level, zlib_compression=zlib_compression_level, zstd_compression = zstd_compression_level)
        self.metadataFileName = self.inputFileNameWithoutExt + '.parquet' 
        self.MaxValList = []
        if batch_size is None:
            Warning('Batch size is not defined; using default batch size of 128')
            self.batch_size = 128
        else:
            self.batch_size = batch_size
        self.quantizer = quantizer
        self.store_compounded = False
        if store_compounded:
            self.store_compounded = True
            self.compounded_codes_list = []
            if os.path.exists(self.inputFileNameWithoutExt + '_compounded.pkl'):
                with open(self.inputFileNameWithoutExt + '_compounded.pkl','rb') as f:
                    self.compounded_idx_list = pickle.load(f)
            else:
                Warning('Compounded file not found, compounded spectra will not be processed')
                
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
    def DecompressAll(self, verbose=0):
        """
        Decompress all spectra from compressed format to MGF.

        This method loads all compressed codes and metadata, then decodes them
        back to m/z and intensity arrays using the neural network model.
        All spectra are processed in memory at once.

        Args:
            verbose (int): Verbosity level for progress reporting. Defaults to 0.
        """
        self.CodesList = []
        df_metadata = pd.read_parquet(self.metadataFileName,engine='fastparquet')
        ListOfSpectrum = []
        
        # Load all codes at once
        total_spectra = df_metadata.shape[0]
        if verbose >= 1:
            print(f'Loading {total_spectra} spectra for decompression...')
        all_output_codes = np.zeros((total_spectra, self.quantizer, self.model.quantizedlen), dtype=np.int32)
        
        # Extract all codes from the bit unpacker
        for j in range(total_spectra):
            if verbose >= 1 and (j % 1000) == 0 and j > 0:
                print(f'Extracted codes for {j} spectra...')
            for k in range(self.quantizer):
                for l in range(self.model.quantizedlen):
                    try:
                        all_output_codes[j, k, l] = self.depacker.pull()
                    except:
                        traceback.print_exc()
                        print(f'Error extracting code for spectrum {j}, quantizer {k}, position {l}')
                        raise
            # Store compounded codes if needed
            if self.store_compounded:
                for j in self.compounded_idx_list:
                    self.compounded_original_codes_dict[str(j)] = all_output_codes[j, :].copy()
        
        # Process all codes in batches
        if verbose >= 1:
            print(f'Processing {total_spectra} spectra in batches of {self.batch_size}...')
        
        ListOfSpectrum = []
        for batch_start in range(0, total_spectra, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_spectra)
            current_batch_size = batch_end - batch_start
            
            if verbose >= 1:
                print(f'Processing batch {batch_start//self.batch_size + 1}, spectra {batch_start}-{batch_end-1}')
            
            # Extract batch codes
            batch_codes = all_output_codes[batch_start:batch_end]
            
            # Transpose for batch processing: B N D -> N B D
            batch_codes = np.transpose(batch_codes, (1, 0, 2))
            
            # Get batch metadata
            metaDataBatch = df_metadata.iloc[batch_start:batch_end].copy()
            
            # Decode batch
            if 'ExtendedMZ' in metaDataBatch.columns:
                bp = batchprocessor.SpectrumBatchDecoder(batch_codes, self.model, 
                                                       MaxValList=metaDataBatch['MaxVal'].values,
                                                       leftovermzList=metaDataBatch['ExtendedMZ'].to_list(), 
                                                       leftoverintensityList=metaDataBatch['ExtendedIT'].to_list())
            else:
                bp = batchprocessor.SpectrumBatchDecoder(batch_codes, self.model, 
                                                       MaxValList=metaDataBatch['MaxVal'].values)
            
            output_mz, output_intensity = bp.getReconstructSpectrum()
            
            # Prepare batch metadata
            metaDataBatch.pop('MaxVal')
            if 'ExtendedMZ' in metaDataBatch.columns:
                metaDataBatch.pop('ExtendedMZ')
                metaDataBatch.pop('ExtendedIT')
            
            # Convert batch to MGF format
            
            for k in range(current_batch_size):
                if verbose >= 2 and (batch_start + k) % 1000 == 0 and (batch_start + k) > 0:
                    print(f'Processed {batch_start + k} spectra...')
                
                # Round m/z values to 2 decimal places
                spectrum = {'m/z array': postprocessing.changeTofloat(list(output_mz[k])), 
                           'intensity array': np.array(list(output_intensity[k]), np.float64)}
                spectrum.update({'params': {key: metaDataBatch[key].values[k] 
                                          for key in metaDataBatch.keys()}})
                spectrum['params'].update({'pepmass': (spectrum['params']['pepmz'], 
                                                      spectrum['params']['pepit'])})
                
                spectrum['params'].pop('pepmz')
                spectrum['params'].pop('pepit')
                ListOfSpectrum.append(spectrum)
        
        # Process chimeric spectra if they exist
        if self.store_compounded and len(self.compounded_idx_list) > 0:
            if verbose >= 1:
                print('Processing compounded spectra')
            
            compounded_indices = np.array(self.compounded_idx_list)
            compounded_codes = np.zeros((compounded_indices.shape[0], self.quantizer, 
                                       self.model.quantizedlen), dtype=np.int32)
            
            # Extract compounded codes
            for j in range(compounded_indices.shape[0]):
                for k in range(self.quantizer):
                    for l in range(self.model.quantizedlen):
                        compounded_codes[j, k, l] = self.depacker.pull()
            
            compounded_codes = np.transpose(compounded_codes, (1, 0, 2))
            compounded_max_value_list = df_metadata['MaxVal'].values[compounded_indices]
            compounded_df = df_metadata.iloc[compounded_indices]
            
            # Process compounded spectra in batches
            for j in range(0, compounded_codes.shape[0], self.batch_size):
                if verbose >= 1:
                    print(f'Processing compounded spectrum batch {j}')
                
                batched_compounded_codes = compounded_codes[j:j+self.batch_size]
                batched_compounded_max_value_list = compounded_max_value_list[j:j+self.batch_size]
                batched_original_codes_list = []
                
                for k in range(len(batched_compounded_max_value_list)):
                    batched_original_codes_list.append(
                        self.compounded_original_codes_dict[str(compounded_indices[j+k])])
                
                batch_original_codes_vector = np.transpose(
                    np.array(batched_original_codes_list, dtype=np.int32), (1, 0, 2))
                
                if 'ExtendedMZ' in df_metadata.columns:
                    bp = batchprocessor.SpectrumBatchDecoder(
                        batched_compounded_codes, self.model,
                        MaxValList=batched_compounded_max_value_list,
                        leftovermzList=compounded_df['ExtendedMZ'].to_list(),
                        leftoverintensityList=compounded_df['ExtendedIT'].to_list())
                else:
                    bp = batchprocessor.SpectrumBatchDecoder(
                        batched_compounded_codes, self.model,
                        MaxValList=batched_compounded_max_value_list)
                
                output_mz, output_intensity = bp.getReconstructSpectrum(batch_original_codes_vector)
                
                metaDataBatch = compounded_df.iloc[j:j+self.batch_size].reset_index(drop=True)
                metaDataBatch.pop('MaxVal')
                
                # Convert compounded spectra to MGF format
                for k in range(len(batched_compounded_max_value_list)):
                    spectrum = {'m/z array': postprocessing.changeTofloat(list(output_mz[k])), 
                               'intensity array': np.array(list(output_intensity[k]), np.float64)}
                    spectrum.update({'params': {key: metaDataBatch[key].values[k] 
                                              for key in metaDataBatch.keys()}})
                    spectrum['params'].update({'title': self.changeSpectrumName(
                        metaDataBatch['title'].values[k])})
                    spectrum['params'].update({'pepmass': (spectrum['params']['pepmz'], 
                                                          spectrum['params']['pepit'])})
                    spectrum['params'].pop('pepmz')
                    spectrum['params'].pop('pepit')
                    ListOfSpectrum.append(spectrum)
                
        # write all spectra to MGF file
        mgf.write(ListOfSpectrum, self.OutputFileMGF)
        
        print(f'Decompression completed. Output written to {self.OutputFileMGF}')
        


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
                leftovermzList=metaDataBatch['ExtendedMZ'].to_list(),leftoverintensityList=metaDataBatch['ExtendedIT'].to_list())
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
                    leftovermzList=compounded_df['ExtendedMZ'].to_list(),leftoverintensityList=compounded_df['ExtendedIT'].to_list())
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
