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
from typing import OrderedDict

DEFAULTMGFPARAMSKEYS = ['title','rtinseconds','pepmass','charge','scans','seq','modifications']
class MGFDecompressor():
    def __init__(self,model,inputFileName,outputfilemgf = None, batch_size = None,quantizer = 6, stored_raw = False,
                 gzip_compression_level = None,zlib_compression_level = 6, zstd_compression_level = None):
        """
        Initialize the MGFDecompressor for decompressing mass spectrometry data.

        Args:
            model: Neural network model for decoding spectra
            inputFileName (str): Path to the compressed input file (.vqms2 format)
            outputfilemgf (str, optional): Path for output MGF file. Defaults to 'output.mgf'.
            batch_size (int, optional): Number of spectra to process in each batch. Defaults to 128.
            quantizer (int): Number of quantization levels used during compression. Defaults to 6.
            stored_raw (bool): Whether to store raw reconstructed spectra in addition to compounded spectra. When False, only compounded spectra are stored if available.
            gzip_compression_level (int, optional): Compression level for gzip. Defaults to None.
            zlib_compression_level (int): Compression level for zlib. Defaults to 6.
        """
        self.model = model
        if outputfilemgf is None:
            folder_path = os.path.dirname(inputFileName)
            self.OutputFileMGF = os.path.join(folder_path, 'output.mgf')
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
        self.stored_raw = stored_raw
        # Always process compounded spectra if available for reconstruction
        self.process_compounded = True
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
        Modify spectrum name to indicate it's a raw spectrum.

        Args:
            name (str): Original spectrum name

        Returns:
            str: Modified spectrum name with '_raw' suffix added to the first part
        """
        name_chunks = name.split('.')
        name_chunks[0] += '_raw'
        name = '.'.join(name_chunks)
        return name

    def formatMGFformat(self, output_mz, output_intensity, metaDataBatchrow:pd.Series, modify_flag:bool=False):
        """
        Format spectrum for MGF output.
        
        """
        spectrum = {'m/z array': postprocessing.changeTofloat(list(output_mz)), 
                               'intensity array': np.array(list(output_intensity), np.float64)}
        spectrum.update({'params': {key: metaDataBatchrow[key]
                                              for key in metaDataBatchrow.index}})
        if modify_flag:
            spectrum['params'].update({'title': self.changeSpectrumName(metaDataBatchrow['title'])})
        else:
            spectrum['params'].update({'title': metaDataBatchrow['title']})
        spectrum['params'].update({'pepmass': (spectrum['params']['pepmz'], 
                                                          spectrum['params']['pepit'])})
        spectrum['params'].pop('pepmz')
        spectrum['params'].pop('pepit')
        return spectrum
    
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
        df_metadata = pd.read_parquet(self.metadataFileName).reset_index(drop = True)
        ListOfSpectrum = []
        CompoundedSpectrumDict = OrderedDict()
        if self.stored_raw:
            ListOfRawSpectrum = []
        # Load all original spectrum codes at once
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
        
        # Process all codes in batches
        if verbose >= 1:
            print(f'Starting to process {total_spectra} spectra in batches of {self.batch_size}...')
        
        # Process the denoised compounded spectrum first if possible
        if self.process_compounded and len(self.compounded_idx_list) > 0:
            if verbose >= 1:
                print('Processing whole spectra')
            
            compounded_indices = np.array(self.compounded_idx_list)
            compounded_codes = np.zeros((compounded_indices.shape[0], self.quantizer, 
                                       self.model.quantizedlen), dtype=np.int32)
            
            # Extract compounded codes
            for j in range(compounded_indices.shape[0]):
                for k in range(self.quantizer):
                    for l in range(self.model.quantizedlen):
                        compounded_codes[j, k, l] = self.depacker.pull()
            
            compounded_max_value_list = df_metadata['MaxVal'].values[compounded_indices]
            df_metadata['OriginalIndex'] = range(len(df_metadata))
            compounded_metadata_df = df_metadata.copy().iloc[compounded_indices].reset_index(drop = True)
            
            original_codes = all_output_codes[compounded_indices,:]

            if verbose >= 1:
                print(f"A total of {compounded_codes.shape[0]} spectra contains compounded data, processing them now...")

            # Process compounded spectra in batches
            for batch_start in range(0, compounded_codes.shape[0], self.batch_size):
                if verbose >= 1:
                    if batch_start % 500 == 0:
                        print(f'Processing compounded denoised spectrum batch {batch_start}')
                batch_end = min(batch_start + self.batch_size, total_spectra)
                current_batch_size = batch_end - batch_start

                batched_compounded_codes = compounded_codes[batch_start:batch_end]
                batched_compounded_max_value_list = compounded_max_value_list[batch_start:batch_end]
                batch_original_codes = original_codes[batch_start:batch_end]
                batched_compounded_metadata = compounded_metadata_df.iloc[batch_start:batch_end]

                # Transpose for batch processing: B N D -> N B D
                batch_original_codes = np.transpose(batch_original_codes, (1, 0, 2))
                batched_compounded_codes = np.transpose(batched_compounded_codes, (1, 0, 2))

                if 'ExtendedMZ' in df_metadata.columns:
                    bp = batchprocessor.SpectrumBatchDecoder(
                        batch_original_codes, self.model,
                        MaxValList=batched_compounded_max_value_list,
                        leftovermzList=batched_compounded_metadata['ExtendedMZ'].to_list(),
                        leftoverintensityList=batched_compounded_metadata['ExtendedIT'].to_list())
                else:
                    bp = batchprocessor.SpectrumBatchDecoder(
                        batch_original_codes, self.model,
                        MaxValList=batched_compounded_max_value_list)
                
                # Insert compouneded spectrum
                output_mz, output_intensity = bp.getReconstructSpectrum(addedIndices = batched_compounded_codes)
                
                batched_compounded_metadata.pop('MaxVal')
                if 'ExtendedMZ' in batched_compounded_metadata.columns:
                    batched_compounded_metadata.pop('ExtendedMZ')
                    batched_compounded_metadata.pop('ExtendedIT')
                

                # Convert compounded spectra to MGF format
                for k in range(len(batched_compounded_max_value_list)):
                    spectrum = self.formatMGFformat(output_mz[k], output_intensity[k], batched_compounded_metadata.iloc[k])
                    CompoundedSpectrumDict[str(batched_compounded_metadata.iloc[k]['OriginalIndex'])] = spectrum
                    CompoundedSpectrumDict[str(batched_compounded_metadata.iloc[k]['OriginalIndex'])]['params'].pop('OriginalIndex')
        # Process other spectrum normally
        for batch_start in range(0, total_spectra, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_spectra)
            current_batch_size = batch_end - batch_start
            
            if verbose >= 1:
                print(f'Processing other raw batch {batch_start//self.batch_size + 1}, spectra {batch_start}-{batch_end-1}')
            
            # Extract batch codes
            batch_codes = all_output_codes[batch_start:batch_end]

            # Transpose for batch processing: B N D -> N B D
            batch_codes = np.transpose(batch_codes, (1, 0, 2))
            
            # Get batch metadata
            metaDataBatch = df_metadata.iloc[batch_start:batch_end].reset_index(drop = True).copy()
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
            metaDataBatch.pop('OriginalIndex')

            # Convert batch to MGF format
            for k in range(current_batch_size):
                if verbose >= 2 and (batch_start + k) % 1000 == 0 and (batch_start + k) > 0:
                    print(f'Processed {batch_start + k} spectra...')
                
                if str(batch_start + k) in CompoundedSpectrumDict.keys():
                    if verbose >= 2:
                        print(f"Spectrum {batch_start + k} is compounded")
                    ListOfSpectrum.append(CompoundedSpectrumDict[str(batch_start + k)].copy())
                    if self.stored_raw:
                        spectrum = self.formatMGFformat(output_mz[k], output_intensity[k], metaDataBatch.iloc[k], modify_flag=True)
                        ListOfRawSpectrum.append(spectrum)
                    continue
                else:
                    # Round m/z values to 2 decimal places
                    spectrum = self.formatMGFformat(output_mz[k], output_intensity[k], metaDataBatch.iloc[k], modify_flag=False)
                    ListOfSpectrum.append(spectrum)
        
        if verbose >=1:
            print('Starting to write spectra to MGF file.')
            print(f"Number of spectra: {len(ListOfSpectrum)}")
        # if verbose >=2:
            # for spec in ListOfSpectrum:
            #     print(spec['params']['title'])
        # write all spectra to MGF file
        if self.stored_raw:
            TotalListSpectrum = ListOfSpectrum + ListOfRawSpectrum
            if verbose >= 1:
                print(f"Number of raw spectra: {len(TotalListSpectrum)}")
            mgf.write(TotalListSpectrum, self.OutputFileMGF)
        else:
            mgf.write(ListOfSpectrum, self.OutputFileMGF)
        
        print(f'Decompression completed. Output written to {self.OutputFileMGF}')

