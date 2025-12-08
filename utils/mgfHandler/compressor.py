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
    def __init__(self,model,mgfFilePath,OutputFileName = None,batch_size = None,quantizer = 6, store_compounded = True,
                 gzip_compression_level = None,zlib_compression_level = 6,outputOutOfRange = True):
        """
        Initialize the MGFCompressor for compressing mass spectrometry data.

        Args:
            model: Neural network model for encoding spectra
            mgfFilePath (str): Path to the input MGF file containing mass spectra
            OutputFileName (str, optional): Name for output files. Defaults to input filename.
            batch_size (int, optional): Number of spectra to process in each batch. Defaults to 128.
            quantizer (int): Number of quantization levels for vector quantization. Defaults to 6.
            store_compounded (bool): Whether to store spectra in a multiple quantizer format. Each spectra is stored as a 2 series of quantizer codes
            gzip_compression_level (int, optional): Compression level for gzip. Defaults to None.
            zlib_compression_level (int): Compression level for zlib. Defaults to 6.
            outputOutOfRange (bool): Whether to output out of range values. Defaults to True.
        """
        self.model = model
        self.mgfFilePath = mgfFilePath
        self.OutputFileName = OutputFileName
        if self.OutputFileName is None:
            self.OutputFileName = os.path.basename(mgfFilePath).split('.')[0] 
            Warning(f"Output file name is not defined; using input name {self.OutputFileName} as output name")
        self.mgfFile = mgf.read(mgfFilePath)
        self.device = next(self.model.parameters()).device
        self.packer = compress.BitPacker(10, open(self.OutputFileName + '.vqms2', 'wb'), gzip_compression=gzip_compression_level, zlib_compression=zlib_compression_level)
        self.gzip_compression_level = gzip_compression_level
        self.parquet_compression_level = zlib_compression_level if zlib_compression_level is not None else gzip_compression_level
        self.quantizer = quantizer
        self.metadataFileName = self.OutputFileName + '.parquet' 
        self.MaxValList = []
        if outputOutOfRange:
            self.ExtendedMzList,self.ExtendedIntensityList = [],[]
        self.outputOutOfRange = outputOutOfRange

        if batch_size is None:
            Warning('Batch size is not defined; using default batch size of 128')
            self.batch_size = 128
        else:
            self.batch_size = batch_size
        if store_compounded:
            self.store_compounded = True
            self.compounded_codes_list = []
            self.compounded_idx_list = []

    def ProcessMetaData(self):
        """
        Process and save metadata from compressed spectra to a parquet file.

        Extracts relevant metadata from the parameter list, processes pepmass and charge data,
        adds maximum values, and saves to a compressed parquet file.
        """
        df_meta = pd.DataFrame.from_dict(self.paramList)
        # df_meta['pepmass'] = df_meta['pepmass'].astype(str)
        df_meta['pepmz'] = df_meta['pepmass'].apply(lambda x: x[0])
        df_meta['pepit'] = df_meta['pepmass'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
        df_meta['charge'] = df_meta['charge'].apply(lambda x: x[0])
        df_meta['MaxVal'] = self.MaxValList
        df_meta.pop('pepmass')
        df_meta.to_parquet(self.metadataFileName,compression = 'gzip',compression_level = self.parquet_compression_level)

    # Mode 1: Load all spectra into MEMORY, then perform compression
    def LoadAll(self):
        """
        Load all spectra from the MGF file into memory.

        Reads all spectra from the MGF file and stores m/z arrays, intensity arrays,
        and parameters for subsequent compression. Initializes the MaxValList for tracking.
        """
        mzList, intensityList,paramList = [],[],[]
        for spectrum in self.mgfFile:
            mzList.append(spectrum['m/z array'])
            intensityList.append(spectrum['intensity array'])
            paramList.append(spectrum['params'])
        self.mzList = mzList
        self.intensityList = intensityList
        self.paramList = paramList
        self.MaxValList = []

    def CompressAll(self,verbose = 0,num_workers = 32):
        """
        Compress all loaded spectra using the neural network model.

        This method performs compression in memory after loading all spectra.
        It processes spectra in batches, extracts vector quantized codes, handles
        chimeric spectra if enabled, and saves compressed data to binary format.

        Args:
            verbose (int): Verbosity level for progress reporting. Defaults to 0.
            num_workers (int): Number of worker processes for parallel processing. Defaults to 32.
        """
        self.LoadAll()
        if verbose >=1:
            print(f'Loaded {len(self.mzList)} spectra from {self.mgfFilePath}')
            print(f'Starting Compression with batch size {self.batch_size} and quantizer {self.quantizer}')
        tb = torchprocessor.SpectrumTorchBatchEncoder(self.mzList,self.intensityList,self.model,chimeric = self.store_compounded)
        output_codes = tb.getReconstructIndices(outputOutOfRange = False,quantizer = self.quantizer,batch_size = self.batch_size, num_workers = num_workers)
        if self.store_compounded:
            raw_codes,chimeric_codes,chimericIdx = output_codes
            self.compounded_codes_list.extend(chimeric_codes)
            self.compounded_idx_list.extend(chimericIdx)
            output_codes = raw_codes
        self.MaxValList.extend(tb.MaxValList)
        output_codes = np.transpose(np.array(output_codes.cpu().numpy(),dtype=np.int32),(1,0,2)) # N B D -> B N D
        # Push the first layer of quantized codes first, then push the 2nd layer of chimeric codes to the packer
        for j in range(output_codes.shape[0]): # Data Size
            for k in range(self.quantizer):
                for l in range(self.model.quantizedlen): # Number of integers in each quantizer
                    self.packer.push(output_codes[j,k,l]) # First in First out
        if self.store_compounded:
            for j in range(len(self.compounded_codes_list)):
                for k in range(self.quantizer): # Number of quantizers
                    for l in range(self.model.quantizedlen): # Number of integers in each quantizer 
                        self.packer.push(self.compounded_codes_list[j][k,l])
        if verbose >=1:
            print(f'Flushing Packer')
        self.packer.flush()
        self.ProcessMetaData()
        if self.store_compounded:
            with open(self.OutputFileName + '_compounded.pkl', 'wb') as f:
                pickle.dump(self.compounded_idx_list, f)
    

    def ProcessSpectra(self,mz,intensity,idx = 0):
        """
        Process a batch of spectra for compression.

        Encodes the given m/z and intensity arrays using the neural network model,
        extracts vector quantized codes, handles chimeric spectra if enabled,
        and stores the compressed codes using the bit packer.

        Args:
            mz (list): List of m/z arrays for the batch of spectra
            intensity (list): List of intensity arrays for the batch of spectra
            idx (int): Starting index for tracking chimeric spectra. Defaults to 0.
        """
        bp = batchprocessor.SpectrumBatchEncoder(mz,intensity,self.model,compounded = self.store_compounded)
        output_codes = bp.getReconstructIndices(outputOutOfRange = self.outputOutOfRange,quantizer = self.quantizer)
        raw_codes, *extra = output_codes
        output_codes = raw_codes
        if len(extra) == 4:
            self.compounded_codes_list.extend(extra[0])
            self.compounded_idx_list.extend(list(np.array(extra[1],dtype=np.int32) + idx * self.batch_size))
            self.ExtendedMzList.extend(extra[2])
            self.ExtendedIntensityList.extend(extra[3])
        elif len(extra) == 2:
            self.compounded_codes_list.extend(extra[0])
            self.compounded_idx_list.extend(list(np.array(extra[1],dtype=np.int32) + idx * self.batch_size))
        elif len(extra) == 1:
            self.ExtendedMzList.extend(extra[0][0])
            self.ExtendedIntensityList.extend(extra[0][1])
        self.MaxValList.extend(bp.MaxValList)
        output_codes = np.transpose(np.array(output_codes.cpu().numpy(),dtype=np.int32),(1,0,2)) # N B D -> B N D
        for j in range(output_codes.shape[0]): # Batch Number
            for k in range(self.quantizer): # Number of quantizers
                for l in range(self.model.quantizedlen): # Number of integers in each quantizer
                    self.packer.push(output_codes[j,k,l]) # First in First out
    # Mode 2: Batch-wise compression
    def Compress(self,verbose = 0,debug = False,CombineMetaData = True):
        """
        Compress MGF file using batch-wise processing.

        This method processes spectra in batches to manage memory usage efficiently.
        It reads spectra from the MGF file, processes them in batches, saves intermediate
        metadata files, combines them at the end, and handles chimeric spectra if enabled.

        Args:
            verbose (int): Verbosity level for progress reporting. Defaults to 0.
            debug (bool): If True, stops after processing 3 spectra for debugging. Defaults to False.
            CombineMetaData (bool): Whether to combine all metadata files into one. Defaults to True.
        """
        mzList, intensityList,paramsList = [],[],[]
        metaDataFileList = []
        spectrumBatchIndex = 0
        for spectrumidx,spectrum in enumerate(self.mgfFile):
            mzList.append(spectrum['m/z array'])
            intensityList.append(spectrum['intensity array'])
            paramsList.append(spectrum['params'])
            if verbose >= 1:
                if (spectrumidx % 1) == 0:
                    print(f'Processing Spectrum {spectrumidx}, Title: {spectrum["params"]["title"]}')
            if len(mzList) == self.batch_size:
                # This performs batch-wise compression so each iterations, the parquet metadata file is updated.
                self.ProcessSpectra(mzList,intensityList,spectrumBatchIndex)
                df_meta = pd.DataFrame.from_dict(paramsList)
                # df_meta['pepmass'] = df_meta['pepmass'].astype(str)
                df_meta['pepmz'] = df_meta['pepmass'].apply(lambda x: x[0])
                df_meta['pepit'] = df_meta['pepmass'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
                df_meta['charge'] = df_meta['charge'].apply(lambda x: x[0])
                df_meta['MaxVal'] = self.MaxValList
                if self.outputOutOfRange:
                    df_meta['ExtendedMZ'] = self.ExtendedMzList
                    df_meta['ExtendedIT'] = self.ExtendedIntensityList
                df_meta.pop('pepmass')
                metaDataFileName = self.OutputFileName + f'_part{(spectrumidx // self.batch_size)}.parquet'
                df_meta.to_parquet(metaDataFileName,compression = 'gzip',compression_level = self.parquet_compression_level)
                metaDataFileList.append(metaDataFileName)
                # if os.path.exists(self.metadataFileName) & ((spectrumidx // self.batch_size) > 0):
                #     existing_df = pd.read_parquet(self.metadataFileName)
                #     combined_df = pd.concat([existing_df, df_meta], ignore_index=True)
                #     combined_df.to_parquet(self.metadataFileName,compression = 'gzip', compression_level = self.parquet_compression_level)
                # else:
                #     df_meta.to_parquet(self.metadataFileName,compression = 'gzip',compression_level = self.parquet_compression_level)
                self.MaxValList.clear()
                spectrumBatchIndex +=1
            if debug & (spectrumidx >= 3):
                break
        # Process Last Batch
        if len(mzList) > 0:
            if verbose >=1:
                print(f'Processing Last Batch of size {len(mzList)}')
            self.ProcessSpectra(mzList,intensityList,spectrumBatchIndex)
            df_meta = pd.DataFrame.from_dict(paramsList)
            # df_meta['pepmass'] = df_meta['pepmass'].astype(str)
            df_meta['pepmz'] = df_meta['pepmass'].apply(lambda x: x[0])
            df_meta['pepit'] = df_meta['pepmass'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
            df_meta['charge'] = df_meta['charge'].apply(lambda x: x[0])
            df_meta['MaxVal'] = self.MaxValList
            if self.outputOutOfRange:
                df_meta['ExtendedMZ'] = self.ExtendedMzList
                df_meta['ExtendedIT'] = self.ExtendedIntensityList
            df_meta.pop('pepmass')
            metaDataFileName = self.OutputFileName + f'_part_compounded.parquet'
            df_meta.to_parquet(metaDataFileName,compression = 'gzip',compression_level = self.parquet_compression_level)
            metaDataFileList.append(metaDataFileName)
            # if os.path.exists(self.metadataFileName) & ((spectrumidx // self.batch_size) > 0):
            #     existing_df = pd.read_parquet(self.metadataFileName)
            #     combined_df = pd.concat([existing_df, df_meta], ignore_index=True)
            #     combined_df.to_parquet(self.metadataFileName,compression = 'gzip', compression_level = self.parquet_compression_level)
            # else:
            #     df_meta.to_parquet(self.metadataFileName,compression = 'gzip',compression_level = self.parquet_compression_level)
            self.MaxValList.clear()
        if CombineMetaData:
            if verbose >=1:
                print(f'Combining MetaData from {len(metaDataFileList)} files')
            combined_df = pd.concat([pd.read_parquet(f) for f in metaDataFileList], ignore_index=True)
            combined_df.to_parquet(self.metadataFileName,compression = 'gzip', compression_level = self.parquet_compression_level)
            for f in metaDataFileList:
                os.remove(f)
        if self.store_compounded:
            for j in range(len(self.compounded_codes_list)):
                for k in range(self.quantizer): # Number of quantizers
                    for l in range(self.model.quantizedlen): # Number of integers in each quantizer
                        self.packer.push(self.compounded_codes_list[j][k,l])
        if verbose >=1:
            print(f'Flushing Packer')
        self.packer.flush()
        
        if self.store_compounded:
            if verbose >=1:
                print(f'Processing Compounded MetaData')
            # stored_dict = {
            #     'chimeric_codes':self.chimeric_codes_list,
            #     'chimeric_idx':self.chimeric_idx_list
            # }
            # pd.DataFrame.from_dict(stored_dict).to_parquet(self.OutputFileName + 'chimeric.parquet', engine='fastparquet')
            with open(self.OutputFileName + '_compounded.pkl', 'wb') as f:
                pickle.dump(self.compounded_idx_list, f)