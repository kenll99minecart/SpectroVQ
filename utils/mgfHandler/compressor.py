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
    def __init__(self,model,mgfFilePath,OutputFileName = None,batch_size = None,quantizer = 6,
                 gzip_compression_level = None,zlib_compression_level = 6,zstd_compression_level = None,outputOutOfRange = True):
        """
        Initialize the MGFCompressor for compressing mass spectrometry data.

        Args:
            model: Neural network model for encoding spectra
            mgfFilePath (str): Path to the input MGF file containing mass spectra
            OutputFileName (str, optional): Name for output files. Defaults to input filename.
            batch_size (int, optional): Number of spectra to process in each batch. Defaults to 128.
            quantizer (int): Number of quantization levels for vector quantization. Defaults to 6.
            gzip_compression_level (int, optional): Compression level for gzip. Defaults to None.
            zlib_compression_level (int): Compression level for zlib. Defaults to 6.
            outputOutOfRange (bool): Whether to include peaks outside the range of 150-1500Th. Defaults to True.
        """
        self.model = model
        self.mgfFilePath = mgfFilePath
        self.OutputFileName = OutputFileName
        if self.OutputFileName is None:
            self.OutputFileName = os.path.basename(mgfFilePath).split('.')[0] 
            Warning(f"Output file name is not defined; using input name {self.OutputFileName} as output name")
        self.mgfFile = mgf.read(mgfFilePath)
        self.device = next(self.model.parameters()).device
        self.packer = compress.BitPacker(10, open(self.OutputFileName + '.vqms2', 'wb'), gzip_compression=gzip_compression_level, zlib_compression=zlib_compression_level, zstd_compression=zstd_compression_level)
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
        if 'charge' in df_meta.columns:
            df_meta['charge'] = df_meta['charge'].apply(lambda x: x[0])
        df_meta['MaxVal'] = self.MaxValList
        if self.outputOutOfRange:
            df_meta['ExtendedMZ'] = self.ExtendedMzList
            df_meta['ExtendedIT'] = self.ExtendedIntensityList
        df_meta.pop('pepmass')
        df_meta.to_parquet(self.metadataFileName,compression = 'gzip',compression_level = self.parquet_compression_level)

    # Mode 1: Load all spectra into MEMORY, then perform compression
    def LoadAll(self, debug: bool = False):
        """
        Load all spectra from the MGF file into memory.

        Reads all spectra from the MGF file and stores m/z arrays, intensity arrays,
        and parameters for subsequent compression. Initializes the MaxValList for tracking.
        """
        mzList, intensityList,paramList = [],[],[]
        for i,spectrum in enumerate(self.mgfFile):
            mzList.append(spectrum['m/z array'])
            intensityList.append(spectrum['intensity array'])
            paramList.append(spectrum['params'])
            if (debug) & (i >= 9):
                break
        self.mzList = mzList
        self.intensityList = intensityList
        self.paramList = paramList
        self.MaxValList = []

    def CompressAll(self,verbose = 0,num_workers = 32,debug = False):
        """
        Compress all loaded spectra using the neural network model.

        This method performs compression in memory after loading all spectra.
        It processes spectra in batches, extracts vector quantized codes, handles
        chimeric spectra if enabled, and saves compressed data to binary format.

        Args:
            verbose (int): Verbosity level for progress reporting. Defaults to 0.
            num_workers (int): Number of worker processes for parallel processing. Defaults to 32.
        """
        self.LoadAll(debug = debug)
        if verbose >=1:
            print(f'Loaded {len(self.mzList)} spectra from {self.mgfFilePath}')
            print(f'Starting Compression with batch size {self.batch_size} and quantizer {self.quantizer}')
        tb = torchprocessor.SpectrumTorchBatchEncoder(self.mzList,self.intensityList,self.model,compounded = True)
        output_codes = tb.getReconstructIndices(outputOutOfRange = self.outputOutOfRange,quantizer = self.quantizer,batch_size = self.batch_size, num_workers = num_workers)
        raw_codes, *extra = output_codes
        output_codes = raw_codes
        if len(extra) == 4:
            self.compounded_codes_list.extend(extra[0])
            self.compounded_idx_list.extend(extra[1])
            self.ExtendedMzList.extend(extra[2])
            self.ExtendedIntensityList.extend(extra[3])
        elif len(extra) == 2:
            self.compounded_codes_list.extend(extra[0])
            self.compounded_idx_list.extend(extra[1])
        elif len(extra) == 1:
            self.ExtendedMzList.extend(extra[0][0])
            self.ExtendedIntensityList.extend(extra[0][1])
        self.MaxValList.extend(tb.MaxValList)
        output_codes = np.transpose(np.array(output_codes.cpu().numpy(),dtype=np.int32),(1,0,2)) # N B D -> B N D
        # Push the first layer of quantized codes first, then push the 2nd layer of chimeric codes to the packer
        for j in range(output_codes.shape[0]): # Data Size
            for k in range(self.quantizer):
                for l in range(self.model.quantizedlen): # Number of integers in each quantizer
                    self.packer.push(output_codes[j,k,l]) # First in First out
            
        if len(self.compounded_codes_list) > 0:
            for j in range(len(self.compounded_codes_list)):
                for k in range(self.quantizer): # Number of quantizers
                    for l in range(self.model.quantizedlen): # Number of integers in each quantizer 
                        self.packer.push(self.compounded_codes_list[j][k,l])
        
        if verbose >=1:
            print(f'Flushing Packer')
        self.packer.flush()
        self.ProcessMetaData()

        if len(self.compounded_idx_list) > 0:
            if verbose >=1:
                print(f'Storing compounded data: {len(self.compounded_idx_list)} entries')
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
        bp = batchprocessor.SpectrumBatchEncoder(mz,intensity,self.model,compounded = True)
        output_codes = bp.getReconstructIndices(outputOutOfRange = self.outputOutOfRange,quantizer = self.quantizer)
        raw_codes, *extra = output_codes
        output_codes = raw_codes
        if len(extra) == 4:
            self.compounded_codes_list.extend(extra[0])
            ids = [int(id + idx * self.batch_size)  for id in extra[1]]
            self.compounded_idx_list.extend(ids)
            self.ExtendedMzList.extend(extra[2])
            self.ExtendedIntensityList.extend(extra[3])
        elif len(extra) == 2:
            self.compounded_codes_list.extend(extra[0])
            ids = [int(id + idx * self.batch_size) for id in extra[1]]
            self.compounded_idx_list.extend(ids)
        elif len(extra) == 1:
            self.ExtendedMzList.extend(extra[0][0])
            self.ExtendedIntensityList.extend(extra[0][1])
        self.MaxValList.extend(bp.MaxValList)
        output_codes = np.transpose(np.array(output_codes.cpu().numpy(),dtype=int),(1,0,2)) # N B D -> B N D
        for j in range(output_codes.shape[0]): # Batch Number
            for k in range(self.quantizer): # Number of quantizers
                for l in range(self.model.quantizedlen): # Number of integers in each quantizer
                    self.packer.push(output_codes[j,k,l]) # First in First out