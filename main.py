# %%
import argparse
import os

# Set up command line argument parser for compression/decompression modes
# argparser = argparse.ArgumentParser(description='Compress or Decompress MGF files')
# argparser.add_argument('mode', type=str, help='Mode: compress or decompress')
# argparser.add_argument('input', type=str, help='Input file name')
# argparser.add_argument('output', type=str, help='Output file name')

# %%
from utils.mgfHandler.compressor import MGFCompressor
# Import the decompressor module (currently commented out)
#from .decompressor import MGFDecompressor
from utils.model import getModel
import os
import torch

# Set CUDA device for GPU acceleration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark=True 
# Load the neural network model for spectrum processing
SpectraStream = getModel.getModel()
# Apply pre-trained weights to the model
SpectraStream = getModel.ApplyWeights(SpectraStream,'/home/james/ResidualVector/SpectraStreamNewOptimizedMini6ReproducedWithSchedulerWithSortPrecursorMZSQRTNative/LightningChk/SpectraStream-epoch=32-val_loss=0.59.ckpt')
# Move model to GPU and set to evaluation mode
SpectraStream.to('cuda')
SpectraStream.eval()
# %%
import time
# Initialize MGF compressor with specific parameters for spectrum compression
mgfcompressor = MGFCompressor(mgfFilePath='/data/data/PXD028735/LFQ_Orbitrap_DDA_Ecoli_02.mgf',OutputFileName='/data3/james/LFQ_Orbitrap_DDA_Ecoli_02',model=SpectraStream,batch_size=1,quantizer=4,store_compounded = True
,gzip_compression_level = None, zlib_compression_level = None, zstd_compression_level = None)
# Perform compression with verbose output and debug mode enabled
start = time.time()
mgfcompressor.CompressAll(verbose = 1,debug = True)
end = time.time()
print(f"Compression time: {end - start} seconds")
# %%
# Import and reload the decompressor module for spectrum decompression
from utils.mgfHandler import decompressor
import importlib
importlib.reload(decompressor)

# Initialize MGF decompressor to reconstruct spectra from compressed format
mgfdecompressor = decompressor.MGFDecompressor(model=SpectraStream,inputFileName='/data3/james/LFQ_Orbitrap_DDA_Ecoli_02.vqms2',outputfilemgf='/data3/james/LFQ_Orbitrap_DDA_Ecoli_02_decompressed.mgf',batch_size=2,quantizer=4,store_compounded = False
,gzip_compression_level = None, zlib_compression_level = None,zstd_compression_level = None)

# Perform decompression to reconstruct original MGF file
mgfdecompressor.DecompressAll()
# %%
from pyteomics import mgf
from matplotlib import pyplot as plt
numSpectra = 0
for idx, spectra in enumerate(mgf.read('/data3/james/LFQ_Orbitrap_DDA_Ecoli_02_decompressed.mgf')):
    if idx == 3:
        # print(spectra)
        plt.stem(spectra['m/z array'],spectra['intensity array'])
        print(spectra['m/z array'])
        print(spectra['intensity array'])
        numSpectra +=1
        break
plt.show()
print(numSpectra)
# %%
for idx, spectra in  enumerate(mgf.read('/data/data/PXD028735/LFQ_Orbitrap_DDA_Ecoli_02.mgf')):
    if idx == 3:
        plt.stem(spectra['m/z array'],spectra['intensity array'])
        print(spectra['m/z array'])
        print(spectra['intensity array'])
        plt.show()
        break
# %%
# import pandas as pd
# a = pd.read_parquet('/data3/james/LFQ_Orbitrap_DDA_Ecoli_02.parquet')
# %%
import os 
import time
filepath = '/data/data/PXD028735/'
for file in os.listdir(filepath):
    if ('Orbitrap' in file) & (file.endswith('mgf')):
        fullpath = os.path.join(filepath,file)
        print(f'Compressing {file}')
        mgfcompressor = MGFCompressor(mgfFilePath=fullpath,OutputFileName=f'/data3/james/resultFiles/gzip_{file}',model=SpectraStream,batch_size=256,quantizer=4,store_compounded = True,
        gzip_compression_level = 4, zlib_compression_level = None)
        # Perform compression with verbose output and debug mode enabled
        start = time.time()
        mgfcompressor.CompressAll(verbose = 0,debug = False)
        end = time.time()
        print(f"Compression time: {end - start} seconds")
        mgfcompressor = MGFCompressor(mgfFilePath=fullpath,OutputFileName=f'/data3/james/resultFiles/{file}',model=SpectraStream,batch_size=256,quantizer=4,store_compounded = True,
        )
        # Perform compression with verbose output and debug mode enabled
        start = time.time()
        mgfcompressor.CompressAll(verbose = 0,debug = False)
        end = time.time()
        print(f"Compression time: {end - start} seconds")
# %%
# Import and reload the decompressor module for spectrum decompression
from utils.mgfHandler import decompressor
import importlib
importlib.reload(decompressor)

# Initialize MGF decompressor to reconstruct spectra from compressed format
mgfdecompressor = decompressor.MGFDecompressor(model=SpectraStream,inputFileName='/data3/james/resultFiles/gzip_LFQ_Orbitrap_DDA_Ecoli_01.mgf.vqms2',outputfilemgf='/data3/james/LFQ_Orbitrap_DDA_Ecoli_01_decompressed.mgf',batch_size=256,quantizer=4,store_compounded = True,
gzip_compression_level = 4, zlib_compression_level = None)
# Perform decompression to reconstruct original MGF file
mgfdecompressor.DecompressAll(verbose = 0)
# %%