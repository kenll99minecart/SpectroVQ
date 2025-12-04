# %%
import argparse
import os

# Set up command line argument parser for compression/decompression modes
argparser = argparse.ArgumentParser(description='Compress or Decompress MGF files')
argparser.add_argument('mode', type=str, help='Mode: compress or decompress')
argparser.add_argument('input', type=str, help='Input file name')
argparser.add_argument('output', type=str, help='Output file name')

# %%
from utils.mgfHandler.compressor import MGFCompressor
# Import the decompressor module (currently commented out)
#from .decompressor import MGFDecompressor
from utils.model import getModel
import os

# Set CUDA device for GPU acceleration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load the neural network model for spectrum processing
SpectraStream = getModel.getModel()
# Apply pre-trained weights to the model
SpectraStream = getModel.ApplyWeights(SpectraStream,'/home/james/ResidualVector/SpectraStreamNewOptimizedMini6ReproducedWithSchedulerWithSortPrecursorMZDifferentAugmentation/LightningChk/SpectraStream-epoch=38-val_loss=0.61.ckpt')
# Move model to GPU and set to evaluation mode
SpectraStream.to('cuda')
SpectraStream.eval()
# %%
# Initialize MGF compressor with specific parameters for spectrum compression
mgfcompressor = MGFCompressor(mgfFilePath='/data/data/PXD028735/LFQ_Orbitrap_DDA_Ecoli_01.mgf',OutputFileName='/data3/james/LFQ_Orbitrap_DDA_Ecoli_01',model=SpectraStream,batch_size=128,quantizer=6)
# Perform compression with verbose output and debug mode enabled
mgfcompressor.Compress(verbose = 1,debug = True)

# %%
# Import and reload the decompressor module for spectrum decompression
from utils.mgfHandler import decompressor
import importlib
importlib.reload(decompressor)

# Initialize MGF decompressor to reconstruct spectra from compressed format
mgfdecompressor = decompressor.MGFDecompressor(model=SpectraStream,inputFileName='/data3/james/LFQ_Orbitrap_DDA_Ecoli_01.vqms2',outputfilemgf='/data3/james/LFQ_Orbitrap_DDA_Ecoli_01_decompressed.mgf',batch_size=128,quantizer=6)
# Perform decompression to reconstruct original MGF file
mgfdecompressor.Decompress()
# %%
