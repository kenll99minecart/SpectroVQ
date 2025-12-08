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
# Initialize MGF compressor with specific parameters for spectrum compression
mgfcompressor = MGFCompressor(mgfFilePath='/data/data/PXD028735/LFQ_Orbitrap_DDA_Ecoli_01.mgf',OutputFileName='/data3/james/LFQ_Orbitrap_DDA_Ecoli_01',model=SpectraStream,batch_size=128,quantizer=3,store_compounded = True)
# Perform compression with verbose output and debug mode enabled
mgfcompressor.Compress(verbose = 1,debug = True)
# %%
# Import and reload the decompressor module for spectrum decompression
from utils.mgfHandler import decompressor
import importlib
importlib.reload(decompressor)

# Initialize MGF decompressor to reconstruct spectra from compressed format
mgfdecompressor = decompressor.MGFDecompressor(model=SpectraStream,inputFileName='/data3/james/LFQ_Orbitrap_DDA_Ecoli_01.vqms2',outputfilemgf='/data3/james/LFQ_Orbitrap_DDA_Ecoli_01_decompressed.mgf',batch_size=128,quantizer=3,store_compounded = True)
# Perform decompression to reconstruct original MGF file
mgfdecompressor.Decompress(verbose = 1)
# %%
from pyteomics import mgf
from matplotlib import pyplot as plt
numSpectra = 0
for idx, spectra in enumerate(mgf.read('/data3/james/LFQ_Orbitrap_DDA_Ecoli_01_decompressed.mgf')):
    if idx == 4:
        # print(spectra)
        plt.stem(spectra['m/z array'],spectra['intensity array'])
        print(spectra['m/z array'])
        print(spectra['intensity array'])
        numSpectra +=1
        break
plt.show()
print(numSpectra)
# %%
for spectra in mgf.read('/data/data/PXD028735/LFQ_Orbitrap_DDA_Ecoli_01.mgf'):
    plt.stem(spectra['m/z array'],spectra['intensity array'])
    print(spectra['m/z array'])
    print(spectra['intensity array'])
    plt.show()
    break
# %%
# import torch
# m = torch.tensor([[[  65,  574,  111,  837,  815,  207,  862,  396,  309,   34,  655,   52,
#           378,  165,  131,  430,  875,  655,  655,   52,  227,  655,  655,  584,
#           314,  201,  301]],
#         [[ 825,  336,   90,  471,  345,  523,  506,  319,  375,  566,   22,  566,
#           928,  628, 1003,   98,  322,  164,  896,  566,  303,  107,  319,  727,
#           230,  950,  290]],
#         [[ 695,  313,  289,  440,  957,  882,  695,  552,  726,  613,  153,  302,
#           153,  577,  577,  619,  200,  613,  451,  639,  552,  451,  597,  349,
#           975,  542,  407]]])
# # %%
# torch.save(SpectraStream.reconstructIndices(m.to('cuda')),'/home/james/SpectroVQ/test2.tensor')

# %%
