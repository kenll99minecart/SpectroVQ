# %%
import argparse
import os
argparser = argparse.ArgumentParser(description='Compress or Decompress MGF files')
argparser.add_argument('mode', type=str, help='Mode: compress or decompress')
argparser.add_argument('input', type=str, help='Input file name')
argparser.add_argument('output', type=str, help='Output file name')

# %%
from utils.mgfHandler.compressor import MGFCompressor
#from .decompressor import MGFDecompressor
from utils.model import getModel
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
SpectraStream = getModel.getModel()
SpectraStream = getModel.ApplyWeights(SpectraStream,'/home/james/ResidualVector/SpectraStreamNewOptimizedMini6ReproducedWithSchedulerWithSortPrecursorMZDifferentAugmentation/LightningChk/SpectraStream-epoch=38-val_loss=0.61.ckpt')
SpectraStream.to('cuda')
SpectraStream.eval()
# %%
mgfcompressor = MGFCompressor(mgfFilePath='/data/data/PXD028735/LFQ_Orbitrap_DDA_Ecoli_01.mgf',OutputFileName='/data3/james/LFQ_Orbitrap_DDA_Ecoli_01',model=SpectraStream,batch_size=128,quantizer=6)
mgfcompressor.Compress(verbose = 1,debug = True)
# %%
from utils.mgfHandler import decompressor
import importlib
importlib.reload(decompressor)
mgfdecompressor = decompressor.MGFDecompressor(model=SpectraStream,inputFileName='/data3/james/LFQ_Orbitrap_DDA_Ecoli_01.vqms2',outputfilemgf='/data3/james/LFQ_Orbitrap_DDA_Ecoli_01_decompressed.mgf',batch_size=128,quantizer=6)
mgfdecompressor.Decompress()
# %%
