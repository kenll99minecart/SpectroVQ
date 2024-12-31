# %%
import argparse
import os
argparser = argparse.ArgumentParser(description='Compress or Decompress MGF files')
argparser.add_argument('mode', type=str, help='Mode: compress or decompress')
argparser.add_argument('input', type=str, help='Input file name')
argparser.add_argument('output', type=str, help='Output file name')


# %%
from compressor import MGFCompressor
#from .decompressor import MGFDecompressor
from utils.model import getModel
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
SpectraStream = getModel.getModel('modelParams.json')
SpectraStream = getModel.ApplyWeights(SpectraStream,'/home/james/ResidualVector/SpectraStreamFineTunedEcoliDataWithDecoder/LightningChk/SpectraStream241-epoch=14-val_loss=0.09.ckpt')
# %%
SpectraStream.eval()
MGFCompressor(mgfFilePath='/data/data/PXD028735/LFQ_Orbitrap_DDA_Ecoli_01.mgf',OutputFileName='/data2/temp1/LFQ_Orbitrap_DDA_Ecoli_01',model=SpectraStream,batch_size=128,quantizer=6).Compress()
# %%
