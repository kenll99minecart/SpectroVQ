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
filepath = "/data3/james/PXD028735/paperNoised"
for file in os.listdir(filepath):
    if file.endswith(".mgf") & ('chimeric' not in file) & ('Binned' not in file):
        print(f'Processing file: {file}')
        mgfcompressor = MGFCompressor(mgfFilePath=os.path.join(filepath, file),OutputFileName=os.path.join('/data3/james/compressedStorage/SpectroVQ2quant', file.split('.')[0]),model=SpectraStream,batch_size=128,quantizer=2)#6
        mgfcompressor.CompressAll()
# %%