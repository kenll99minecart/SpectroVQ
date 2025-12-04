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
# Define path to directory containing MGF files to process
filepath = "/data3/james/PXD028735/paperNoised"

# Process all MGF files in the directory that don't contain 'chimeric' or 'Binned' in their names
for file in os.listdir(filepath):
    if file.endswith(".mgf") & ('chimeric' not in file) & ('Binned' not in file):
        print(f'Processing file: {file}')
        # Initialize MGF compressor for each file with specific parameters
        mgfcompressor = MGFCompressor(mgfFilePath=os.path.join(filepath, file),OutputFileName=os.path.join('/data3/james/compressedStorage/SpectroVQ', file.split('.')[0]),model=SpectraStream,batch_size=64,quantizer=6)
        # Compress all spectra in the file using all available CPU cores
        mgfcompressor.CompressAll(verbose=1,num_workers=32)
# %%