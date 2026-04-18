# %%
from utils.mgfHandler.compressor import MGFCompressor
from utils.mgfHandler.decompressor import MGFDecompressor
from utils.model import getModel
import os
import torch
import argparse
import os

# Set up command line argument parser for compression/decompression modes
argparser = argparse.ArgumentParser(description='Compress or Decompress MGF files')
argparser.add_argument('--compress','-m', action = 'store_true', help='Mode: Compress')
argparser.add_argument('--decompress','-d', action = 'store_true', help='Mode: Decompress')
argparser.add_argument('--input','-i', type=str, help='Input MGF filePath')
argparser.add_argument('--output','-o', type=str, help='Output file name')
argparser.add_argument('--weights','-w', type=str, help='Path to model weights')
argparser.add_argument('--compression_method','-cM',type = str, help = 'Compression/Decompress Method. Available Compression Method: [gzip,zlib,zstd]')
argparser.add_argument('--compression_level','-cL',type = str, help = 'Compression Level')
argparser.add_argument('--batch_size','-b',type = int, help = 'Batch size for compression')
argparser.add_argument('--quantizer','-q',type = int, help = 'Quantizer level for compression')
argparser.add_argument('--stored_raw','-sR', action = 'store_true', help = 'Store raw reconstructed spectra')
argparset.add_argument('--verbose','-v', action = 'store_true', help = 'Verbose output')
# %%
arguments = argparser.parse_args()
compress = arguments.compress
decompress = arguments.decompress
input_file_path = arguments.input
output_file_path = arguments.output
weights_path = arguments.weights
compression_method = arguments.compression_method
compression_level = arguments.compression_level
batch_size = arguments.batch_size
stored_raw = arguments.stored_raw
verbose = arguments.verbose

# Double check the input
if input_file_path is None:
    raise ValueError("Input file path is required")
if output_file_path is None:
    raise ValueError("Output file path is required")
if weights_path is None:
    raise ValueError("Weights path is required")
if compression_method is None:
    raise ValueError("Compression method is required")
if compression_level is None:
    raise ValueError("Compression level is required")


# Load 
torch.backends.cudnn.benchmark=True 
# Load the neural network model for spectrum processing
SpectraStream = getModel.getModel()
# Apply pre-trained weights to the model
SpectraStream = getModel.ApplyWeights(SpectraStream,weights_path)
# Move model to GPU and set to evaluation mode
SpectraStream.to('cuda')
SpectraStream.eval()

level = int(compression_level)
if compress + decompress > 1:
    raise ValueError("Mode must be either compress or decompress, not both")
elif compress:
    if not input_file_path.endswith(".mgf"):
        raise ValueError("Input file must be a MGF file")
    if compression_method == "gzip":
        mgfcompressor = MGFCompressor(mgfFilePath=input_file_path,OutputFileName=output_file_path,model=SpectraStream,batch_size=batch_size,quantizer=4
    ,gzip_compression_level = level, zlib_compression_level = None, zstd_compression_level = None)
    elif compression_method == "zlib":
        mgfcompressor = MGFCompressor(mgfFilePath=input_file_path,OutputFileName=output_file_path,model=SpectraStream,batch_size=batch_size,quantizer=4
    ,gzip_compression_level = None, zlib_compression_level = level, zstd_compression_level = None)
    elif compression_method == "zstd":
        mgfcompressor = MGFCompressor(mgfFilePath=input_file_path,OutputFileName=output_file_path,model=SpectraStream,batch_size=batch_size,quantizer=4
    ,gzip_compression_level = None, zlib_compression_level = None, zstd_compression_level = level)
    else:
        raise ValueError("Compression method must be either 'gzip', 'zlib', or 'zstd'")
    if verbose:
        print("Compressing...")
    verboseInt = 2 if verbose else 0
    mgfcompressor.CompressAll(verbose=verboseInt, debug=False)
elif decompress:
    if not input_file_path.endswith(".vqms2"):
        raise ValueError("Input file must be a .vqms2")
    if compression_method == "gzip":
        mgfcompressor = MGFDecompressor(mgfFilePath=input_file_path,OutputFileName=output_file_path,model=SpectraStream,batch_size=batch_size,quantizer=4,stored_raw = stored_raw
    ,gzip_compression_level = level, zlib_compression_level = None, zstd_compression_level = None)
    elif compression_method == "zlib":
        mgfcompressor = MGFDecompressor(mgfFilePath=input_file_path,OutputFileName=output_file_path,model=SpectraStream,batch_size=batch_size,quantizer=4,stored_raw = stored_raw
    ,gzip_compression_level = None, zlib_compression_level = level, zstd_compression_level = None)
    elif compression_method == "zstd":
        mgfcompressor = MGFDecompressor(mgfFilePath=input_file_path,OutputFileName=output_file_path,model=SpectraStream,batch_size=batch_size,quantizer=4,stored_raw = stored_raw
    ,gzip_compression_level = None, zlib_compression_level = None, zstd_compression_level = level)
    else:
        raise ValueError("Compression method must be either 'gzip', 'zlib', or 'zstd'")
    if verbose:
        print("Decompressing...")
    verboseInt = 2 if verbose else 0
    mgfcompressor.DecompressAll(verbose=verboseInt, debug=False)
# %%