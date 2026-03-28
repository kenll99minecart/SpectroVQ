# SpectroVQ
SpectroVQ is an autoencoder framework to compress and denoise DDA proteomics mass spectrum data without the need for peptide sequences.

Neural Proteomics Mass Spectrum Compression

![](https://github.com/kenll99minecart/SpectroVQ/blob/main/RVQ.png)

## Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (recommended for training and inference)
- Conda environment manager

### Step 1: Create Conda Environment
```bash
conda create -n spectrovq python=3.8
conda activate spectrovq
```

### Step 2: Install PyTorch
**IMPORTANT**: Install PyTorch based on your CUDA version first. Check your CUDA version with `nvidia-smi`.

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training (train.py)
Train a new SpectroVQ model on consensus library data.

```bash
python train.py --data_path library.feather --output_dir ./models --batch_size 32 --learning_rate 1e-4 --max_epochs 50
```

**Arguments:**
- `--data_path`: Path to consensus library file (.feather format) [default: library.feather]
- `--output_dir`: Output directory for trained model checkpoints [default: .]
- `--batch_size`: Batch size for training [default: 32]
- `--num_workers`: Number of workers for data loading [default: 4]
- `--learning_rate`: Learning rate for training [default: 1e-4]
- `--max_epochs`: Maximum number of training epochs [default: 50]

### 2. Denoising (denoise.py)
Denoise MGF spectra using a trained SpectroVQ model.

```bash
python denoise.py --input_mgf input.mgf --model_path model.ckpt --output_mgf denoised.mgf --num_quantizers 3
```

**Arguments:**
- `--input_mgf`: Path to input MGF file [required]
- `--model_path`: Path to trained model checkpoint (.ckpt or .pt file) [required]
- `--output_mgf`: Path to output denoised MGF file [required]
- `--yaml_config`: Path to YAML configuration file for model parameters [optional]
- `--device`: Device to use (cuda or cpu) [default: auto-detect]
- `--num_quantizers`: Number of quantizers to use for denoising [default: 3]
- `--retain_original_peaks`: Retain original peaks outside the m/z range [default: True]
- `--no_retain_original_peaks`: Do not retain original peaks outside the m/z range
- `--one_percent_threshold`: Apply 1% intensity threshold [default: True]
- `--no_one_percent_threshold`: Do not apply 1% intensity threshold
- `--compounded_spectra`: Enable chimeric spectrum reconstruction [default: True]
- `--no_compounded_spectra`: Disable chimeric spectrum reconstruction

### 3. Compression/Decompression (main_cmd_line.py)
Compress or decompress MGF files using SpectroVQ.

#### Compression:
```bash
python main_cmd_line.py --compress --input input.mgf --output output.vqms2 --weights model.ckpt --compression_method gzip --compression_level 6 --batch_size 32 --quantizer 4
```

#### Decompression:
```bash
python main_cmd_line.py --decompress --input input.vqms2 --output output.mgf --weights model.ckpt --compression_method gzip --compression_level 6 --batch_size 32 --quantizer 4
```

**Arguments:**
- `--compress, -m`: Enable compression mode
- `--decompress, -d`: Enable decompression mode
- `--input, -i`: Input file path [required]
- `--output, -o`: Output file path [required]
- `--weights, -w`: Path to model weights [required]
- `--compression_method, -cM`: Compression method [gzip, zlib, zstd] [required]
- `--compression_level, -cL`: Compression level [required]
- `--batch_size, -b`: Batch size for processing [optional]
- `--quantizer, -q`: Quantizer level for compression [default: 4]
- `--store_compounded, -sC`: Store compounded spectra [flag]

## File Formats
- **Input**: MGF files for denoising and compression
- **Compressed Output**: .vqms2 files (SpectroVQ compressed format)
- **Model Checkpoints**: .ckpt or .pt files
- **Training Data**: .feather files (should contain the columns: 'PrecursorMZ', 'CompoundName', 'Comment', 'm/z', 'intensity', 'Charge')

### Training Data Format

The training data for `train.py` should be a `.feather` file containing the following columns:
- `PrecursorMZ`: Precursor m/z value
- `CompoundName`: Peptide sequence with charge (e.g., 'NTDSIELALSYAK/2')
- `Comment`: Metadata and annotations for the spectrum
- `m/z`: Array of m/z values for peaks
- `intensity`: Array of intensity values for peaks
- `Charge`: Charge state of the precursor ion

**Example row:**
```python
# Pandas DataFrame row structure:
{
    'PrecursorMZ': '712.8670',
    'CompoundName': 'NTDSIELALSYAK/2',
    'Comment': 'AvePrecursorMz=713.2927 BestRawSpectrum=20101210_Velos1_AnWe_SA_U2OS_4.23852.23852 BinaryFileOffset=4069648408 CollisionEnergy=40.0 ConsFracAssignedPeaks=0.660 DotConsensus=0.83,0.07;1/6 FracUnassigned=0.65,3/5;0.47,7/20;0.52,272/380 Inst=1/orbitrap,6,8 MassDiff=0.0050 MassDiffCounts=1/0',
    'm/z': array([102.0552, 110.0712, 120.0808, 126.0552, 129.0708, 129.0975, 130.0864, 136.0759, 137.0793, 143.0818, 144.081, 147.113, 153.0659, 155.1011, 157.1261, 169.0975, 170.0927, 171.0768, 171.1128, 173.1288, 181.0608, 183.1291, 185.1234, 185.1578, 187.0716, 187.1367, 188.1032, 189.0879, 197.129, 198.0876, 199.0717, 200.1394, 201.1237, 203.0669, 211.1444, 213.087, 215.1395, 216.0982, 217.0822, 218.1505, 223.1073, 228.1343, 235.1068, 243.1342, 244.1654, 251.1033, 254.151, 260.1975, 270.0718, 272.1602, 286.1034, 298.1448, 304.1133, 312.1555, 313.1147, 314.1263, 314.1432, 316.1508, 330.1669, 331.1248, 355.1616, 356.2192, 367.2342, 371.1931, 373.1362, 381.2138, 383.1186, 383.1939, 400.1467, 401.1298, 417.2165, 418.1581, 427.183, 427.2567, 445.1924, 468.2466, 469.2494, 496.2781, 513.2329, 514.2884, 546.2512, 558.2807, 581.3308, 582.3352, 629.3163, 652.3674, 653.3726, 765.4535, 766.4645, 894.4956, 895.4965, 1007.5797, 1008.5803, 1094.6135, 1095.6171, 1209.6326, 1210.636]),
    'intensity': array([1251.9, 3081., 10000., 460.7, 1902.3, 4007.2, 1789.7, 4250.5, 413.5, 2695.4, 3648.6, 3577., 558.1, 522.3, 646.2, 104.1, 420.8, 3134.1, 174.1, 1020.8, 602.2, 960.8, 1101.9, 468.9, 714.2, 112.2, 5302.8, 465.7, 2731.5, 1436.8, 2044.1, 1062.9, 1553.4, 492.7, 579.8, 1168., 662.6, 6195.6, 675.1, 2593.2, 424.2, 3342.1, 349.8, 822.8, 383.4, 243.2, 614.5, 577.7, 126.7, 375.6, 678.7, 1053.9, 629.6, 129.1, 667.1, 612.9, 1127.5, 369.9, 328.6, 578., 437.2, 162.7, 120.6, 584.7, 558.3, 2099.5, 529.3, 451.1, 2640.1, 628.3, 138.7, 783.3, 170.2, 448.8, 345.4, 3475.9, 744.4, 900.8, 475., 609.6, 426., 485.8, 2657.4, 501.3, 499.5, 3170.6, 1354.8, 2683.4, 993.7, 4282.7, 2469.1, 1064.8, 807.5, 1357.3, 562.1, 1604., 1197.5]),
    'Charge': 2
}
```

## Notes
- GPU acceleration is highly recommended for training and large-scale processing
- The model automatically detects CUDA availability
- For training, ensure your consensus library contains proper peak annotations and metadata