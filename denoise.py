#!/usr/bin/env python3
"""
Denoise spectra using MGFDenoiser from SpectroVQ.

This script loads a trained SpectroVQ model and uses it to denoise MGF files.
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add utils to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from utils.mgfHandler.denoiser import MGFDenoiser
from utils.model.getModel import getModel, ApplyWeights


def load_model(model_path, device='cuda', yaml_config=None):
    """
    Load a trained SpectroVQ model from checkpoint.
    
    Args:
        model_path (str): Path to the model checkpoint file
        device (str): Device to load the model on ('cuda' or 'cpu')
        yaml_config (str): Optional path to YAML configuration file
    
    Returns:
        SpectroVQ: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model using getModel function
    model = getModel(yaml_config)
    
    # Load weights using ApplyWeights function
    model = ApplyWeights(model, model_path)
    
    model.to(device)
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Denoise MGF spectra using SpectroVQ')
    
    # Required arguments
    parser.add_argument('--input_mgf', type=str, required=True,
                        help='Path to input MGF file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.ckpt or .pt file)')
    parser.add_argument('--output_mgf', type=str, required=True,
                        help='Path to output denoised MGF file')
    
    # Optional arguments
    parser.add_argument('--yaml_config', type=str, default=None,
                        help='Path to YAML configuration file for model parameters')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num_quantizers', type=int, default=3,
                        help='Number of quantizers to use for denoising')
    parser.add_argument('--retain_original_peaks', action='store_true', default=True,
                        help='Retain original peaks outside the m/z range')
    parser.add_argument('--no_retain_original_peaks', dest='retain_original_peaks', action='store_false',
                        help='Do not retain original peaks outside the m/z range')
    parser.add_argument('--one_percent_threshold', action='store_true', default=True,
                        help='Apply 1% intensity threshold')
    parser.add_argument('--no_one_percent_threshold', dest='one_percent_threshold', action='store_false',
                        help='Do not apply 1% intensity threshold')
    parser.add_argument('--compounded_spectra', action='store_true', default=True,
                        help='Enable chimeric spectrum reconstruction')
    parser.add_argument('--no_compounded_spectra', dest='compounded_spectra', action='store_false',
                        help='Disable chimeric spectrum reconstruction')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_mgf):
        print(f"Error: Input MGF file not found: {args.input_mgf}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_mgf)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    try:
        # Load model
        print(f"Loading model from {args.model_path}...")
        if args.yaml_config:
            print(f"Using configuration from {args.yaml_config}")
        model = load_model(args.model_path, args.device, args.yaml_config)
        print(f"Model loaded successfully on {args.device}")
        
        # Create denoiser
        denoiser = MGFDenoiser(
            model=model,
            inputMGF=args.input_mgf,
            outputMGF=args.output_mgf
        )
        
        # Run denoising
        print(f"Starting denoising process...")
        print(f"Input: {args.input_mgf}")
        print(f"Output: {args.output_mgf}")
        print(f"Using {args.num_quantizers} quantizers")
        
        denoiser.denoiseMGF(
            numQuantSingle=args.num_quantizers,
            retainOriginalPeaks=args.retain_original_peaks,
            OnePercentThreshold=args.one_percent_threshold,
            compoundedSpectra=args.compounded_spectra
        )
        
        print("Denoising completed successfully!")
        
    except Exception as e:
        print(f"Error during denoising: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
