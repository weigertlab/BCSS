#!/usr/bin/env python3
"""
Test script to download additional crops from a few slides
Usage: python test_additional_crops.py [--size 1024] [--type zarr]
"""

import sys
import os

# Add current directory to path
CWD = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CWD)

import configs as cf
from download_additional_crops import download_additional_crops, parse_arguments

# Test with just a single slide
cf.SLIDES_TO_KEEP = ['TCGA-A1-A0SK']
cf.CROPS_PER_SLIDE = 1  # Just 1 crop per slide for testing

if __name__ == '__main__':
    print("Testing additional crop download with limited slides...")
    print(f"Test slides: {cf.SLIDES_TO_KEEP}")
    print(f"Crops per slide: {cf.CROPS_PER_SLIDE}")
    
    # Parse arguments for crop size
    args = parse_arguments()
    
    if args.size:
        print(f"Using custom crop size: {args.size}x{args.size}")
    else:
        print(f"Using default crop size from config: {cf.ADDITIONAL_CROP_WIDTH}x{cf.ADDITIONAL_CROP_HEIGHT}")
    
    print(f"Output format: {args.type}")
    
    download_additional_crops(
        crop_size=args.size, 
        output_type=args.type, 
        compression=args.compression,
        nfiles=1  # Only download from 1 slide
    )