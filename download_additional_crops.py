#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import girder_client
import datetime
import csv
import numpy as np
import random
import argparse
from PIL import Image
try:
    import zarr
    import dask.array as da
    from skimage.transform import downscale_local_mean
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    print("Warning: zarr and/or dask not available. Install with: pip install zarr dask[array] scikit-image")

CWD = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CWD)

from utils import create_directory_structure, printNlog, get_image_from_htk_response
import configs as cf

# Configuration imported from configs.py

# =============================================================================

def download_multiscale_zarr_crop(gc, slide_id, center_x, center_y, crop_size, zarr_path, cf_config):
    """Download and save multiscale zarr with same shape but different context"""
    if not ZARR_AVAILABLE:
        raise ImportError("zarr, dask, and scikit-image are required for zarr output. Install with: pip install zarr dask[array] scikit-image")
    
    # Create zarr group
    zarr_group = zarr.open_group(zarr_path, mode='w')
    
    scales = [1, 2, 4]  # Scale factors for each level
    
    for level, scale in enumerate(scales):
        # Calculate region size for this level (larger area at lower resolution)
        region_size = crop_size * scale
        half_size = region_size // 2
        
        # Calculate bounds centered on the same coordinate
        xmin = center_x - half_size
        ymin = center_y - half_size
        xmax = center_x + half_size
        ymax = center_y + half_size
        
        # Build API request for this level
        getStr = f"/item/{slide_id}/tiles/region?left={xmin}&right={xmax}&top={ymin}&bottom={ymax}"
        
        # Add resolution specification
        if cf_config.MPP is not None:
            mm = 0.001 * cf_config.MPP
            getStr += f"&mm_x={mm:.4f}&mm_y={mm:.4f}"
        elif cf_config.MAG is not None:
            getStr += f"&magnification={cf_config.MAG:.2f}"
        
        # Force output size to be consistent across levels
        getStr += f"&width={crop_size}&height={crop_size}"
        
        try:
            # Download the crop for this level
            resp = gc.get(getStr, jsonResp=False)
            image = get_image_from_htk_response(resp)
            image_array = np.array(image)
            
            # Ensure consistent output size
            if image_array.shape[:2] != (crop_size, crop_size):
                from PIL import Image as PILImage
                image_pil = PILImage.fromarray(image_array)
                image_pil = image_pil.resize((crop_size, crop_size), PILImage.LANCZOS)
                image_array = np.array(image_pil)
            
            # Save to zarr
            zarr_group.create_dataset(str(level), data=image_array, chunks=(512, 512, 3), dtype=image_array.dtype)
            
        except Exception as e:
            printNlog(f"Error downloading level {level} (scale {scale}x): {e}", level='error')
            # Fill with zeros if download fails
            zarr_group.create_dataset(str(level), data=np.zeros((crop_size, crop_size, 3), dtype=np.uint8), 
                                    chunks=(512, 512, 3), dtype=np.uint8)
    
    # Add metadata for multiscale
    zarr_group.attrs['multiscales'] = [{
        'version': '0.4',
        'name': 'image',
        'axes': [
            {'name': 'y', 'type': 'space'},
            {'name': 'x', 'type': 'space'}, 
            {'name': 'c', 'type': 'channel'}
        ],
        'datasets': [
            {'path': '0', 'coordinateTransformations': [{'type': 'scale', 'scale': [1.0, 1.0, 1.0]}]},
            {'path': '1', 'coordinateTransformations': [{'type': 'scale', 'scale': [2.0, 2.0, 1.0]}]},
            {'path': '2', 'coordinateTransformations': [{'type': 'scale', 'scale': [4.0, 4.0, 1.0]}]}
        ]
    }]

def get_slide_dimensions(gc, slide_id):
    """Get slide dimensions from HistomicsTK API"""
    try:
        props = gc.get(f'item/{slide_id}/tiles')
        return props.get('sizeX', 0), props.get('sizeY', 0)
    except Exception as e:
        printNlog(f"Error getting dimensions for slide {slide_id}: {e}", level='error')
        return 0, 0

def generate_random_crop_coordinates(slide_width, slide_height, crop_width, crop_height, margin, n_crops):
    """Generate random crop coordinates ensuring they fit within slide bounds"""
    crops = []
    
    # Ensure we have enough space for crops
    max_x = slide_width - crop_width - margin
    max_y = slide_height - crop_height - margin
    
    if max_x <= margin or max_y <= margin:
        printNlog(f"Slide too small for crops: {slide_width}x{slide_height}", level='error')
        return crops
    
    for _ in range(n_crops):
        xmin = random.randint(margin, max_x)
        ymin = random.randint(margin, max_y)
        xmax = xmin + crop_width
        ymax = ymin + crop_height
        
        crops.append({
            'xmin': xmin,
            'ymin': ymin, 
            'xmax': xmax,
            'ymax': ymax
        })
    
    return crops

def download_additional_crops(crop_size=None, output_type='png'):
    """Download additional crops from TCGA slides without masks"""
    
    # Determine crop dimensions
    if crop_size is not None:
        # Square crops of specified size
        crop_width = crop_height = crop_size
        printNlog(f"Using command-line crop size: {crop_size}x{crop_size} pixels")
    else:
        # Use config defaults
        crop_width = cf.ADDITIONAL_CROP_WIDTH
        crop_height = cf.ADDITIONAL_CROP_HEIGHT
        printNlog(f"Using config crop size: {crop_width}x{crop_height} pixels")
    
    # Set random seed for reproducibility
    random.seed(cf.RANDOM_SEED)
    
    # Connect to API
    gc = girder_client.GirderClient(apiUrl=cf.APIURL)
    if cf.apiKey is not None:
        gc.authenticate(apiKey=cf.apiKey)
    else:
        gc.authenticate(interactive=True)
    
    # Validate output type
    if output_type == 'zarr' and not ZARR_AVAILABLE:
        raise ImportError("zarr output requires zarr, dask, and scikit-image. Install with: pip install zarr dask[array] scikit-image")
    
    # Create output directory
    dir_suffix = '_zarr' if output_type == 'zarr' else ''
    additional_crops_dir = os.path.join(cf.SAVEPATH, f'additional_crops{dir_suffix}')
    os.makedirs(additional_crops_dir, exist_ok=True)
    
    # Get all slides
    resp = gc.get("item?folderId=%s&limit=1000000" % cf.source_folder_id)
    slides = {s['name'][:12]: {'name': s['name'], '_id': s['_id']} for s in resp}
    
    # Filter slides if specified
    slide_list = list(slides.keys())
    if cf.SLIDES_TO_KEEP is not None:
        slide_list = [j for j in slide_list if j in cf.SLIDES_TO_KEEP]
    
    printNlog(f"Processing {len(slide_list)} slides for additional crops")
    
    # Track crop metadata
    crop_metadata = []
    
    for sldid, slide_name in enumerate(slide_list):
        printNlog(f"Processing slide {sldid + 1}/{len(slide_list)}: {slide_name}")
        
        # Get slide dimensions
        slide_id = slides[slide_name]['_id']
        slide_width, slide_height = get_slide_dimensions(gc, slide_id)
        
        if slide_width == 0 or slide_height == 0:
            printNlog(f"Skipping slide {slide_name} - could not get dimensions", level='error')
            continue
        
        printNlog(f"Slide dimensions: {slide_width} x {slide_height}")
        
        # Generate random crop coordinates
        crops = generate_random_crop_coordinates(
            slide_width, slide_height, 
            crop_width, crop_height, 
            cf.EDGE_MARGIN, cf.CROPS_PER_SLIDE
        )
        
        if not crops:
            printNlog(f"Skipping slide {slide_name} - no valid crop regions", level='error')
            continue
        
        # Download each crop
        for crop_idx, crop in enumerate(crops):
            printNlog(f"  Downloading crop {crop_idx + 1}/{len(crops)}")
            
            try:
                # Calculate crop center coordinates
                center_x = (crop['xmin'] + crop['xmax']) // 2
                center_y = (crop['ymin'] + crop['ymax']) // 2
                
                # Determine resolution string for filename
                if cf.MPP is not None:
                    append_str = f"MPP-{cf.MPP:.4f}"
                elif cf.MAG is not None:
                    append_str = f"MAG-{cf.MAG:.2f}"
                else:
                    append_str = "MAG-0"
                
                filename = f"{slide_name}_crop{crop_idx:02d}_cx{center_x}_cy{center_y}_{append_str}"
                
                if output_type == 'png':
                    # For PNG, download single crop as before
                    getStr = f"/item/{slide_id}/tiles/region?left={crop['xmin']}&right={crop['xmax']}&top={crop['ymin']}&bottom={crop['ymax']}"
                    
                    if cf.MPP is not None:
                        mm = 0.001 * cf.MPP
                        getStr += f"&mm_x={mm:.4f}&mm_y={mm:.4f}"
                    elif cf.MAG is not None:
                        getStr += f"&magnification={cf.MAG:.2f}"
                    
                    resp = gc.get(getStr, jsonResp=False)
                    rgb = get_image_from_htk_response(resp)
                    
                    filepath = os.path.join(additional_crops_dir, filename + ".png")
                    rgb.save(filepath)
                    
                elif output_type == 'zarr':
                    # For zarr, use multiscale download approach
                    filepath = os.path.join(additional_crops_dir, filename + ".zarr")
                    download_multiscale_zarr_crop(gc, slide_id, center_x, center_y, crop_size if crop_size else crop_width, filepath, cf)
                
                # Store metadata
                crop_metadata.append({
                    'slide_name': slide_name,
                    'crop_index': crop_idx,
                    'center_x': center_x,
                    'center_y': center_y,
                    'xmin': crop['xmin'],
                    'ymin': crop['ymin'],
                    'xmax': crop['xmax'],
                    'ymax': crop['ymax'],
                    'filename': filename + ('.png' if output_type == 'png' else '.zarr'),
                    'output_type': output_type,
                    'resolution': append_str
                })
                
            except Exception as e:
                printNlog(f"Error downloading crop {crop_idx} from {slide_name}: {e}", level='error')
                continue
    
    # Save metadata CSV
    metadata_path = os.path.join(additional_crops_dir, 'crop_metadata.csv')
    with open(metadata_path, 'w', newline='') as csvfile:
        fieldnames = ['slide_name', 'crop_index', 'center_x', 'center_y', 'xmin', 'ymin', 'xmax', 'ymax', 'filename', 'output_type', 'resolution']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(crop_metadata)
    
    printNlog(f"Downloaded {len(crop_metadata)} additional crops")
    printNlog(f"Metadata saved to {metadata_path}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Download additional crops from TCGA slides without masks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_additional_crops.py --size 1024 --type png
  python download_additional_crops.py --size 2048 --crops-per-slide 5 --type zarr
  python download_additional_crops.py --type zarr  # Use config defaults, zarr output
  python download_additional_crops.py  # Use config defaults, PNG output
        """
    )
    
    parser.add_argument(
        '--size', 
        type=int, 
        default=2048,
        help='Square crop size in pixels (e.g., --size 1024 for 1024x1024 crops). '
             'If not specified, uses ADDITIONAL_CROP_WIDTH/HEIGHT from configs.py'
    )
    
    parser.add_argument(
        '--crops-per-slide',
        type=int,
        default=10,
        help='Number of crops to sample per slide. If not specified, uses CROPS_PER_SLIDE from configs.py'
    )
    
    parser.add_argument(
        '--type',
        choices=['png', 'zarr'],
        default='zarr',
        help='Output format: "png" for PNG images or "zarr" for multiscale zarr3 format with 1x/2x/4x downsampling'
    )

    parser.add_argument(
        '-o', '--outdir',
        type=str,
        default=None,
        help='Root output directory to write results and logs (overrides configs.SAVEPATH)'
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Override config if specified
    if args.crops_per_slide is not None:
        cf.CROPS_PER_SLIDE = args.crops_per_slide

    # Override save path if outdir is provided
    if getattr(args, 'outdir', None):
        cf.SAVEPATH = args.outdir
        
    # Setup logging
    now = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_')
    log_dir = os.path.join(cf.SAVEPATH, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(log_dir, f'additional_crops_{now}.log'),
        format='%(asctime)s %(levelname)-3s %(message)s', 
        level=logging.INFO
    )
    
    printNlog("Starting additional crop download")
    download_additional_crops(crop_size=args.size, output_type=args.type)
    printNlog("Finished additional crop download")

if __name__ == '__main__':
    main()