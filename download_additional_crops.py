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
import shutil
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

from utils import create_directory_structure, printNlog, get_image_from_htk_response, has_excessive_white_background
import configs as cf

# Configuration imported from configs.py

# =============================================================================

def download_multiscale_zarr_crop(gc, slide_id, center_x, center_y, crop_size, zarr_path, cf_config, compression='blosc'):
    """Download and save multiscale zarr with same shape but different context"""
    if not ZARR_AVAILABLE:
        raise ImportError("zarr, dask, and scikit-image are required for zarr output. Install with: pip install zarr dask[array] scikit-image")
    
    zarr_group = None
    try:
        # Create zarr group (v3 compatible)
        zarr_group = zarr.open_group(zarr_path, mode='w')
        
        # Set up compression based on parameter (v3 compatible)
        if compression == 'blosc':
            try:
                from zarr.codecs import BloscCodec
                compressor = BloscCodec(cname='zstd', clevel=3, shuffle='shuffle')
            except ImportError:
                printNlog("Warning: BloscCodec not available, using no compression", level='error')
                compressor = None
        elif compression == 'none' or compression is None:
            compressor = None
        else:
            raise ValueError(f"Unsupported compression type: {compression}. Use 'none' or 'blosc'.")
        
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
            
            # Download the crop for this level
            resp = gc.get(getStr, jsonResp=False)
            image = get_image_from_htk_response(resp)
            image_array = np.array(image)
            
            # Apply white background filter on the first level (highest resolution)
            if level == 0 and cf_config.ENABLE_WHITE_BACKGROUND_FILTER:
                if has_excessive_white_background(image, cf_config.WHITE_BACKGROUND_THRESHOLD, cf_config.MAX_WHITE_PERCENTAGE):
                    raise ValueError("Excessive white background detected")
            
            # Ensure consistent output size
            if image_array.shape[:2] != (crop_size, crop_size):
                from PIL import Image as PILImage
                image_pil = PILImage.fromarray(image_array)
                image_pil = image_pil.resize((crop_size, crop_size), PILImage.LANCZOS)
                image_array = np.array(image_pil)
            
            # Save to zarr - v3 compatible
            if compressor is not None:
                zarr_group.create_array(f's{str(level)}', data=image_array, chunks=(512, 512, 3), 
                                      compressor=compressor)
            else:
                zarr_group.create_array(f's{str(level)}', data=image_array, chunks=(512, 512, 3))
        
        # Add metadata for multiscale (simplified for v3)
        zarr_group.attrs['multiscales'] = [{
            'version': '0.4',
            'datasets': [
                {'path': '0'},
                {'path': '1'}, 
                {'path': '2'}
            ]
        }]
        
    except Exception as e:
        # If any error occurs, clean up the zarr directory
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        raise e

def get_slide_dimensions(gc, slide_id):
    """Get slide dimensions from HistomicsTK API"""
    try:
        props = gc.get(f'item/{slide_id}/tiles')
        return props.get('sizeX', 0), props.get('sizeY', 0)
    except Exception as e:
        printNlog(f"Error getting dimensions for slide {slide_id}: {e}", level='error')
        return 0, 0

def download_slide_overview(gc, slide_id, target_size=1000):
    """Download a low-resolution overview of the entire slide for tissue detection"""
    try:
        # Get slide dimensions
        props = gc.get(f'item/{slide_id}/tiles')
        slide_width = props.get('sizeX', 0)
        slide_height = props.get('sizeY', 0)
        
        if slide_width == 0 or slide_height == 0:
            return None, 0, 0
        
        # Calculate downsampling to get roughly target_size on longest dimension
        scale_factor = max(slide_width, slide_height) / target_size
        overview_width = int(slide_width / scale_factor)
        overview_height = int(slide_height / scale_factor)
        
        # Download overview at low resolution
        getStr = f"/item/{slide_id}/tiles/region?left=0&right={slide_width}&top=0&bottom={slide_height}&width={overview_width}&height={overview_height}"
        
        resp = gc.get(getStr, jsonResp=False)
        overview_image = get_image_from_htk_response(resp)
        
        return overview_image, scale_factor, (slide_width, slide_height)
        
    except Exception as e:
        printNlog(f"Error downloading slide overview: {e}", level='error')
        return None, 0, 0

def create_tissue_mask(overview_image, white_threshold=200):
    """Create a binary mask identifying tissue regions (non-white areas)"""
    # Convert to numpy array
    overview_array = np.array(overview_image)
    
    # Create tissue mask (inverse of white background detection)
    if len(overview_array.shape) == 3:
        # For RGB images, tissue is where NOT all channels are above threshold
        tissue_mask = ~np.all(overview_array > white_threshold, axis=2)
    else:
        # For grayscale images
        tissue_mask = overview_array <= white_threshold
    
    return tissue_mask

def sample_tissue_coordinates(tissue_mask, scale_factor, slide_dims, crop_width, crop_height, margin, n_crops, max_attempts=1000):
    """Sample crop coordinates from tissue regions using the tissue mask"""
    crops = []
    slide_width, slide_height = slide_dims
    
    # Ensure we have enough space for crops
    max_x = slide_width - crop_width - margin
    max_y = slide_height - crop_height - margin
    
    if max_x <= margin or max_y <= margin:
        printNlog(f"Slide too small for crops: {slide_width}x{slide_height}", level='error')
        return crops
    
    # Get tissue pixel coordinates in overview space
    tissue_coords = np.where(tissue_mask)
    if len(tissue_coords[0]) == 0:
        printNlog("No tissue regions found in slide overview", level='error')
        return crops
    
    attempts = 0
    while len(crops) < n_crops and attempts < max_attempts:
        attempts += 1
        
        # Randomly select a tissue pixel from the overview
        idx = random.randint(0, len(tissue_coords[0]) - 1)
        overview_y = tissue_coords[0][idx]
        overview_x = tissue_coords[1][idx]
        
        # Convert overview coordinates back to full resolution
        center_x = int(overview_x * scale_factor)
        center_y = int(overview_y * scale_factor)
        
        # Calculate crop bounds centered on tissue region
        half_width = crop_width // 2
        half_height = crop_height // 2
        xmin = center_x - half_width
        ymin = center_y - half_height
        xmax = center_x + half_width
        ymax = center_y + half_height
        
        # Check if crop fits within slide bounds with margin
        if (xmin >= margin and ymin >= margin and 
            xmax <= slide_width - margin and ymax <= slide_height - margin):
            
            crops.append({
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })
    
    if len(crops) < n_crops:
        printNlog(f"Could only find {len(crops)} valid tissue regions out of {n_crops} requested after {attempts} attempts")
    
    return crops

def generate_random_crop_coordinates(slide_width, slide_height, crop_width, crop_height, margin, n_crops):
    """Generate random crop coordinates ensuring they fit within slide bounds (fallback method)"""
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

def download_additional_crops(crop_size=None, output_type='png', compression='blosc', nfiles=None):
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
    additional_crops_dir = cf.SAVEPATH
    os.makedirs(additional_crops_dir, exist_ok=True)
    
    # Get all slides
    resp = gc.get("item?folderId=%s&limit=1000000" % cf.source_folder_id)
    slides = {s['name'][:12]: {'name': s['name'], '_id': s['_id']} for s in resp}
    
    # Filter slides if specified
    slide_list = list(slides.keys())
    if cf.SLIDES_TO_KEEP is not None:
        slide_list = [j for j in slide_list if j in cf.SLIDES_TO_KEEP]
    
    # Limit number of slides if nfiles specified
    if nfiles is not None:
        slide_list = slide_list[:nfiles]
    
    
    
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
        
        # Choose sampling strategy based on configuration
        if cf.ENABLE_TISSUE_AWARE_SAMPLING:
            # Download slide overview for tissue detection
            printNlog("  Downloading slide overview for tissue detection...")
            overview_image, scale_factor, slide_dims = download_slide_overview(gc, slide_id, target_size=cf.OVERVIEW_SIZE)
            
            if overview_image is None:
                printNlog(f"Skipping slide {slide_name} - could not download overview", level='error')
                continue
            
            # Save overview to logs for inspection
            log_dir = os.path.join(cf.SAVEPATH, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            overview_path = os.path.join(log_dir, f"{slide_name}_overview.jpg")
            overview_image.save(overview_path, "JPEG", quality=85)
            printNlog(f"  Saved overview to {overview_path}")
            
            # Create tissue mask from overview
            tissue_mask = create_tissue_mask(overview_image, cf.WHITE_BACKGROUND_THRESHOLD)
            
            # Sample crop coordinates from tissue regions
            crops = sample_tissue_coordinates(
                tissue_mask, scale_factor, slide_dims,
                crop_width, crop_height, 
                cf.EDGE_MARGIN, cf.CROPS_PER_SLIDE,
                max_attempts=cf.MAX_SAMPLING_ATTEMPTS
            )
            
            # Fallback to random sampling if tissue sampling fails
            if not crops:
                printNlog("  Falling back to random coordinate sampling...")
                crops = generate_random_crop_coordinates(
                    slide_width, slide_height, 
                    crop_width, crop_height, 
                    cf.EDGE_MARGIN, cf.CROPS_PER_SLIDE
                )
        else:
            # Use original random sampling approach
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
                    
                    # Apply white background filter if enabled
                    if cf.ENABLE_WHITE_BACKGROUND_FILTER:
                        if has_excessive_white_background(rgb, cf.WHITE_BACKGROUND_THRESHOLD, cf.MAX_WHITE_PERCENTAGE):
                            printNlog(f"    Skipping crop {crop_idx} - excessive white background")
                            continue
                    
                    filepath = os.path.join(additional_crops_dir, filename + ".png")
                    rgb.save(filepath)
                    
                elif output_type == 'zarr':
                    # For zarr, use multiscale download approach
                    filepath = os.path.join(additional_crops_dir, filename + ".zarr")
                    try:
                        download_multiscale_zarr_crop(gc, slide_id, center_x, center_y, crop_size if crop_size else crop_width, filepath, cf, compression)
                    except ValueError as e:
                        if "Excessive white background detected" in str(e):
                            printNlog(f"    Skipping crop {crop_idx} - excessive white background")
                            continue
                        else:
                            raise
                
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
        '--compression',
        choices=['none', 'blosc'],
        default='none',
        help='Compression method for zarr files: "none" for no compression, "blosc" for lossless compression (default)'
    )

    parser.add_argument(
        '-o', '--outdir',
        type=str,
        default=None,
        help='Root output directory to write results and logs (overrides configs.SAVEPATH)'
    )
    
    parser.add_argument(
        '-n', '--nfiles',
        type=int,
        default=None,
        help='Maximum number of files to download (default: None means download all)'
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
    download_additional_crops(crop_size=args.size, output_type=args.type, compression=args.compression, nfiles=args.nfiles)
    printNlog("Finished additional crop download")

if __name__ == '__main__':
    main()