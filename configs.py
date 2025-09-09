APIURL = 'https://demo.kitware.com/histomicstk/api/v1/'
# apiKey = None  # interactive login
apiKey = 'n0Kp1ez8YOnOiWNoACryzeBlIzbUDW3iOD2DmPLI'

source_folder_id = '5bbdeba3e629140048d017bb'

SAVEPATH = './'
ROIBOUNDSPATH = './meta/roiBounds.csv'

# Set either MPP or MAG.
# If both are None, base (scan) magnification is used.

# Microns-per-pixel -- best use this
# MPP of 0.25 is "standardized" at 40x using original Aperio scanners
# MPP = None
MPP = 0.25

# If you prefer to use whatever magnification is reported
MAG = None
# MAG = 40.0

# What things to download? -- comment out whet you dont want
PIPELINE = (
    'images',
    'masks',
    # 'annotations',
)

# if you only want to download data for specific slides
SLIDES_TO_KEEP = None
# SLIDES_TO_KEEP = ['TCGA-BH-A1FC', 'TCGA-A8-A07C', 'TCGA-A1-A0SK']

# =============================================================================
# Additional crop settings (for download_additional_crops.py)
# =============================================================================

# Target crop size (similar to annotated crops)
ADDITIONAL_CROP_WIDTH = 5000   # pixels at base resolution
ADDITIONAL_CROP_HEIGHT = 4300  # pixels at base resolution

# Number of random crops per slide
CROPS_PER_SLIDE = 3

# Minimum distance from slide edges (to avoid empty regions)
EDGE_MARGIN = 10000  # pixels

# Random seed for reproducible sampling
RANDOM_SEED = 42

# White background filtering settings
ENABLE_WHITE_BACKGROUND_FILTER = True
WHITE_BACKGROUND_THRESHOLD = 200  # Pixel values above this are considered "white"
MAX_WHITE_PERCENTAGE = 40  # Maximum allowed percentage of white pixels
