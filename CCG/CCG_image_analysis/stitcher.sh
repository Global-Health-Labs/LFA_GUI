#!/bin/bash

# Directory containing the CCG directories
SOURCE_DIR="/home/anochmohan/LFA_GUI/CCG/Cropped"

# Output directory for concatenated images
OUTPUT_DIR="/home/anochmohan/LFA_GUI/CCG/Stitched"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Loop through each CCG directory in the source directory
for ccg_dir in "$SOURCE_DIR"/CCG*; do
  # Check if it's a directory
  if [ -d "$ccg_dir" ]; then
    # Extract the CCG name from the directory path
    ccg_name=$(basename "$ccg_dir")

    # Concatenate all .tif files in the directory into one .tif file
    # The files are sorted naturally to maintain order
    output_file="$OUTPUT_DIR/${ccg_name}_concatenated.tif"
    convert "$ccg_dir"/*.tif +append "$output_file"

    echo "Concatenated images in $ccg_dir into $output_file"
  fi
done
