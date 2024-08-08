#!/bin/bash

# Directory containing the image files
SOURCE_DIR="$1"

# Destination directory where the CCG directories will be created
DEST_DIR="$2"

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

# Loop through each file in the source directory
for file in "$SOURCE_DIR"/*.tif; do
  # Extract the CCG prefix (e.g., CCG10 from CCG10_003.tif)
  base_name=$(basename "$file")
  ccg_prefix=$(echo "$base_name" | grep -oE '^CCG[0-9]+')

  # Create a directory for the CCG prefix if it doesn't exist
  ccg_dir="$DEST_DIR/$ccg_prefix"
  mkdir -p "$ccg_dir"

  # Move the file into the corresponding directory
  mv "$file" "$ccg_dir/"
done
