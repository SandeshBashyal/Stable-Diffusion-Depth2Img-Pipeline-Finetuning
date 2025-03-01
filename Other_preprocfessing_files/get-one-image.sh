#!/bin/bash

# Define the source directory and output directory
src_dir="/home/rmuproject/rmuproject/data"
dst_dir="/home/rmuproject/rmuproject/one-image-folder"

# Create output directory if it doesn't exist
mkdir -p "$dst_dir"

# Initialize a counter
counter=1

# Find and process the images
find "$src_dir" -mindepth 3 -name "0.png" | while read file; do
    # Define new filename
    new_filename="image_${counter}.png"

    # Copy and rename the file
    cp "$file" "$dst_dir/$new_filename"

    # Increment counter
    ((counter++))
done

echo "Renaming and copying completed!"
