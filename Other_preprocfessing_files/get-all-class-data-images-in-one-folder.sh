mkdir -p class_images_dir

count=1
find /home/rmuproject/rmuproject/data -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | while read file; do
    extension="${file##*.}"  # Extract file extension
    cp "$file" "class_images_dir/image_${count}.${extension}"
    ((count++))  # Increment counter
done
