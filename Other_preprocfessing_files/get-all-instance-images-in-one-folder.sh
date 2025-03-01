mkdir -p /home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Instance_images_dir

count=1
find /home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Input -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | while read file; do
    extension="${file##*.}"  # Extract file extension
    cp "$file" "/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Instance_images_dir/input_${count}.${extension}"
    ((count++))  # Increment counter
done
