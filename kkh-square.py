from PIL import Image
import os

def add_padding_and_convert_to_bw(image_path, output_path):
    with Image.open(image_path) as img:
        # Convert the image to grayscale
        img = img.convert("L")

        width, height = img.size

        # Determine the size of the new square image
        new_size = max(width, height)

        # Create a new white square image (with grayscale mode)
        new_img = Image.new("L", (new_size, new_size), 255)  # 255 is white in grayscale
        
        # Calculate padding sizes
        left_padding = (new_size - width) // 2
        top_padding = (new_size - height) // 2
        
        # Paste the original image onto the new square image with padding
        new_img.paste(img, (left_padding, top_padding))

        # Save the new image
        new_img.save(output_path)

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg'):
            input_path = os.path.join(directory, filename)
            output_path = input_path  # Overwrite the original image
            add_padding_and_convert_to_bw(input_path, output_path)

if __name__ == "__main__":
    # Define directories
    train_dir = 'data/train_kr_sq'
    test_dir = 'data/test_sq'
    valid_dir = 'data/valid_sunho_sq'

    # Process both directories
    process_directory(train_dir)
    process_directory(test_dir)
    process_directory(valid_dir)
