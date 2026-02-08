from PIL import Image
import os

def add_white_background(image_path):
    try:
        img = Image.open(image_path).convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        
        # Alpha composite the image onto the white background
        combined = Image.alpha_composite(background, img)
        
        # Convert to RGB to remove alpha channel properly
        combined_rgb = combined.convert("RGB")
        
        combined_rgb.save(image_path)
        print(f"Successfully added white background to {image_path}")
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    target_image = "/Users/yangruoliu/Desktop/VideoDetective/images/example.png"
    if os.path.exists(target_image):
        add_white_background(target_image)
    else:
        print(f"File not found: {target_image}")
