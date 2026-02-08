from PIL import Image
import os

def add_white_background(image_path):
    try:
        img = Image.open(image_path).convert("RGBA")
        
        # Create a white background image
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        
        # Composite the image onto the background
        combined = Image.alpha_composite(background, img)
        
        # Convert back to RGB to remove alpha channel (optional, but good for compatibility)
        final_img = combined.convert("RGB")
        
        # Save
        final_img.save(image_path)
        print(f"Successfully added white background to {image_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    target_file = "/Users/yangruoliu/Desktop/VideoDetective/images/example.png"
    if os.path.exists(target_file):
        add_white_background(target_file)
    else:
        print(f"File not found: {target_file}")
