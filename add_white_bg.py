from PIL import Image

def add_white_background(image_path):
    try:
        img = Image.open(image_path).convert("RGBA")
        
        # Create a white background image
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        
        # Paste the image on top of the background
        # usage of alpha_composite is better for transparency issues
        combined = Image.alpha_composite(background, img)
        
        # Convert to RGB to remove alpha channel
        final_img = combined.convert("RGB")
        
        # Save the image, overwriting the original
        final_img.save(image_path)
        print(f"Successfully added white background to {image_path}")
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    add_white_background("/Users/yangruoliu/Desktop/VideoDetective/images/figure1_final_final.png")
