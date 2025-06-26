from PIL import Image
import os
import numpy as np
import json

def convert_tif_to_png_and_delete_original(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_dir, output_filename)

            try:
                with Image.open(input_path) as img:
                    img.convert("RGB").save(output_path, "PNG")
                os.remove(input_path)
                print(f"Converted and deleted: {filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

# Example usage
# convert_tif_to_png_and_delete_original("data/BingRGB/images/train", "data/BingRGB/images/train")
# convert_tif_to_png_and_delete_original("data/BingRGB/images/val", "data/BingRGB/images/val")

def generate_demo_gt(folder, demo):
    for file in os.listdir(folder):
        demo.save(f'{folder.replace("images", "annotations")}/{file}')

def generate_captions():

    data = {}
    example_text = "This patch is located in Shyampur union, under Tejgaon Development Circle Upazila of Dhaka District. Shyampur union has a population of 214,000, with a density of 16,525 people per square kilometer. The area of the union is 12.95 square kilometers, and the literacy rate is 97.0%. This patch is 10 km away from the nearest district center and 10 km away from the nearest upazila center. This patch is located outside of the District Sadar. This patch is located outside of the Upazila Sadar."

    for file in os.listdir('data/BingRGB/images/val'):
        data[file] = example_text
    
    for file in os.listdir('data/BingRGB/images/train'):
        data[file] = example_text
    
    for file in os.listdir('data/BingRGB/images/test'):
        data[file] = example_text
      
    
    with open('data/BingRGB/captions.json', "w") as f:
            json.dump(data, f, indent=4)


img = Image.open('data/ade20k/ADEChallengeData2016/annotations/training/ADE_train_00000001.png')
resized_img = img.resize((513, 513), resample=Image.BILINEAR)  # or Image.Resampling.BILINEAR for PIL>=10

arr = np.array(resized_img)

# Replace values > 5 with random integers between 0 and 5
mask = arr > 5
arr[mask] = np.random.randint(0, 6, size=np.count_nonzero(mask))

# Convert back to image
new_img = Image.fromarray(arr)

# generate_demo_gt('data/BingRGB/images/val', new_img)

generate_captions()

