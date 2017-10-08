import os
from PIL import Image

def resize_image(image, size):
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, resized_dir, size):
    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, [256,256])
                img.save(os.path.join(resized_dir, image), img.format)
            if i%100 == 0:
                print("{} of {} images are saved to {}".format(i, num_images, resized_dir))