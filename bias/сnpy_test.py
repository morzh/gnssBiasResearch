import numpy as np
from PIL import Image
import os


filepath = '/home/morzh/work/neutral_wideSmile.png'
path2save = '/home/morzh/work/FaceProjectApp/assets/image_sequences/neutral_wideSmile'

image = Image.open(filepath)
for idx in range(10):
    current_image = image.crop((1024*idx, 0, 1024*(idx + 1), 1024))
    current_image.save(os.path.join(path2save, str(idx)+'.jpg'))

