import os
import random
from PIL import Image
import numpy as np
import pickle

train_dir = 'data/train2017/'
val_dir = 'data/val2017/'
test_dir = 'data/test2017/'

def resize_and_convert(img, width=100, height=100):
    """Resizes img using Pillow, then converts to (width, height,
    channels)-dimensional numpy array."""
    img = img.resize((width, height))
    return np.asarray(img)

def process_images(dir, max_img=None):
    """For all files in the specified directory, will resize and convert each
    image, as well as creating three randomly cropped duplicate images. Returns
    a list of numpy arrays."""
    all_img = []

    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    if max_img is not None:
        files = random.sample(files, max_img)
    files_len = len(files)
    for i, file in enumerate(files):
        fullpath = os.path.join(dir, file)
        try:
            img = Image.open(fullpath)
            size = img.size
            all_img.append(resize_and_convert(img))

            # for each image, randomly crop up to 5 px from each side
            for _ in range(3):
                crop_dim = (random.randint(0, 5), random.randint(0, 5),
                            random.randint(0, 5), random.randint(0, 5))
                # img.crop((x_start, y_start, x_end, y_end))
                crop = img.crop((crop_dim[0], crop_dim[1],
                    size[0]-crop_dim[2], size[1]-crop_dim[3]))
                all_img.append(resize_and_convert(crop))
        except:
            pass
        if i % 500 == 0:
            print("({}/{}) {}%".format(i, files_len, (i/files_len)*100))
    return all_img

print('Processing training data:')
training = process_images(train_dir, max_img=10000)
random.shuffle(training)
with open('data/training_data.pkl', 'wb') as f:
    pickle.dump(training, f, protocol=pickle.HIGHEST_PROTOCOL)
del training  # remove from memory

print('Processing validation data:')
validation = process_images(val_dir)
random.shuffle(validation)
with open('data/validation_data.pkl', 'wb') as f:
    pickle.dump(validation, f, protocol=pickle.HIGHEST_PROTOCOL)
del validation

print('Processing test data:')
test = process_images(test_dir, max_img=10000)
random.shuffle(test)
with open('data/test_data.pkl', 'wb') as f:
    pickle.dump(test, f, protocol=pickle.HIGHEST_PROTOCOL)
del test  # remove from memory

