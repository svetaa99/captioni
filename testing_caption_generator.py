import numpy as np
from pickle import load
from PIL import Image
import matplotlib.pyplot as plt
import argparse

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import Xception, preprocess_input

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image path")
args = vars(ap.parse_args())
img_path = args['image']


def extract_features(filename, model):
    image = None

    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image!")
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break

    return in_text


def generate_caption(img_url):
    max_length = 32
    root_path = "C:\\Users\\Lenovo\\Documents\\Faks\\3_Godina\\6_Semestar\\ORI\\Captioni\\Flickr8k_Dataset\\Flicker8k_Dataset"
    path = root_path + "\\" + img_url
    tokenizer = load(open("tokenizer.p", "rb"))
    model = load_model('models/model_8.h5')
    xception_model = Xception(include_top=False, pooling='avg')

    photo = extract_features(path, xception_model)
    description = generate_desc(model, tokenizer, photo, max_length)

    print("\n\n")
    print(description)

    return description

if __name__ == "__main__":
  max_length = 32
  tokenizer = load(open("tokenizer.p", "rb"))
  model = load_model('models/model_8.h5')
  xception_model = Xception(include_top=False, pooling='avg')

  photo = extract_features(img_path, xception_model)
  description = generate_desc(model, tokenizer, photo, max_length)

  print("\n\n")
  print(description)
