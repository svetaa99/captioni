import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load

from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout

from tqdm import tqdm_notebook as tqdm

dataset_text = "C:\\Users\\Lenovo\\Documents\\Faks\\3_Godina\\6_Semestar\\ORI\\Captioni\\Flickr8k_text"
dataset_images = "C:\\Users\\Lenovo\\Documents\\Faks\\3_Godina\\6_Semestar\\ORI\\Captioni\\Flickr8k_Dataset"


def load_doc(filename):
    f = open(filename)
    text = f.read()
    f.close()

    return text


def all_img_captions(filename):
    f = load_doc(filename)
    captions = {}
    for line in f.split("\n"):
        tokens = line.split("#")
        img_num = int(tokens[1][0])      # first element after # in line
        img_source = tokens[0]
        if img_num == 0:
            captions[img_source] = []
        else:
            img_description = tokens[1].split("\t")[1]
            captions[img_source].append(img_description)

    return captions


def clean_text(captions):
    table = str.maketrans('', '', string.punctuation)
    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            img_caption.replace("-", " ")
            desc = img_caption.split()

            # converts to lowercase
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [word.translate(table) for word in desc]
            # remove hanging 's and a
            desc = [word for word in desc if (len(word) > 1)]
            # remove tokens with numbers in them
            desc = [word for word in desc if (word.isalpha())]
            # convert back to string

            img_caption = ' '.join(desc)
            captions[img][i] = img_caption

    return captions


def create_text_vocabulary(captions):
    text_vocabulary = set()
    for k, v in captions.items():
        for value in v:
            text_vocabulary.update(value.split())

    return text_vocabulary


def save_descriptions(captions):
    f = open("descriptions.txt", "a")
    for k, v in captions.items():
        for value in v:
            line = k + "\t" + value + "\n"
            f.write(line)

    f.close()


if __name__ == "__main__":
    tokens_file = dataset_text + "/" + "Flickr8k.token"

    captions = all_img_captions(tokens_file)
    captions = clean_text(captions)

    vocabulary = create_text_vocabulary(captions)

    save_descriptions(captions)
