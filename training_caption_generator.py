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

from tqdm import tqdm

dataset_text = "C:\\Users\\Lenovo\\Documents\\Faks\\3_Godina\\6_Semestar\\ORI\\Captioni\\Flickr8k_text"
dataset_images = "C:\\Users\\Lenovo\\Documents\\Faks\\3_Godina\\6_Semestar\\ORI\\Captioni\\Flickr8k_Dataset\\Flicker8k_Dataset"


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
        img_num = int(tokens[1][0])  # first element after # in line
        img_source = tokens[0]
        if img_num == 0:
            captions[img_source] = []
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


def save_descriptions(descriptions):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc)
    data = "\n".join(lines)
    file = open("descriptions.txt", "w")
    file.write(data)
    file.close()


def extract_features(directory):
    # removing last layer from the net because there is no need to classify the object, but only to get the vector
    model = Xception(include_top=False, pooling='avg')
    features = {}
    for img in tqdm(os.listdir(directory)):
        filename = directory + "/" + img
        image = Image.open(filename)
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        image = image / 127.5
        image = image - 1.0

        feature = model.predict(image)
        features[img] = feature
    return features


def load_photos(filename):
    text = load_doc(filename)
    photos = text.split("\n")[:-1]

    return photos


def load_clean_description(filename, photos):
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):
        words = line.split()
        if len(words) < 1:
            continue

        image, image_caption = words[0], words[1:]

        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = "<start> " + " ".join(image_caption) + " <end>"
            descriptions[image].append(desc)

    return descriptions


def load_features(photos):
    all_features = load(open("features.p", "rb"))
    features = {k: all_features[k] for k in photos}

    return features


def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(desc) for desc in descriptions[key]]

    return all_desc


def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)

    return tokenizer


def calculate_max_length(descriptions):
    desc_list = dict_to_list(descriptions)

    return max(len(d.split()) for d in desc_list)


if __name__ == "__main__":
    # CREATING DESCRIPTION.TXT AND CLEANING TOKEN.TXT FILE

    tokens_file = dataset_text + "/" + "Flickr8k.token.txt"

    captions = all_img_captions(tokens_file)
    captions = clean_text(captions)

    vocabulary = create_text_vocabulary(captions)

    save_descriptions(captions)

    # EXTRACTING FEATURES WITH XCEPTION MODEL

    # features = extract_features(dataset_images)
    # dump(features, open("features.p", "wb"))

    # LOADING TRAINING DATASET

    filename = dataset_text + "/" + "Flickr_8k.trainImages.txt"
    train_imgs = load_photos(filename)
    train_descriptions = load_clean_description("descriptions.txt", train_imgs)
    train_features = load_features(train_imgs)

    # CREATING TOKENIZER.P FILE WITH TOKENIZED INDEXED FROM WORDS IN DESCRIPTIONS

    tokenizer = create_tokenizer(train_descriptions)
    dump(tokenizer, open("tokenizer.p", "wb"))
    vocab_size = len(tokenizer.word_index) + 1

    max_lengt = calculate_max_length(train_descriptions)
