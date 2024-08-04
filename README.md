# HTML Captioning with VGG16 and LSTM

This project demonstrates a neural network model that generates HTML captions for images. It leverages the VGG16 model for image feature extraction and a Long Short-Term Memory (LSTM) network for sequence generation.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Generating Captions](#generating-captions)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Introduction

This project focuses on generating HTML captions for images using a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN). The VGG16 model, pre-trained on ImageNet, is used to extract features from the images. These features are then passed to an LSTM network that generates the corresponding HTML captions.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or Google Colab
- Keras
- TensorFlow
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/html-captioning.git
    cd html-captioning
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Load and preprocess the data:
    ```python
    from keras.preprocessing.image import img_to_array, load_img
    import numpy as np
    import yfinance as yf

    images = []
    for i in range(2):
        images.append(img_to_array(load_img('/content/screenshot.png', target_size=(224, 224))))
    images = np.array(images, dtype=float)
    images = preprocess_input(images)
    ```

2. Define and train the model:
    ```python
    from keras.applications.vgg16 import VGG16, preprocess_input
    from keras.models import Model
    from keras.layers import Embedding, TimeDistributed, RepeatVector, LSTM, concatenate, Input, Reshape, Dense
    from keras.optimizers import RMSprop

    # Load the VGG16 model
    VGG = VGG16(weights='imagenet', include_top=True)
    features = VGG.predict(images)

    # Define the model architecture
    vgg_feature = Input(shape=(1000,))
    vgg_feature_dense = Dense(5)(vgg_feature)
    vgg_feature_repeat = RepeatVector(3)(vgg_feature_dense)

    language_input = Input(shape=(3, 3))
    language_model = LSTM(5, return_sequences=True)(language_input)

    decoder = concatenate([vgg_feature_repeat, language_model])
    decoder = LSTM(5, return_sequences=False)(decoder)
    decoder_output = Dense(3, activation='softmax')(decoder)

    model = Model(inputs=[vgg_feature, language_input], outputs=decoder_output)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop())
    ```

3. Train the model:
    ```python
    html_input = np.array([[[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]], [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    next_words = np.array([[0., 1., 0.], [0., 0., 1.]])

    model.fit([features, html_input], next_words, batch_size=2, shuffle=False, epochs=1000)
    ```

## Model Architecture

The model consists of the following components:

- **VGG16 Model**: Extracts features from the input images.
- **Dense Layer**: Reduces the dimensionality of the extracted features.
- **RepeatVector Layer****: Repeats the feature vector to match the length of the caption.
- **LSTM Layers**: Processes the input sequence and the repeated feature vector.
- **Dense Output Layer**: Predicts the next word in the sequence.

## Training

The model is trained using a small dataset of HTML captions. Each image is associated with a sequence of HTML tokens, which are used to train the LSTM network. The model is optimized using the RMSprop optimizer and categorical cross-entropy loss.

## Generating Captions

After training, the model can generate HTML captions for new images. The process involves:

1. Preprocessing the image using VGG16.
2. Feeding the preprocessed image and a start token into the model.
3. Predicting the next token in the sequence.
4. Repeating the process until the end token is generated.

Example:
```python```
start_token = [1., 0., 0.]
sentence = np.zeros((1, 3, 3))
sentence[0][2] = start_token

second_word = model.predict([np.array([features[1]]), sentence])
sentence[0][1] = start_token
sentence[0][2] = np.round(second_word)

third_word = model.predict([np.array([features[1]]), sentence])
sentence[0][0] = start_token
sentence[0][1] = np.round(second_word)
sentence[0][2] = np.round(third_word)

vocabulary = ["start", "<HTML><center><H1>Hello World!</H1><center></HTML>", "end"]
html = ""
for i in sentence[0]:
    html += vocabulary[np.argmax(i)] + ' '

from IPython.core.display import display, HTML
display(HTML(html[6:49]))

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## Acknowledgements

We would like to extend our gratitude to the following resources and communities:

- [Keras](https://keras.io/): For providing a user-friendly API for building and training neural network models.
- [TensorFlow](https://www.tensorflow.org/): For offering a robust platform for machine learning and deep learning.
- [VGG16 Model](https://keras.io/api/applications/vgg/): For the pre-trained model used for image feature extraction.
- [Google Colab](https://colab.research.google.com/): For providing an accessible platform to develop and test machine learning models.

---

Feel free to customize this `README.md` file to better fit your project's specifics and your preferences.

