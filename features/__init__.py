import numpy as np
import tensorflow as tf
from keras import backend as K
from skimage.feature import hog

from utils import rgb2gray


def extract_hog(img, orientations, pixels_per_cell, cells_per_block):
    gray = rgb2gray(img)

    return hog(
        image=gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=False,
        transform_sqrt=True
    )


def extract_cnn_codes(model_factory, X, y, data_gen, batch_size=16, save=None):
    with tf.Session() as sess:
        K.set_session(sess)
        K.set_learning_phase(0)  # 0 - test,  1 - train

        model = model_factory()

        cnn_codes = model.predict_generator(
            data_gen(sess, X, y, batch_size), len(y)/batch_size, verbose=1
        )

        if save:
            np.savez_compressed(
                f'./features/{model_factory.__name__}_{save}.npz',
                cnn_codes=cnn_codes,
                y=y
            )

        return cnn_codes
