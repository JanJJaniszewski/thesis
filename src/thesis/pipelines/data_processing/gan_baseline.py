# Import necessary libraries
from ctgan import CTGAN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def train_gan_baseline_sev(train_data, parameters):
    nums = parameters['numeric_features']
    cats = parameters['categorical_features']

    column_trans = ColumnTransformer(
        [
            ("passthrough_numeric", "passthrough", nums),
            ("onehot_categorical", OneHotEncoder(sparse=False), cats)
        ],
        remainder="drop"
    )

    Xy = column_trans.fit_transform(train_data)

    # Define the CTGAN GAN object
    ctgan = CTGAN(verbose=False, cuda=True, epochs=2)
    ctgan.fit(Xy)
    ctgan.save('data/07_models/gan_baseline_sev.pickle')
    return ctgan
