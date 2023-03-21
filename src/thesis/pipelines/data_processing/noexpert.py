# Import necessary libraries
from ctgan import CTGAN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def dataprep_noexpert(freq, sev):
    return freq, sev


def regression_noexpert_freq():
    pass

def regression_noexpert_sev():
    pass

def datagen_noexpert_freq(ctgan):
    return generated_data

def datagen_noexpert_sev(ctgan):
    return generated_data

def gan_noexpert_sev(train_data, parameters):
    nums = parameters['numeric_features']
    cats = parameters['categorical_features']

    column_trans = ColumnTransformer(
        [
            ("passthrough_numeric", "passthrough", nums),
            ("onehot_categorical", OneHotEncoder(sparse_output=False), cats)
        ],
        remainder="drop"
    )

    Xy = column_trans.fit_transform(train_data)

    # Define the CTGAN GAN object
    print('Running CTGAN')
    ctgan = CTGAN(verbose=False, cuda=True, epochs=2)
    ctgan.fit(Xy)
    print('Finished CTGAN. Saving it')
    ctgan.save('data/07_models/gan_baseline_sev.pickle')
    return ctgan

def gan_noexpert_freq(train_data, parameters):
    ctgan = CTGAN(verbose=False, cuda=True, epochs=2)
    return ctgan