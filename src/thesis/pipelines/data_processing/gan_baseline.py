# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools
import plotly.express as px
from scipy import stats
from copy import deepcopy
from sklearn.model_selection import GroupShuffleSplit
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
# Import necessary libraries
import pandas as pd
from ctgan import CTGAN
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_gan_baseline(train_data, parameters):
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