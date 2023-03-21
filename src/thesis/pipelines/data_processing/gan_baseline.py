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

%load_ext autoreload
%autoreload 2
pd.set_option('display.float_format', lambda x: '%.5f' % x)
# Load the freMTPL2freq dataset
train_data = pd.read_pickle('Data/Throughput/MTPL_sev_train.pickle')
test_data = pd.read_pickle('Data/Throughput/MTPL_sev_test.pickle'# Transforming into bins
column_trans = ColumnTransformer(
    [
        ("passthrough_numeric",
         "passthrough",
         ["BonusMalus", 'Density', 'VehAge', 'DrivAge', 'Area', 'ClaimAmount']),
        (
            "onehot_categorical",
            OneHotEncoder(),
            ["VehBrand", "VehPower", "VehGas", "Region"],
        )
    ],
    remainder="drop",
)
Xy = column_trans.fit_transform(train_data)
Xy = pd.DataFrame.sparse.from_spmatrix(Xy)
Xy.columns = [f'Col{x}' for x in Xy.columns]
Xy.head()


# Define the hyperparameters for the CTGAN GAN
param_grid = {'epochs': [100, 200, 300],
              'gen_dim': [(128, 128), (256, 256), (512, 512)],
              'dis_dim': [(128, 128), (256, 256), (512, 512)],
              'batch_norm': [True, False],
              'batch_size': [500, 1000, 2000]}

# Define the CTGAN GAN object
ctgan = CTGAN(verbose=False, cuda=True, epochs=2)
ctgan.fit(Xy)
ctgan.sample(2)
# Perform GridSearchCV to tune hyperparameters
#gs = GridSearchCV(estimator=ctgan, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')


# Generate new synthetic data using the best hyperparameters
#ctgan.set_params(**gs.best_params_)
#ctgan.fit(train_data)
#synthetic_data = ctgan.sample(len(test_data))

# Compute evaluation metrics
#print('AUC:', roc_auc_score(test_data['ClaimNb'], synthetic_data['ClaimNb']))
#print('Precision:', precision_score(test_data['ClaimNb'], synthetic_data['ClaimNb']))
#print('Recall:', recall_score(test_data['ClaimNb'], synthetic_data['ClaimNb']))
#print('F1 score:', f1_score(test_data['ClaimNb'], synthetic_data['ClaimNb']))

# Plot the distribution of the original and synthetic data
#sns.histplot(data['ClaimNb'], kde=True, color='blue', alpha=0.5, label='Original')
#sns.histplot(synthetic_data['ClaimNb'], kde=True, color='orange', alpha=0.5, label='Synthetic')
#plt.legend()
#plt.show()
