def generate_gan(ctgan, train):
    generated_data = ctgan.sample(1000)
    return generated_data

def regression_common_freq(train):
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import itertools
    from sklearn.linear_model import GammaRegressor

    df_freq = freq0.copy(deep=True)
    df_freq = freq0.iloc[freq0.drop(['IDpol', 'Exposure', 'ClaimNb'], axis=1).drop_duplicates().index]
    df_freq = df_freq.reset_index(drop=True)
    df_freq['GroupID'] = df_freq.index + 1
    df_freq = pd.merge(freq0, df_freq, how='left')
    df_freq['GroupID'] = df_freq['GroupID'].fillna(method='ffill')

    for df1 in [train, test]:
        df1['VehAge'] = df1['VehAge'].apply(lambda x: 20 if x > 20 else x)
        df1['DrivAge'] = df1['DrivAge'].apply(lambda x: 90 if x > 90 else x)
        df1['BonusMalus'] = df1['BonusMalus'].apply(lambda x: 150 if x > 150 else int(x))
        df1['VehPowerGLM'] = df1['VehPower'].apply(lambda x: 9 if x > 9 else x)
        df1['VehPowerGLM'] = df1['VehPowerGLM'].apply(lambda x: str(x))
        df1['VehAgeGLM'] = pd.cut(df1['VehAge'], bins=[0, 1, 10, np.inf], labels=[1, 2, 3], include_lowest=True)
        df1['DrivAgeGLM'] = pd.cut(df1['DrivAge'], bins=[18, 21, 26, 31, 41, 51, 71, np.inf],
                                   labels=[1, 2, 3, 4, 5, 6, 7], include_lowest=True)
        df1['BonusMalusGLM'] = df1['BonusMalus']
        df1['DensityGLM'] = df1['Density']

    mask_train = train["ClaimAmount"] > 0
    mask_test = test["ClaimAmount"] > 0

    glm_sev = GammaRegressor()

    glm_sev.fit(
        X_train[mask_train.values],
        df_train.loc[mask_train, "AvgClaimAmount"],
        sample_weight=df_train.loc[mask_train, "ClaimNb"],
    )

    scores = score_estimator(
        glm_sev,
        X_train[mask_train.values],
        X_test[mask_test.values],
        df_train[mask_train],
        df_test[mask_test],
        target="AvgClaimAmount",
        weights="ClaimNb",
    )
    print("Evaluation of GammaRegressor on target AvgClaimAmount")
    print(scores)

    lst_model = []
    lst_params = []
    lst_aic = []
    lst_insampleloss = []
    lst_outsampleloss = []
    lst_avgfreq = []

    def save_result(fittedmodel, model_desc, col_name):
        train.insert(0, col_name, fittedmodel.predict(train, offset=np.log(train['Exposure'])))
        test.insert(0, col_name, fittedmodel.predict(test, offset=np.log(test['Exposure'])))
        lst_model.append(model_desc)
        lst_params.append(len(fittedmodel.params))
        lst_aic.append((2 * (fittedmodel.df_model + 1) - 2 * fittedmodel.llf))
        lst_insampleloss.append(poisson_deviance(train[col_name], train['ClaimNb']))
        lst_outsampleloss.append(poisson_deviance(test[col_name], test['ClaimNb']))
        lst_avgfreq.append(sum(test[col_name]) / sum(test['Exposure']))
        return pd.DataFrame({
            'Model': lst_model,
            'ParameterCount': lst_params,
            'AIC': lst_aic,
            'InSampleLoss': lst_insampleloss,
            'OutSampleLoss': lst_outsampleloss,
            'AvgFrequency': lst_avgfreq
        })

    def drop1(formula, model, data):
        x = [i for i in formula.split('~')[1].split('+')]
        drop1_stats = {}
        for k in range(1, len(x) + 1):
            for variables in itertools.combinations(x, k):
                if len(variables) == len(x) - 1:
                    predictors = list(variables)
                    i = True
                    independent = ''
                    for p in predictors:
                        if i:
                            independent = p
                            i = False
                        else:
                            independent += '+ {}'.format(p)
                    regression = 'ClaimNb ~ {}'.format(independent)
                    print('Dropping <' + [i for i in x if i not in predictors][
                        0] + '> from the model and recalculate statistics ... \n')
                    res = smf.glm(formula=regression, data=data,
                                  family=sm.families.Poisson(link=sm.families.links.log()),
                                  offset=np.log(data['Exposure'])).fit()
                    drop1_stats[[i for i in x if i not in predictors][0]] = (2 * (k + 1) - 2 * res.llf,
                                                                             res.deviance,
                                                                             res.deviance - model.deviance,
                                                                             res.df_model,
                                                                             # 1-stats.chi2.cdf(res.deviance - model.deviance, res.df_model)
                                                                             )
        df_out = pd.DataFrame(drop1_stats).T
        df_out.columns = ['AIC', 'Deviance', 'LRT', 'DoF']
        return df_out

    formula1 = "Frequency ~ VehPowerGLM + C(VehAgeGLM, Treatment(reference=2)) + C(DrivAgeGLM, Treatment(reference=5)) + BonusMalusGLM + VehBrand + VehGas + DensityGLM + C(Region, Treatment(reference='R24')) + AreaGLM"
    glm1 = smf.glm(formula=formula1, data=train, family=sm.families.Poisson(link=sm.families.links.log())).fit()

    formula2 = "Frequency ~ VehPowerGLM + C(VehAgeGLM, Treatment(reference=2)) + C(DrivAgeGLM, Treatment(reference=5)) + BonusMalusGLM + VehBrand + VehGas + DensityGLM + C(Region, Treatment(reference='R24'))"
    glm2 = smf.glm(formula=formula2, data=train, family=sm.families.Poisson(link=sm.families.links.log())).fit()

    formula3 = "Frequency ~ VehPowerGLM + C(VehAgeGLM, Treatment(reference=2)) + C(DrivAgeGLM, Treatment(reference=5)) + BonusMalusGLM + VehGas + DensityGLM + C(Region, Treatment(reference='R24'))"
    glm3 = smf.glm(formula=formula3, data=train, family=sm.families.Poisson(link=sm.families.links.log())).fit()

    obs = test['Frequency']
    pred = glm1.predict(test)
    make_results(obs, pred)

    obs = test['Frequency']
    pred = glm2.predict(test)
    make_results(obs, pred)

    obs = test['Frequency']
    pred = glm3.predict(test)
    make_results(obs, pred)

    def make_results(glm_freq, X, y):
        # Using MAPE error metrics to check for the error rate and accuracy level

        LR_MAPE = MAPE(y, glm_freq.predict(X))
        GINI = gini(y, glm_freq.predict(X))
        poissdev = poisson_deviance(y, glm_freq.predict(X))
        print("Poisson-Loss: ", poissdev)

    df_drop1_glm3 = drop1(formula3, glm3, train)
    return glm_sev

def regression_common_sev(train):
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import itertools
    from sklearn.linear_model import GammaRegressor

    df_freq = freq0.copy(deep=True)
    df_freq = freq0.iloc[freq0.drop(['IDpol', 'Exposure', 'ClaimNb'], axis=1).drop_duplicates().index]
    df_freq = df_freq.reset_index(drop=True)
    df_freq['GroupID'] = df_freq.index + 1
    df_freq = pd.merge(freq0, df_freq, how='left')
    df_freq['GroupID'] = df_freq['GroupID'].fillna(method='ffill')

    for df1 in [train, test]:
        df1['VehAge'] = df1['VehAge'].apply(lambda x: 20 if x > 20 else x)
        df1['DrivAge'] = df1['DrivAge'].apply(lambda x: 90 if x > 90 else x)
        df1['BonusMalus'] = df1['BonusMalus'].apply(lambda x: 150 if x > 150 else int(x))
        df1['VehPowerGLM'] = df1['VehPower'].apply(lambda x: 9 if x > 9 else x)
        df1['VehPowerGLM'] = df1['VehPowerGLM'].apply(lambda x: str(x))
        df1['VehAgeGLM'] = pd.cut(df1['VehAge'], bins=[0, 1, 10, np.inf], labels=[1, 2, 3], include_lowest=True)
        df1['DrivAgeGLM'] = pd.cut(df1['DrivAge'], bins=[18, 21, 26, 31, 41, 51, 71, np.inf],
                                   labels=[1, 2, 3, 4, 5, 6, 7], include_lowest=True)
        df1['BonusMalusGLM'] = df1['BonusMalus']
        df1['DensityGLM'] = df1['Density']

    mask_train = train["ClaimAmount"] > 0
    mask_test = test["ClaimAmount"] > 0

    glm_sev = GammaRegressor()

    glm_sev.fit(
        X_train[mask_train.values],
        df_train.loc[mask_train, "AvgClaimAmount"],
        sample_weight=df_train.loc[mask_train, "ClaimNb"],
    )

    scores = score_estimator(
        glm_sev,
        X_train[mask_train.values],
        X_test[mask_test.values],
        df_train[mask_train],
        df_test[mask_test],
        target="AvgClaimAmount",
        weights="ClaimNb",
    )
    print("Evaluation of GammaRegressor on target AvgClaimAmount")
    print(scores)

    lst_model = []
    lst_params = []
    lst_aic = []
    lst_insampleloss = []
    lst_outsampleloss = []
    lst_avgfreq = []

    def save_result(fittedmodel, model_desc, col_name):
        train.insert(0, col_name, fittedmodel.predict(train, offset=np.log(train['Exposure'])))
        test.insert(0, col_name, fittedmodel.predict(test, offset=np.log(test['Exposure'])))
        lst_model.append(model_desc)
        lst_params.append(len(fittedmodel.params))
        lst_aic.append((2 * (fittedmodel.df_model + 1) - 2 * fittedmodel.llf))
        lst_insampleloss.append(poisson_deviance(train[col_name], train['ClaimNb']))
        lst_outsampleloss.append(poisson_deviance(test[col_name], test['ClaimNb']))
        lst_avgfreq.append(sum(test[col_name]) / sum(test['Exposure']))
        return pd.DataFrame({
            'Model': lst_model,
            'ParameterCount': lst_params,
            'AIC': lst_aic,
            'InSampleLoss': lst_insampleloss,
            'OutSampleLoss': lst_outsampleloss,
            'AvgFrequency': lst_avgfreq
        })

    def drop1(formula, model, data):
        x = [i for i in formula.split('~')[1].split('+')]
        drop1_stats = {}
        for k in range(1, len(x) + 1):
            for variables in itertools.combinations(x, k):
                if len(variables) == len(x) - 1:
                    predictors = list(variables)
                    i = True
                    independent = ''
                    for p in predictors:
                        if i:
                            independent = p
                            i = False
                        else:
                            independent += '+ {}'.format(p)
                    regression = 'ClaimNb ~ {}'.format(independent)
                    print('Dropping <' + [i for i in x if i not in predictors][
                        0] + '> from the model and recalculate statistics ... \n')
                    res = smf.glm(formula=regression, data=data,
                                  family=sm.families.Poisson(link=sm.families.links.log()),
                                  offset=np.log(data['Exposure'])).fit()
                    drop1_stats[[i for i in x if i not in predictors][0]] = (2 * (k + 1) - 2 * res.llf,
                                                                             res.deviance,
                                                                             res.deviance - model.deviance,
                                                                             res.df_model,
                                                                             # 1-stats.chi2.cdf(res.deviance - model.deviance, res.df_model)
                                                                             )
        df_out = pd.DataFrame(drop1_stats).T
        df_out.columns = ['AIC', 'Deviance', 'LRT', 'DoF']
        return df_out

    formula1 = "Frequency ~ VehPowerGLM + C(VehAgeGLM, Treatment(reference=2)) + C(DrivAgeGLM, Treatment(reference=5)) + BonusMalusGLM + VehBrand + VehGas + DensityGLM + C(Region, Treatment(reference='R24')) + AreaGLM"
    glm1 = smf.glm(formula=formula1, data=train, family=sm.families.Poisson(link=sm.families.links.log())).fit()

    formula2 = "Frequency ~ VehPowerGLM + C(VehAgeGLM, Treatment(reference=2)) + C(DrivAgeGLM, Treatment(reference=5)) + BonusMalusGLM + VehBrand + VehGas + DensityGLM + C(Region, Treatment(reference='R24'))"
    glm2 = smf.glm(formula=formula2, data=train, family=sm.families.Poisson(link=sm.families.links.log())).fit()

    formula3 = "Frequency ~ VehPowerGLM + C(VehAgeGLM, Treatment(reference=2)) + C(DrivAgeGLM, Treatment(reference=5)) + BonusMalusGLM + VehGas + DensityGLM + C(Region, Treatment(reference='R24'))"
    glm3 = smf.glm(formula=formula3, data=train, family=sm.families.Poisson(link=sm.families.links.log())).fit()

    obs = test['Frequency']
    pred = glm1.predict(test)
    make_results(obs, pred)

    obs = test['Frequency']
    pred = glm2.predict(test)
    make_results(obs, pred)

    obs = test['Frequency']
    pred = glm3.predict(test)
    make_results(obs, pred)

    def make_results(glm_freq, X, y):
        # Using MAPE error metrics to check for the error rate and accuracy level

        LR_MAPE = MAPE(y, glm_freq.predict(X))
        GINI = gini(y, glm_freq.predict(X))
        poissdev = poisson_deviance(y, glm_freq.predict(X))
        print("Poisson-Loss: ", poissdev)

    df_drop1_glm3 = drop1(formula3, glm3, train)

    return glm_sev
