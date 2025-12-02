def lecture():

    """
    Statsmodels OLS
    """

    import statsmodels.formula.api as smf

    import pandas as pd, numpy as np

    data = pd.read_csv("https://github.com/dustywhite7/Econ8320/blob/master/AssignmentData/assignment8Data.csv?raw=true")

    reg = smf.ols("np.log(I(hrwage+1)) ~ age + I(age**2) + C(educ)", data= data).fit()
    # C(var) makes var categorical

    reg = reg.get_robustcov_results(cov_type='HC0')
    #robust standard error (White 1980)

    reg.summary()

    """
    Other Statsmodels Modeling Options
    """

    model = smf.logit("married ~ nchlt5 + C(educ)", data = data)
    #logistic model, not linear

    model = smf.poisson("nchlt5 ~ hrwage", data = data[data['nchlt5']>0])

    modelFit = model.fit()

    modelFit.summary()

    """
    Using the Patsy Library
    """

    #patsy allows:
        #seperation of exo and endo genous vars
        # dummy out categorical vars
        # easily transform vars mathematically
        # use identical transformations on future data

    data1 = data.sample(1000)

    data2 = data.sample(1000)

    import patsy as pt

    y, x = pt.dmatrices("hhincome ~ married + nchlt5 + educ", data = data1)
    #design matrices

    y
    # dependent var matrix
    
    x
    # independent vars matrix
    # has extra column (intercept column)

    np.linalg.inv(x.T @ x) @ x.T @ y
    #linear algebra, inverse, Transpose, but idr the meanings
    # this matrix gives the beta coef for OLS, just showing how it is done bts
        #bts statsmodels uses patsy
    # doing it this way, with x and y allows us to take the pre-built xs and ys to other models

    import statsmodels.api as sm

    model = sm.OLS(y, x).fit()

    model.summary()

    x2 = pt.build_design_matrices([x.design_info], data2)
    # same structre of x is applied to x2 even though it is different data

    model.predict(x2)

    """
    introducing scikit-learn
    """

    #sk-learn is best for ml in python

    data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/DataSets/occupancyTrain.csv")

    y, x = pt.dmatrices("Occupancy ~ -1 + Light +CO2", data = data)
    # -1 in patsy means no intercept column bc most ml models expect or like intercept models

    from sklearn import tree
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    x1, x2, y1, y2 = train_test_split(x, y)
    # creates training x and y and testing x and y from og x and y
    
    clf = tree.DecisionTreeClassifier().fit(x1, y1)
    #clf is classification tree
    # fit model based on training x and y

    pred = clf.predict(x2)
    #predict based on x2

    accuracy_score(y2, pred)
    # accuracy score is testing if model is accurate against y2
        # in this case the percentage of correct observations that the room is occupied

    from sklearn import svm
    #support vector model

    clf = svm.SVC().fit(x1, y1)
    # support vector classifier

    pred = clf.predict(x2)

    accuracy_score(y2, pred)

    return

    

lecture()

def linearRegression():

    import statsmodels.formula.api as smf

    import pandas as pd  

    data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/pythonMikkeli/master/exampleData/wagePanelData.csv")

    model = smf.ols("log_wage ~ years_experience + weeks_worked + C(education) + south_region + metropolitan_resident+ms+C(female)+union_member+C(is_black)", data = data)

    reg = model.fit()

    reg.summary()

    return

linearRegression()

def logisticRegression():

    import statsmodels.formula.api as smf

    import pandas as pd

    data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/pythonMikkeli/master/exampleData/passFailTrain.csv")

    model = smf.logit("G3 ~ school+sex+age+address+famsize+Pstatus+Medu+Fedu+Mjob+Fjob+reason+guardian+traveltime+studytime+failures+schoolsup+famsup+paid+activities+nursery+higher+internet+romantic+freetime+health+absences", data = data)

    reg = model.fit()
    
    reg.summary()

    return

logisticRegression()

def randomForest():

    import statsmodels.formula.api as smf

    import pandas as pd

    from sklearn import ensemble
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    import patsy as pt

    data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/Econ8320/master/AssignmentData/assignment12Data.csv")

    y, x = pt.dmatrices("Playoffs ~ -1 + Revenues + TVDeal + LaborContract + OperatingIncome+DebtToValue+Value", data = data)

    y

    x1, x2, y1, y2 = train_test_split(x, y)
    
    playoffForest = ensemble.RandomForestClassifier(n_estimators = 100).fit(x1, y1.ravel())
    #ravel() flattens the 2d matrix to a 1d matrix

    y2

    pred = playoffForest.predict(x2)
    #predict based on x2

    accuracy_score(y2, pred)

    return