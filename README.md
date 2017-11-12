# Supervised Prediction Model for Bank Loan Provision

This project is an example of a data science task workflow. Specifically, it makes use of many Python libraries which you can find below and the three supervised classification models of Logistic Regression, Decision Trees and Random Forests. The question asked is the following:

"Use the Machine Learning Workflow to process and transform the Mortgage Loan dataset to create a prediction model. This model must predict which people are likely to be approved for a mortage loan with 75% or greater accuracy."

The project was a good learning exercise for me, and hopefully, is a good reference for you.

## Dataset Used

[Loan Prediction](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction/) : This data set corresponds to a set of anonymized financial transactions associated with loan provision and individual data. There are nearly 1000 observations and 12 features. Each observation is independent from the previous.

## Learning Material:

### Explore and describe the dataset
- [Aggregating Data - Pivot Tables](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html)
- [Histograms - Distribution of Numerical Values](https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.hist.html)
- [Boxplots - Distribution of Numerical values for each category of categorical variables](https://seaborn.pydata.org/generated/seaborn.boxplot.html)
- [Cross Tabulation - Frequency Distribution of Categorical Variables](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html)
- [Figure with two bar subplots - Correlation of Variables Distribution to make an Hypothesis](https://matplotlib.org/examples/pylab_examples/subplots_demo.html)

### Transform and cleanse the dataset
- [Binarize Categorical Values](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
- [Impute Missing Values](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html)

### Select accuracy metrics
- [K fold](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- [Classification Accuracy score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

### Build a supervised a model
- [Random Forest Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Decision Trees](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

- [Feature Selection](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Learn Data Science with Python](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/) 
- [Purpose of the loan & applicant income: important data to be embeded in the model and reduce bias](https://www.technologyreview.com/s/609338/new-research-aims-to-solve-the-problem-of-ai-bias-in-black-box-algorithms/)

## Requirements
To run the project, it is required that the following are installed in your system:

- [anaconda](https://docs.continuum.io/anaconda/navigator)
- [Pyhton](https://www.python.org/download/releases/2.7/) version: "^2.7"
- [NumPy](http://www.numpy.org/) version: "^1.11.3"
- [Matplotlib](https://matplotlib.org/) version: "^2.02"
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/install.html) version: "^0.20.1"
- [scikit-learn](http://scikit-learn.org/stable/install.html) version: "^0.18.1"

If anaconda is installed, you do not have to install python and the packages since they are already included in anaconda packages.
