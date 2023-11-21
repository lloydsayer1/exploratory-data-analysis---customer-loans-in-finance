import yaml
from sqlalchemy import create_engine, inspect
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import normaltest
import matplotlib
import statsmodels
import numpy as np
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import seaborn as sns
import plotly
from plotly import express

class RDSDatabaseConnector:
    def __init__(self, yamlCredentials):
        self.yamlCredentials = yamlCredentials

    def credentialsLoader(self):
        with open(self.yamlCredentials, 'r') as file:
            yamlPass = yaml.safe_load(file)
        return yamlPass

    def connector(self):
        yamlPass = self.credentialsLoader()
        engine = create_engine(f"postgresql+psycopg2://{yamlPass['RDS_USER']}:{yamlPass['RDS_PASSWORD']}@{yamlPass['RDS_HOST']}:{yamlPass['RDS_PORT']}/{yamlPass['RDS_DATABASE']}")
        engine.execution_options(isolation_level='AUTOCOMMIT').connect()
        engine.connect()
        return engine

    def createDataframe(self):
        engine = self.connector()
        loan_payments = pd.read_sql_table('loan_payments', engine)
        return loan_payments

    def saveData(self):
        loan_payments = self.createDataframe()
        loan_payments.to_csv('loanpayments.csv')    

    def csvToDataframe(self):
        df = pd.read_csv('loanpayments.csv')
        print(df.head())
        print(df.describe())
        print(df.dtypes)
        pd.set_option('display.max_columns', None)
        return df

class DataTransform:

    def __init__(self, df):
        self.df = df

    
    def termToCat(self):
        dfUse = self.df
        dfUse['term'] = dfUse['term'].str.replace(' months', '')
        dfUse.term = dfUse.term.astype('category', errors = 'ignore')
        return dfUse
    
    def gradeToCat(self):
        dfUse = self.termToCat()
        dfUse.grade = dfUse.grade.astype('category', errors = 'ignore') 
        return dfUse

    def subgradeToCat(self):
        dfUse = self.gradeToCat()
        dfUse.sub_grade = dfUse.sub_grade.astype('category', errors = 'ignore')
        return dfUse
    
    def employlengthToInt(self):
        dfUse = self.subgradeToCat()
        dfUse['employment_length'] = dfUse['employment_length'].str.replace(' years', '')
        dfUse['employment_length'] = dfUse['employment_length'].str.replace(' year', '')
        dfUse['employment_length'] = dfUse['employment_length'].str.replace('+', '')
        dfUse['employment_length'] = dfUse['employment_length'].str.replace('< 1', '0')
        dfUse.employment_length = pd.to_numeric(dfUse['employment_length'])
        return dfUse 

    def homeownerToCat(self):
        dfUse = self.employlengthToInt()
        dfUse.home_ownership = dfUse.home_ownership.astype('category', errors = 'ignore')
        return dfUse
    
    def verificationToCat(self):
        dfUse = self.homeownerToCat()
        dfUse.verification_status = dfUse.verification_status.astype('category', errors = 'ignore')
        return dfUse
    
    def loanstatusToCat(self):
        dfUse = self.verificationToCat()
        dfUse.loan_status = dfUse.loan_status.astype('category', errors = 'ignore')
        return dfUse

    def purposeToCat(self):
        dfUse = self.loanstatusToCat()
        dfUse.purpose = dfUse.purpose.astype('category', errors = 'ignore')
        return dfUse
    
    def lastpayToDate(self):
        dfUse = self.purposeToCat()
        dfUse['last_payment_date'] = pd.to_datetime(dfUse['last_payment_date'])
        return dfUse
    
    def nextpayToDate(self):
        dfUse = self.lastpayToDate()
        dfUse['next_payment_date'] = pd.to_datetime(dfUse['next_payment_date'])
        return dfUse
    
    def lastcreditToDate(self):
        dfUse = self.nextpayToDate()
        dfUse['last_credit_pull_date'] = pd.to_datetime(dfUse['last_credit_pull_date'])
        return dfUse

    def issuedateToDateTime(self):
        dfUse = self.lastcreditToDate()
        dfUse['issue_date'] = pd.to_datetime(dfUse['issue_date'])
        return dfUse

    def earliestCredit(self):
        dfUse = self.issuedateToDateTime()
        dfUse['earliest_credit_line'] = pd.to_datetime(dfUse['earliest_credit_line'])
        return dfUse
    
class DataFrameInfo:

    def __init__(self, df):
        self.df = df

    def description(self):
        dfUse = self.df
        dfDescription = dfUse.describe()
        return dfDescription

    def dataTypes(self):
        dfUse = self.df
        dfDtypes = dfUse.dtypes
        return dfDtypes

    def uniqueValues(self):
        dfUse = self.df
        dfCat = dfUse[['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status', 'loan_status', 'payment_plan', 'purpose', 'application_type']].copy()
        for col in dfCat:
            print(dfCat[(col)].value_counts())
    
    def dfShapeOut(self):
        dfUse = self.df
        dfShape = dfUse.shape
        return dfShape
    
    def nullPercentage(self):
        dfUse = self.df
        nullCalc = (dfUse.isnull().sum()/len(dfUse))*100
        return nullCalc
    
class Plotter:

    def __init__(self, df, inputColumn):
        self.df = df
        self.inputColumn = inputColumn
    
    def qqPlotter(self):
        dfUse = self.df
        col = self.inputColumn
        qq_plot = qqplot(dfUse[col] , scale=1 ,line='q', fit=True)
        return pyplot.show()
    
    def boxPlotter(self):
        dfUse = self.df
        dfUse = dfUse.drop(['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status', 'loan_status', 'purpose', 'application_type', 'issue_date', 'earliest_credit_line', 'last_payment_date', 'last_credit_pull_date', 'payment_plan', 'next_payment_date'], axis = 1)
        for column in dfUse:
            pyplot.figure()
            dfUse.boxplot([column])

    def heatmapPlotter(self):
        dfUse = self.df
        dfHeatmap = dfUse[['loan_amount', 'funded_amount', 'funded_amount_inv', 'int_rate', 'instalment', 'employment_length', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_accounts', 'total_accounts', 'out_prncp', 'out_prncp_inv', 'total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_payment_amount', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'policy_code']]
        x = sns.heatmap(dfHeatmap.corr())
        return x
        

    




class DataFrameTransform:

    def __init__(self, df):
        self.df = df

    def dropData(self):
        dfUse = self.df
        dfUse = dfUse.drop(['mths_since_last_delinq', 'mths_since_last_record', 'next_payment_date', 'mths_since_last_major_derog'], axis = 1)
        dfUse = dfUse.dropna(subset = ['last_payment_date', 'last_credit_pull_date', 'collections_12_mths_ex_med'])
        return dfUse

    def nullHandler(self):
        dfUse = self.dropData()
        dfUse['funded_amount'] = dfUse['funded_amount'].fillna(dfUse['loan_amount'])
        dfUse['term'] = dfUse['term'].fillna(dfUse['term'].mode())
        dfUse['int_rate'] = dfUse['int_rate'].fillna(df['int_rate'].median())
        dfUse['employment_length'] = dfUse['employment_length'].fillna(df['employment_length'].median())
        return dfUse
    
    def skewOutput(self):
        dfUse = self.df
        dfNew = dfUse.drop(['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status', 'loan_status', 'purpose', 'application_type', 'issue_date', 'earliest_credit_line', 'last_payment_date', 'last_credit_pull_date', 'payment_plan', 'next_payment_date'], axis = 1)
        dfNew = dfNew.skew()
        return dfNew
    
    def skewFixer(self):
        dfUse = self.df
        loan_amount_boxcox = dfUse["loan_amount"]
        loan_amount_boxcox = stats.boxcox(loan_amount_boxcox)
        loan_amount_boxcox = pd.Series(loan_amount_boxcox[0])
        t=sns.histplot(loan_amount_boxcox,label="Skewness: %.2f"%(loan_amount_boxcox.skew()) )
        qq_plot = qqplot(loan_amount_boxcox , scale=1 ,line='q', fit=True)
        t.legend()
        dfUse['loan_amount'] = loan_amount_boxcox

        annual_inc_boxcox = dfUse["annual_inc"]
        annual_inc_boxcox = stats.boxcox(annual_inc_boxcox)
        annual_inc_boxcox = pd.Series(annual_inc_boxcox[0])
        t=sns.histplot(annual_inc_boxcox,label="Skewness: %.2f"%(annual_inc_boxcox.skew()) )
        qq_plot = qqplot(annual_inc_boxcox , scale=1 ,line='q', fit=True)
        t.legend()
        dfUse['annual_inc'] = annual_inc_boxcox

        delinq_2yrs_yeojohnson = dfUse["delinq_2yrs"]
        delinq_2yrs_yeojohnson = stats.yeojohnson(delinq_2yrs_yeojohnson)
        delinq_2yrs_yeojohnson = pd.Series(delinq_2yrs_yeojohnson[0])
        t=sns.histplot(delinq_2yrs_yeojohnson,label="Skewness: %.2f"%(delinq_2yrs_yeojohnson.skew()) )
        qq_plot = qqplot(delinq_2yrs_yeojohnson , scale=1 ,line='q', fit=True)
        t.legend()
        dfUse['delinq_2yrs'] = delinq_2yrs_yeojohnson

        return dfUse
    
    def outlierRemover(self):
        dfUse = self.df
        dfOutliers = ['loan_amount', 'funded_amount', 'funded_amount_inv', 'int_rate', 'instalment', 'employment_length', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_accounts', 'total_accounts', 'out_prncp', 'out_prncp_inv', 'total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_payment_amount', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'policy_code']
        for col in dfOutliers:
            Q1 = dfUse[col].quantile(0.25)
            Q3 = dfUse[col].quantile(0.75)
            IQR = stats.iqr(dfUse[col])
            threshold = 1.5
            outliers = dfUse[(dfUse[col] < Q1 - threshold * IQR) | (dfUse[col] > Q3 + threshold * IQR)]
            dfUse = dfUse.drop(outliers.index)
        dfUse.shape
        return dfUse
    
    def correlationRemover(self):
        dfUse = self.df
        dfUse = dfUse.drop(['funded_amount_inv', 'total_payment_inv', 'total_rec_prncp'], axis = 1)
        return dfUse

            





       
loadCSV = RDSDatabaseConnector('credentials.yml')
df = loadCSV.csvToDataframe()

transformData = DataTransform(df)
df = transformData.earliestCredit()

dataframetransformation = DataFrameTransform(df)
df = dataframetransformation.nullHandler()
dfnew = dataframetransformation.skewOutput()
print(dfnew)
df = dataframetransformation.skewFixer()

qqplotshow = Plotter(df, 'annual_inc')
qqplotshow.qqPlotter()

df = dataframetransformation.outlierRemover()
df = dataframetransformation.correlationRemover()
df.shape


qqplotshow.heatmapPlotter()















#dfNew = df.drop(['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status', 'loan_status', 'purpose', 'application_type', 'issue_date', 'earliest_credit_line', 'last_payment_date', 'last_credit_pull_date', 'payment_plan'], axis = 1)

