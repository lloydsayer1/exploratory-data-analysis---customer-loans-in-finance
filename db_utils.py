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

class DataTransform: #transforms all the data into correct format

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
    
class DataFrameInfo: # all methods for various pieces of info on the dataframe

    def __init__(self, df):
        self.df = df

    def description(self): # description of the dataframe
        dfUse = self.df
        dfDescription = dfUse.describe()
        return dfDescription

    def dataTypes(self): # datatypes of the dataframe
        dfUse = self.df
        dfDtypes = dfUse.dtypes
        return dfDtypes

    def uniqueValues(self): # unique values of the different columns in the dataframe
        dfUse = self.df
        dfCat = dfUse[['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status', 'loan_status', 'payment_plan', 'purpose', 'application_type']].copy()
        for col in dfCat:
            print(dfCat[(col)].value_counts())
    
    def dfShapeOut(self): # shape of the dataframe
        dfUse = self.df
        dfShape = dfUse.shape
        return dfShape
    
    def nullPercentage(self): # NaN values in the dataframe
        dfUse = self.df
        nullCalc = (dfUse.isnull().sum()/len(dfUse))*100
        return nullCalc
    
class Plotter:

    def __init__(self, df, inputColumn):
        self.df = df
        self.inputColumn = inputColumn
    
    def qqPlotter(self):  # makes a qqplot of the inputted columns in the dataframe
        dfUse = self.df
        col = self.inputColumn
        qq_plot = qqplot(dfUse[col] , scale=1 ,line='q', fit=True)
        return pyplot.show()
    
    def boxPlotter(self): # creates a boxplot for all the relevant columns in the dataframe
        dfUse = self.df
        dfUse = dfUse.drop(['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status', 'loan_status', 'purpose', 'application_type', 'issue_date', 'earliest_credit_line', 'last_payment_date', 'last_credit_pull_date', 'payment_plan', 'next_payment_date'], axis = 1)
        for column in dfUse:
            pyplot.figure()
            dfUse.boxplot([column])

    def heatmapPlotter(self): # makes the correlation heatmap for all of the relevant dataframe columns
        dfUse = self.df
        dfHeatmap = dfUse[['loan_amount', 'funded_amount', 'funded_amount_inv', 'int_rate', 'instalment', 'employment_length', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_accounts', 'total_accounts', 'out_prncp', 'out_prncp_inv', 'total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_payment_amount', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'policy_code']]
        x = sns.heatmap(dfHeatmap.corr())
        return x
        

class DataFrameTransform:

    def __init__(self, df):
        self.df = df

    def dropData(self):  # drops data not needed in the dataframe
        dfUse = self.df
        dfUse = dfUse.drop(['mths_since_last_delinq', 'mths_since_last_record', 'next_payment_date', 'mths_since_last_major_derog'], axis = 1)
        dfUse = dfUse.dropna(subset = ['last_payment_date', 'last_credit_pull_date', 'collections_12_mths_ex_med'])
        return dfUse

    def nullHandler(self):  # removes nulls from relevant columns in the dataframe
        dfUse = self.dropData()
        dfUse['funded_amount'] = dfUse['funded_amount'].fillna(dfUse['loan_amount'])
        dfUse['term'] = dfUse['term'].fillna(dfUse['term'].mode())
        dfUse['int_rate'] = dfUse['int_rate'].fillna(df['int_rate'].median())
        dfUse['employment_length'] = dfUse['employment_length'].fillna(df['employment_length'].median())
        return dfUse
    
    def skewOutput(self):  # shows the skew of relevant columns in the dataframe
        dfUse = self.df
        dfNew = dfUse.drop(['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status', 'loan_status', 'purpose', 'application_type', 'issue_date', 'earliest_credit_line', 'last_payment_date', 'last_credit_pull_date', 'payment_plan', 'next_payment_date'], axis = 1)
        dfNew = dfNew.skew()
        return dfNew
    
    def skewFixer(self): # fixes the skew for columns decided upon
        dfUse = self.df

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
    
    def outlierRemover(self): # removes outliers from columns decided upon
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
    
    def correlationRemover(self): # I felt that I ended up needing to use the columns that I initially dropped for being too correlated
        dfUse = self.df
        dfUse = dfUse.drop([], axis = 1)
        return dfUse
       
loadCSV = RDSDatabaseConnector('credentials.yml')
df = loadCSV.csvToDataframe()

transformData = DataTransform(df)
df = transformData.earliestCredit()

dataframetransformation = DataFrameTransform(df)
df = dataframetransformation.nullHandler()
df = dataframetransformation.skewFixer()
df = dataframetransformation.outlierRemover()
df = dataframetransformation.correlationRemover()
df.shape





def milestone4task1(df):
    dfBar = df[['total_rec_prncp', 'funded_amount', 'funded_amount_inv', 'last_payment_amount', 'term', 'issue_date', 'last_payment_date']]
    dfBar = dfBar.dropna(subset = 'term', axis = 0)
    dfBar['term'] = dfBar['term'].astype('int64')
    dfBar['months_paid'] = (dfBar['last_payment_date'].dt.year - dfBar['issue_date'].dt.year) *12 + (dfBar['last_payment_date'].dt.month - dfBar['issue_date'].dt.month)
    dfBar['remaining_months'] = dfBar['term'] - dfBar['months_paid']
    percentRepaid = round((dfBar['total_rec_prncp'].sum())/(dfBar['funded_amount'].sum())*100, 2)
    print(f"{percentRepaid}% of the funded amount has been repaid")
    percentRepaidInv = round((dfBar['total_rec_prncp'].sum())/(dfBar['funded_amount_inv'].sum())*100, 2)
    print(f"{percentRepaidInv}% of the amount funded by investors has been repaid")
    dfBar['remaining_months'].mask(dfBar['remaining_months'] < 0, 0, inplace = True)
    dfBar['remaining_months'].mask(dfBar['remaining_months'] > 6, 6, inplace = True)
    dfBar['projection'] = dfBar['last_payment_amount'] * dfBar['remaining_months']
    projected_amount = dfBar['projection'].sum()
    current_amount = dfBar['total_rec_prncp'].sum()
    funded_amount = dfBar['funded_amount'].sum()
    funded_inv_amount = dfBar['funded_amount_inv'].sum()
    pyplot.clf()
    x = np.array(["Current amount paid back", "Amount funded", "Amount funded by investors", "Paid back in next 6 months"])
    y = np.array([current_amount, funded_amount, funded_inv_amount, projected_amount])
    sns.barplot(x = x, y = y)
    return


def milestone4task2(df):
    dfCharged = df[['total_rec_prncp', 'loan_status', 'funded_amount']]
    chargedOffSum = dfCharged.loc[dfCharged['loan_status'] == "Charged Off", 'total_rec_prncp'].sum()
    print(f"The amount of money paid towards these loans was £{chargedOffSum}")
    percentageSum = round((5571/54231)*100, 2)
    print(f"The percentage of charged off loans is {percentageSum}%")
    return 

def milestone4task3(df):
    dfCharged = df[['total_rec_prncp', 'loan_status', 'funded_amount', 'last_payment_amount', 'term', 'issue_date', 'last_payment_date', 'int_rate']]
    dfCharged = dfCharged.dropna()
    dfCharged['term'] = dfCharged['term'].astype(np.float64)
    dfCharged['months_paid'] = (dfCharged['last_payment_date'].dt.year - dfCharged['issue_date'].dt.year) *12 + (dfCharged['last_payment_date'].dt.month - dfCharged['issue_date'].dt.month)
    dfCharged['remaining_months'] = dfCharged['term'] - dfCharged['months_paid']
    chargedOffProjectionDf = dfCharged[dfCharged.loan_status == 'Charged Off']
    chargedOffProjectionDf['lostRevenue'] = chargedOffProjectionDf['last_payment_amount']*pow(1+(chargedOffProjectionDf['int_rate']/100), (chargedOffProjectionDf['remaining_months']/12))
    amount_lost_interest = round(chargedOffProjectionDf['lostRevenue'].sum(), 2)
    print(f"The amount of revenue lost through interest on charged off loans is £{amount_lost_interest}")
    chargedOffProjectionDf['charged_off_unpaid_loss'] = chargedOffProjectionDf['last_payment_amount']*chargedOffProjectionDf['remaining_months']
    amount_lost_total = round(chargedOffProjectionDf['charged_off_unpaid_loss'].sum() + amount_lost_interest, 2)
    print(f"Amount of revenue lost in total from charged off loans including interest revenue is £{amount_lost_total}")
    return amount_lost_total

def milestone4task4(df, amount_lost_total):
    lateTotal = round(((106+580)/54231)*100, 2)
    print(f"The percentage of people late on payments is {lateTotal}%")
    dfCharged = df[['total_rec_prncp', 'loan_status', 'funded_amount', 'last_payment_amount', 'term', 'issue_date', 'last_payment_date', 'int_rate']]
    dfCharged = dfCharged.dropna()
    dfCharged['term'] = dfCharged['term'].astype(np.float64)
    dfCharged['months_paid'] = (dfCharged['last_payment_date'].dt.year - dfCharged['issue_date'].dt.year) *12 + (dfCharged['last_payment_date'].dt.month - dfCharged['issue_date'].dt.month)
    dfCharged['remaining_months'] = dfCharged['term'] - dfCharged['months_paid']
    late_a = dfCharged["loan_status"] == "Late (16-30 days)"
    late_b = dfCharged["loan_status"] == "Late (31-120 days)"
    any_late = (late_a | late_b)
    late_df = dfCharged.loc[any_late, :]
    late_df['amount_lost'] = late_df['last_payment_amount']*late_df['remaining_months']
    amount_lost = round(late_df['amount_lost'].sum(), 2)
    print(f"If all of these customers were to be changed to 'Charged Off' then the loss of potential revenue would be £{amount_lost}")
    amount_lost_total = round(amount_lost_total + amount_lost, 2)
    print(f"Including the already Charged Off customers, the total loss from this group of customers would be £{amount_lost_total}")
    return late_df

def milestone5task5(late_df, df):
    late_a = df["loan_status"] == "Late (16-30 days)"
    late_b = df["loan_status"] == "Late (31-120 days)"
    late_c = df["loan_status"] == "Charged Off"
    any_late = (late_a | late_b | late_c)
    late_df = df.loc[any_late, :]
    late_df = late_df[['loan_status', 'grade', 'payment_plan', 'dti', 'purpose', 'int_rate']]
    fig, ax = pyplot.subplots()
    ax.bar(late_df['loan_status'], late_df['grade'].value_counts())
    ax.set_ylabel('Grade')
    ax.set_title('Loan Status')

    pyplot.show()
    
    return

milestone4task1(df)

milestone4task2(df)

amount_lost_total = milestone4task3(df)

late_df = milestone4task4(df, amount_lost_total)


milestone5task5(late_df, df)