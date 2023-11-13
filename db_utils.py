import yaml
from sqlalchemy import create_engine, inspect
import pandas as pd
import os

class RDSDatabaseConnector:
    def __init__(self, yamlCredentials):
        self.yamlCredentials = yamlCredentials

    def credentialsLoader(self):
        with open(self.yamlCredentials, 'r') as file:
            yamlCredentials = yaml.safe_load(file)
        return yamlCredentials

    def connector(self):
        self.credentialsLoader()
        engine = create_engine(f"postgresql+psycopg2://{self.yamlCredentials['RDS_USER']}:{self.yamlCredentials['RDS_PASSWORD']}@{self.yamlCredentials['RDS_HOST']}:{self.yamlCredentials['RDS_PORT']}/{self.yamlCredentials['RDS_DATABASE']}")
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

p1 = RDSDatabaseConnector('credentials.yml')
p1.saveData()
        










#with open('credentials.yml', 'r') as file:
#  yamlCredentials = yaml.safe_load(file)
#engine = create_engine(f"postgresql+psycopg2://{yamlCredentials['RDS_USER']}:{yamlCredentials['RDS_PASSWORD']}@{yamlCredentials['RDS_HOST']}:{yamlCredentials['RDS_PORT']}/{yamlCredentials['RDS_DATABASE']}")
#engine.execution_options(isolation_level='AUTOCOMMIT').connect()
#engine.connect()
#inspector = inspect(engine)
#print(inspector.get_table_names())
#loan_payments = pd.read_sql_table('loan_payments', engine)
#loan_payments.head(10)

#print(yamlCredentials['RDS_HOST'])