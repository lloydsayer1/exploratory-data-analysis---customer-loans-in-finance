import yaml

class RDSDatabaseConnector:
    def __init__(self, yamlCredentials):
        self.yamlCredentials = yamlCredentials
    
    def credentialsLoader():
        with open('credentials.yaml', 'r') as file:
            yamlCredentials = yaml.safe_load(file)

    