import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump, load


class Model(object):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=150, n_jobs=-1)
        self.model_name = 'Random Forest classifier'
        self.classification_report = {}

    def load_model(self, path_to_model: str) -> None:
        '''
        Loads the model from file
        :param path_to_model: path to the file containing the model
        '''
        try:
            self.model = load(path_to_model)
        except FileNotFoundError:
            print('Model not found!')

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        '''
        Trains the model
        :param X_train, y_train: features and target labels already transformed and scaled
        '''
        self.model.fit(X_train, y_train)

    def get_scores(self, X, y) -> float:
        '''
        Returns the models accuracy, precision, recall and f1-scores
        :param X,y: the features and target labels against which to calculate the metrics/scores
        :return : the scores
        '''
        self.classification_report = classification_report(y, self.model.predict(X), output_dict=True)
        accuracy = self.classification_report['accuracy']
        precision = self.classification_report['weighted avg']['precision']
        recall = self.classification_report['weighted avg']['recall']
        f1score = self.classification_report['weighted avg']['f1-score']
        return accuracy, precision, recall, f1score

    def save_model(self, model_name = 'rf_test.joblib'):
        dump(self.model, f'models/{model_name}')