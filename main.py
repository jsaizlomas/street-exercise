import pandas as pd
from lib.transformer import transform, split_and_scale
from lib.model import Model
import warnings

if __name__ == "__main__":
    warnings.filterwarnings('always')

    print('Loading data...')
    data = pd.read_json('data/street_group_data_science_bedrooms_test.json', lines=True)
    print('Transforming data...')
    data = transform(data)
    X_train, y_train, X_test, y_test = split_and_scale(data)

    model = Model()
    print('Training model...')
    model.train(X_train, y_train)
    #model.save_model()
    print('Getting scores...')
    a, p, r, f1 = model.get_scores(X_test, y_test)
    print("\n Accuracy = {:2.4f} \n Precision = {:2.4f} \n Recall = {:2.4f} \n F1-score = {:2.4f}".format(a, p, r, f1))
