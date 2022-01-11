import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def transform(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function takes a dataframe and returns a transformed one, i.e. ready to feed to a ML algorithm
    :param df: the pandas dataframe
    :return : the transformed pandas dataframe
    '''
    # Labels the property type by its initial letter
    df['property_type'] = df['property_type'].map(lambda x: x[0])

    # One hot encode property types and drop redundant property type category
    df = pd.concat([df, pd.get_dummies(df['property_type'], drop_first=True)], axis=1)
    df.drop(labels=['property_type'], axis=1, inplace=True)

    return df


def split_and_scale(df: pd.DataFrame, random_state: int = 17) -> tuple[pd.DataFrame]:
    '''
    This method takes the transformed dataframe as input and returns a train-test split. The method
    also performs a MinMaxScaling
    :param df: the transformed pandas dataframe
    :return: X_train, y_train, X_test, y_test
    '''
    X = df.drop(labels='bedrooms', axis=1)
    y = df['bedrooms']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    scl = MinMaxScaler()
    X_train[:] = scl.fit_transform(X_train)
    X_test[:] = scl.transform(X_test)
    return X_train, y_train, X_test, y_test
