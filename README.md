# Street Group Data Scientist application exercise

The goal of this exercise is to develop a model that predicts the number of
bedrooms in a property given a set of features. According to the guidelines provided 
the time spent to write all this code was around 3 hours.

## How to run

Ensure all required libraries are installed. For this refer to 'requirements.txt'.

```shell
python main.py
```

The exploratory data analysis and model selection trainings can be run on the jupyter
notebook after installing all libraries on 'requirements_nb.txt'.

In order for the code to run, ensure the data file '*street_group_data_science_bedrooms_test.json*' is 
inside the repositories  *data/* folder. This data file has not been uploaded to the repo due to its large size.

## Conclusion

The data was explored and different models were trained on it, namely:

* Random Forest: overall the best performance and fastest one to train
* Naive Bayes Multinomial: poor performance
* K-Nearest-Neighbours: performance
* XGBoost: similar performance to RF but significantly longer training time
* DNN: several NN architectures were attempted. Similar performance to RF but significantly longer training time.

Given more data and more features the DNN would have probably outperformed the other models. The RF scores were:
* Accuracy = 0.7575 
* Precision = 0.7533 
* Recall = 0.7575 
* F1-score = 0.7534

There is significantly more data for properties with 1 to 5 bedrooms. As a result, the predictions for these properties
achieve much larger scores than for the properties with 0 bedrooms or more than 5 bedrooms. Provided we had more data,
more features and longer training time, larger scores would be achievable.
