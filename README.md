<h1 align="center"> Diabetes Prediction With ANN </h1>

<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_horizontal.png">
</div>

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4724125.svg)](https://doi.org/10.5281/zenodo.4724125)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1486/badge)](https://bestpractices.coreinfrastructure.org/projects/1486)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/tensorflow/tensorflow/badge)](https://securityscorecards.dev/viewer/?uri=github.com/tensorflow/tensorflow)
[![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow)
[![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow-py.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow-py)
[![OSSRank](https://shields.io/endpoint?url=https://ossrank.com/shield/44)](https://ossrank.com/p/44)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)
[![TF Official Continuous](https://tensorflow.github.io/build/TF%20Official%20Continuous.svg)](https://tensorflow.github.io/build#TF%20Official%20Continuous)
[![TF Official Nightly](https://tensorflow.github.io/build/TF%20Official%20Nightly.svg)](https://tensorflow.github.io/build#TF%20Official%20Nightly)

## Description
<p>
  Diabetes Prediction Model using Artificial Neural Networks (ANN) and <a href="https://www.tensorflow.org/ target="_blank"> Tensorflow</a> as part of the final project in Artificial Intelligence course. 
</p>

## Installation
```bash
  !pip install tensorflow==2.15.0
  !pip install ydata-profiling
```

## Import Libraries
```bash
  # Base Libraries
  import pandas as pd
  import numpy as np
  from ydata_profiling import ProfileReport
  import datetime
  import pickle
  
  # Visualization
  import seaborn as sns
  import matplotlib.pyplot as plt
  from tensorflow.keras.utils import plot_model
  
  # Preprocessing
  from sklearn.preprocessing import StandardScaler
  from sklearn.model_selection import train_test_split
  
  # Find Best Model
  from sklearn.model_selection import GridSearchCV
  from sklearn.preprocessing import OneHotEncoder
  
  # Algorithms
  import tensorflow as tf
  from xgboost import XGBClassifier
  
  from google.colab import files
```

```bash
  print(tf.__version__)
```

## Import Dataset
```bash
  df = pd.read_csv('diabetes.csv')
  df
```

## EDA ( Exploratory Data Analysis ) On Dataset
```bash
  cm = sns.light_palette("green", as_cmap=True)
  info = pd.DataFrame({"name": df.columns,
                       "non-nulls": len(df) - df.isnull().sum().values,
                       "nulls": df.isnull().sum().values,
                       "type": df.dtypes.values})
  
  info.style.background_gradient(cmap=cm)
```

```bash
  pd.DataFrame(df.dtypes.value_counts()).style.background_gradient(cmap=cm)
```
```bash
  pd.DataFrame(df.nunique()).style.background_gradient(cmap=cm)
```
```bash
  pd.DataFrame(df.describe().T).style.background_gradient(cmap=cm)
```

```bash
  Q1 = df.quantile(0.25)
  Q3 = df.quantile(0.75)
  IQR = Q3 - Q1
  outliers_count_specified = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
  
  pd.DataFrame(outliers_count_specified.sort_values(ascending=False)).style.background_gradient(cmap=cm)
```

```bash
  sns.heatmap(df.corr(), annot=True)
```
```bash
  #Pandas Profiling Report
  ProfileReport(df)
```

## Data Preprocessing
```bash
  x = df.iloc[:, :-1].values
  y = df.iloc[:, -1].values
```
```bash
  x
```
```bash
  y
```

```bash
  #Normalize
  norm = StandardScaler()
  X    = norm.fit_transform(X)
```
```bash
  X
```

```bash
  #Train, Test, Split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```
```bash
  onehot_encoder = OneHotEncoder(sparse_output=False)
  y_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
  y_test = onehot_encoder.transform(y_test.reshape(-1, 1))
```
```bash
  print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
  print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
```

## Prepare Network
Use XGBoost
```bash
  xgboost = XGBClassifier()
  param_gs_xgboost = [{'gamma':[0.1, 0.2, 0.3],
                       'eta':[1e-3, 1e-2],
                       'n_estimators':[50 ,100]
                      }]
  gs_xgboost = GridSearchCV(estimator=xgboost,
                            param_grid=param_gs_xgboost,
                            scoring='accuracy',
                            cv=3)
  
  gs_XGBoost = gs_xgboost.fit(X_train, y_train)
```

```bash
  print('Best score XGBoost method : {}'.format(gs_XGBoost.best_score_))
  print('Best parameters XGBoost method : {}'.format(gs_XGBoost.best_params_))
```
```bash
  xgboost_model = gs_XGBoost.best_estimator_
```
```bash
  with open('xgboost_model.pkl', 'wb') as model_file:
    pickle.dump(xgboost_model, model_file)
```

Network
```bash
  model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='elu', input_shape=(8,)),
        tf.keras.layers.Dense(20, activation='elu'),
        tf.keras.layers.Dense(30, activation='elu'),
        tf.keras.layers.Dense(20, activation='elu'),
        tf.keras.layers.Dense(17, activation='elu'),
        tf.keras.layers.Dense(15, activation='elu'),
        tf.keras.layers.Dense(10, activation='elu'),
        tf.keras.layers.Dense(15, activation='elu',use_bias=False),
        tf.keras.layers.Dense(8,  activation='elu'),
        tf.keras.layers.Dense(4,  activation='elu'),
        tf.keras.layers.Dense(2,  activation='softmax')
  ])
```
```bash
  model.summary()
```
```bash
  plot_model(model, show_layer_activations=True, show_shapes=True)
```

## Prepare Train Objects
```bash
  OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.01)
  LOSS      = tf.keras.losses.CategoricalCrossentropy()
  METRICS   = [tf.keras.metrics.CategoricalAccuracy(),
               tf.keras.metrics.Recall(name='recall'),
               tf.keras.metrics.Precision(name='precision')]
  
  model.compile(optimizer=OPTIMIZER,
                loss=LOSS,
                metrics=METRICS)
```

## Train
```bash
  model.fit(X_train, y_train,
          epochs=60,
          validation_data=(X_test, y_test),
          batch_size=1,
          workers=2)
```

## Evaluate
```bash
  # Access the accuracy and loss values over epochs
  accuracy = model.history.history['categorical_accuracy']  # Replace 'accuracy' with 'acc' for older versions of Keras
  val_accuracy = model.history.history['val_categorical_accuracy']  # Replace 'val_accuracy' with 'val_acc' for older versions of Keras
  loss = model.history.history['loss']
  val_loss = model.history.history['val_loss']
  
  # If you have precision and recall metrics, you can access them as well
  precision = model.history.history['precision']
  val_precision = model.history.history['val_precision']
  recall = model.history.history['recall']
  val_recall = model.history.history['val_recall']
  
  # Create subplots for accuracy and loss
  plt.figure(figsize=(12, 4))
  plt.subplot(1, 2, 1)
  plt.plot(accuracy, label='Training Accuracy')
  plt.plot(val_accuracy, label='Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title('Training and Validation Accuracy')
  plt.legend()
  
  plt.subplot(1, 2, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training and Validation Loss')
  plt.legend()
  
  plt.tight_layout()
  
  # If precision and recall metrics are available, create additional plots
  if 'precision' in model.history.history:
      plt.figure(figsize=(12, 4))
      plt.subplot(1, 2, 1)
      plt.plot(precision, label='Training Precision')
      plt.plot(val_precision, label='Validation Precision')
      plt.xlabel('Epoch')
      plt.ylabel('Precision')
      plt.title('Training and Validation Precision')
      plt.legend()
  
  if 'recall' in model.history.history:
      plt.subplot(1, 2, 2)
      plt.plot(recall, label='Training Recall')
      plt.plot(val_recall, label='Validation Recall')
      plt.xlabel('Epoch')
      plt.ylabel('Recall')
      plt.title('Training and Validation Recall')
      plt.legend()
  
  plt.tight_layout()
  plt.show()
```

```bash
  model.save('diabetes_model1.h5')
  files.download('diabetes_model1.h5')
```
## Support
[TensorFlow](https://www.tensorflow.org/) is an end-to-end open source platform
for machine learning. 
Keep up-to-date with release announcements and security updates by subscribing
to
[announce@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce).
See all the [mailing lists](https://www.tensorflow.org/community/forums).

## Resources
 - [Tensorflow](https://www.tensorflow.org/)
 - [Kaggle](https://www.kaggle.com)

## License
[Apache License 2.0](LICENSE)
