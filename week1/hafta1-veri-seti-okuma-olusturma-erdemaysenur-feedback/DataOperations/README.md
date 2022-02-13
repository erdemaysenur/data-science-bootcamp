<p align="center" width="100%">
    <img src="do.png">
</p>

DataOperations class is made to carry out some quick loadings, analysing and visualizing. This project is the 1st homework of [Kodluyoruz & Carbon Consulting - Data Science Bootcamp](https://github.com/K132-Veri-Bilimi-Bootcamp).

## Usage

If you'd like to use this tool, you should clone this repository or download the file 'ayse.py'


Once you have 'ayse.py' in your local, you can use it in your project that is located in the same directory. Then you can import it as follows:

```
from ayse import DataOperations

operator = DataOperations()
```

## Loading data

After defined an operator object, you can use it to load, analyse and visualize your dataset.
To search and read data from given directory, the function needed:

```
data_from_directory = operator.read_data_from_directory("path_to_directory")
```
- This function searches directory and it is able to load .csv or .json files and convert them pandas.DataFrame object.

If you have a numpy array in your workspace or would like to directly create as input, the function needed:

```
data_from_numpy = operator.read_data_from_numpy(array)
```
- Returns pandas.DataFrame object.


You can also make the tool create random dataset and specify parameters. The function needed:

```
random_created_data = operator.create_random_dataframe(**kwargs)
```
- Returns pandas.DataFrame
- **kwargs:

    - n_samples=100,
    - n_features=20,
    - *,
    - n_informative=2,
    - n_redundant=2,
    - n_repeated=0,
    - n_classes=2,
    - n_clusters_per_class=2,
    - weights=None,
    - flip_y=0.01,
    - class_sep=1.0,
    - hypercube=True,
    - shift=0.0,
    - scale=1.0,
    - shuffle=True,
    - random_state=None

    [source](https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/datasets/_samples_generator.py#L39)

## Analysis

DataOperations class provides a quick statistical informations shown in tables. To use it:

```
operator.analyse(df)
```

## Visualization

To use DataOperations visualizer:

```
operator.fast_plot(df, target=None, idx=None)
```

If target column does not exit in your dataset or index-like column, you can pass these parameters.

## Requirements

- Columnar==1.4.1
- matplotlib==3.4.3
- numpy==1.19.5
- pandas==1.3.2
- seaborn==0.11.2

## Contribution

Since this is a beginner friendly homework project, any kind of contributions or suggestions are welcome. If you have a word, open a issue, fork, and send pull request. ✨

