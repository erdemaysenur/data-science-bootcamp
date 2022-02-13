import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os
from columnar import columnar
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")


class DataOperations:
    """
    DataOperations class. Create an operator object to carry out operations. Attributes:
    read_data_from_directory: Searchs csv or json data from given directory. Returns pandas.DataFrame
    read_data_from_numpy: Converts numpy.array object to pandas.DataFrame
    create_random_dataframe: Returns random pandas.DataFrame. If there is not given parameters, it will use default parameters. For all available parameters: sklearn.datasets.make_classification
    """

    def __init__(self):
        print("DataOperations object created.")


    def read_data_from_directory(self, path):
        """
        DataOperations class read_data_from_path method. Reads files from given directory. Available formats:
        .csv
        .json
        """
        for file in os.listdir(path):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, file))
            elif file.endswith(".json"):
                df = pd.read_json(os.path.join(path, file), orient="records")
        return df


    def read_data_from_numpy(self, array):
        """
        DataOperations class read_data_from_numpy method. Returns pandas.DataFrame from given numpy.array
        """
        df = pd.DataFrame(data=array, index=[i for i in range(array.shape[0])], columns=['feature'+str(i) for i in range(array.shape[1])])
        return df


    def create_random_dataframe(self, **kwargs):
        """
        DataOperations class create_random_dataframe object. 
        
        """
        random_created_dataset = make_classification(n_classes=2, n_samples=100, n_features=5, **kwargs)
        self.rows = random_created_dataset[0].shape[0]
        self.cols = random_created_dataset[0].shape[1]
        data = random_created_dataset[0]
        df = pd.DataFrame(data=data, index=[i for i in range(self.rows)], columns=['feature'+str(i) for i in range(self.cols)])
        df["label"] = random_created_dataset[1]
        return df


    def analyse(self, df):
        """
        A quick statistical analysis of dataset. Tables of numeric features and non-numerics features are shown seperately.
        """
        df_numeric = df.select_dtypes(exclude=["string","object"])
        table_numeric = [["NAME", "DTYPE", "NUMBER OF UNIQUE VALUES", "MEAN", "MEDIAN", "MODE", "STD", "MAX", "MIN"]]
        for col in df_numeric.columns:
            table_numeric.append([col, df_numeric[col].dtype, len(df_numeric[col].unique()), round(df_numeric[col].mean(),2), df_numeric[col].median(), df_numeric[col].mode()[0], round(df_numeric[col].std(),2), df_numeric[col].max(), df_numeric[col].min()])

        df_object = df.select_dtypes(include=["string", "object"])
        table_object = [["NAME", "DTYPE", "NUMBER OF UNIQUE VALUES", "MODE"]]

        for col in df_object.columns:
            table_object.append([col, df_object[col].dtype, len(df_object[col].unique()), df_object[col].mode()[0]])

        print(columnar(table_numeric, no_borders=True, justify="c")) 
        print(columnar(table_object, no_borders=True, justify="c")) 


    def fast_plot(self, df, target=None, idx=None):
        """
        fast_plot method. Creates plots from given dataframe, target column and id column.
        """        
        
        if target != None and idx != None:
            label = df[target]
            plt.figure(figsize=(8,6))
            plt.title("Target value counts")
            label.value_counts().plot(kind="bar",color="deepskyblue")
            plt.show()
            for i in df.columns:
                plt.figure(figsize=(8,6))
                plt.title(f"Distribution of {i} column")
                sns.scatterplot(data=df, x=idx, y=i, hue=target, palette="inferno")
                plt.show()
            cm = df.drop(idx, axis=1).corr()
            mask = np.zeros_like(cm)
            mask[np.triu_indices_from(mask)] = True
            sns.heatmap(cm, annot=False, mask=mask, cmap="cubehelix")
            plt.show()
        else:
            plt.figure(figsize=(10,7))
            sns.boxplot(data=df, orient="h", palette="inferno")
            plt.show()

            cm = df.corr()
            mask = np.zeros_like(cm)
            mask[np.triu_indices_from(mask)] = True
            sns.heatmap(cm, annot=False, mask=mask, cmap="cubehelix")
            plt.show()

