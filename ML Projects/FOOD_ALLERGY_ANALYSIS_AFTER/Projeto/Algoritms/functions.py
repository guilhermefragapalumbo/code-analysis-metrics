import numpy as np
from matplotlib import pyplot as plt


def resolve_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3 - q1
    superior = df[~(df > (q3 + 1.5 * IQR))].max()
    inferior = df[~(df < (q1 - 1.5 * IQR))].min()

    df_temp = np.where(df > superior, df.mean(),
                       np.where(df < inferior, df.mean(),
                                np.where(df <= 0, df.mean(),
                                         np.where(df >= 2023, 1998, df
                                                  )
                                         )
                                )
                       )
    return df_temp
