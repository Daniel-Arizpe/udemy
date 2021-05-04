# -*- coding: utf-8 -*-
"""
Created on Tue May  4 00:08:18 2021

@author: arizp
"""

#Queremos analizar si existe una relacion entre los datos de los cursos de Udemy
#Del numero de resenias, con el numero de suspcriptores

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from tabulate import tabulate
import statsmodels.api as sm
from scipy import stats
import numbers
from scipy.stats import pearsonr

def name_columnsdf(file_name:str)->None:
    df_complete = pd.read_csv(file_name)
    df_complete.columns=['ID del curso', 'titulo del curso', 'URL','gratuito o paga', 'precio',
       'suscriptores', 'resenias', 'conferencias',
       'dificultad', 'duracion',
       'published_timestamp', 'subject']
    return df_complete

df=name_columnsdf('Udemy2.csv')
print(df)

fig, ax = plt.subplots(figsize=(7, 3.85))

df.plot(
    x    = 'duracion',
    y    = 'conferencias',
    c    = 'firebrick',
    kind = "scatter",
    ax   = ax
)
ax.set_title('Distribución de duracion y conferencias');

# Correlación lineal entre las dos variables

corr_test = pearsonr(x = df['duracion'], y =  df['conferencias'])
print("Coeficiente de correlación de Pearson: ", corr_test[0])
print("P-value: ", corr_test[1])



def transform_variable(df: pd.DataFrame, x:str)->pd.Series:
    if isinstance(df[x][0], numbers.Number):
        return df[x] # type: pd.Series
    else:
        return pd.Series([i for i in range(0, len(df[x]))])


def linear_regression(df: pd.DataFrame, x:str, y: str)->None:
    fixed_x = transform_variable(df, x)
    model= sm.OLS(df[y],sm.add_constant(fixed_x)).fit()
    print(model.summary())

    coef = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]['coef']
    df.plot(x=x,y=y, kind='scatter')
    plt.plot(df[x],[pd.DataFrame.mean(df[y]) for _ in fixed_x.items()], color='green')
    plt.plot(df[x],[ coef.values[1] * x + coef.values[0] for _, x in fixed_x.items()], color='red')
    plt.xticks(rotation=90)
    plt.savefig(f'data_minig/lr_{y}_{x}.png')
    plt.close()


#df = pd.read_csv(".csv") # type: pd.DataFrame
#print_tabulate(df.hea'd(50'))
'''df_by_sal = df.groupby("Fecha")\
              .aggregate(sueldo_mensual=pd.NamedAgg(column="Sueldo Neto", aggfunc=pd.DataFrame.mean))
# df_by_sal["sueldo_mensual"] = df_by_sal["sueldo_mensual"]**10
df_by_sal.reset_index(inplace=True)
print(df_by_sal.head(5))'''
linear_regression(df, "duracion", "conferencias")