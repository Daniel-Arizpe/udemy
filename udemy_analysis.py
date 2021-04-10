import requests 
import io
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
from typing import Tuple, List
import re
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url)
    return BeautifulSoup(response.content, 'html.parser')

def get_csv_from_url(url:str) -> pd.DataFrame:
    s=requests.get(url).content
    return pd.read_csv(io.StringIO(s.decode('utf-8')))

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

def analysis(file_name:str)->None:
    df_complete = pd.read_csv(file_name)
    df_complete.columns=['ID del curso', 'titulo del curso', 'URL','gratuito o paga', 'precio',
       'suscriptores', 'resenias', 'conferencias',
       'dificultad', 'duracion',
       'published_timestamp', 'subject']
    df_by_dificultad = df_complete.groupby("dificultad")[['duracion']]
    print_tabulate(df_by_dificultad.max().sort_values(by=["duracion"], ascending=True))
    df_by_subject = df_complete.groupby("subject")[['precio']]
    print('')
    print_tabulate(df_by_subject.median().sort_values(by=["precio"], ascending=True))
    print('')
    print_tabulate(df_by_subject.max().sort_values(by=["precio"],ascending=True))
    return df_complete
dficul = analysis('Udemy2.csv')


def fraccion(file_name:str)->pd.DataFrame:
    df=pd.read_csv(file_name)
    df.columns=['ID del curso', 'titulo del curso', 'URL','gratuito o paga', 'precio',
       'suscriptores', 'resenias', 'conferencias',
       'dificultad', 'duracion',
       'published_timestamp', 'subject']
    df['rese']=df['suscriptores']/df['resenias']
    print((df['suscriptores']))
    return df
df = fraccion('Udemy2.csv')

df.columns
df = fraccion('Udemy2.csv')
dfhead = df.head()
print_tabulate(dfhead)
print_tabulate(df.describe())



#Leer el archivo
df = pd.read_csv('Udemy2.csv')
df.index
df.columns=['ID del curso', 'titulo del curso', 'URL','gratuito o paga', 'precio',
       'suscriptores', 'resenias', 'conferencias',
       'dificultad', 'duracion',
       'published_timestamp', 'subject']
df.columns


#Registros por pago
pd.value_counts(df['gratuito o paga'])
plt.pie(pd.value_counts(df['subject']))
df['rese'] = df['suscriptores'] / df['resenias']


#Boxplot
sns.boxplot(x=df['subject'], y=df['rese'])
plt.axis([ 0, 200])
#grafico
Tabla1=df.duracion.groupby(df.dificultad).count().plot(kind='pie', cmap='Paired')
plt.axis('equal')
tabla2 = df.suscriptores.groupby(df.subject).sum().plot(kind='barh')
