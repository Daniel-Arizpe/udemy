import requests
import io
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
from typing import Tuple, List
import re
import datetime

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
    df_by_dificultad = df_complete.groupby("Dificultad del curso")[['Duración de todos los materiales del curso']]
    print_tabulate(df_by_dificultad.min().sort_values(by=["Duración de todos los materiales del curso"], ascending=False))
   
    df_by_subject = df_complete.groupby("subject")[['Precio del curso']]
    print('')
    print_tabulate(df_by_subject.sum().sort_values(by=["Precio del curso"], ascending=False))
   
    return df_by_dificultad, df_by_subject
    
df = analysis('Udemy2.csv')

#print_tabulate(df.head())
#print_tabulate(df.describe())
#print(df["Precio del curso"].sum())    
#df = pd.read_csv("C:/Users/arizp/OneDrive/Documents/FCFM/7° Semestre/Minería de datos/data_mining/Udemy2.csv")
