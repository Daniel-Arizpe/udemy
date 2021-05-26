import requests 
import io
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
from typing import Tuple, List
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from typing import Tuple, Dict


def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url)
    return BeautifulSoup(response.content, 'html.parser')

def get_csv_from_url(url:str) -> pd.DataFrame:
    s=requests.get(url).content
    return pd.read_csv(io.StringIO(s.decode('utf-8')))

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

#Cambiamos el nombre de las columnas y eliminamos las columnas que no son necesarias.    
def name_columnsdf(file_name:str)->None:
    df_complete = pd.read_csv(file_name)
    df_complete.columns=['ID del curso', 'titulo del curso', 'URL','gratuito o paga', 'precio',
       'suscriptores', 'resenias', 'conferencias',
       'dificultad', 'duracion',
       'fecha', 'subject']
    df_complete.drop('URL', inplace=True,axis=1)
    return df_complete

df=name_columnsdf('Udemy2.csv')
df["ID del curso"] = str(df["ID del curso"])
print_tabulate(df.head(6))
df.columns
#Modificamos la columna fecha para que sea mas eficaz en el manejo de la informacion
def limpiar_fecha(str_date: str)-> datetime:
    if(type(str_date)!=str):
        str_date=str(str_date)
    date_solo=str_date.replace('-','')
    return datetime.strptime(date_solo[0:6], '%Y%m')
df['fecha'] = df['fecha'].transform(limpiar_fecha)
print(df['fecha'].head(10))

#Analisis descriptivo
def analysis(df:pd.DataFrame)->None:

    df_by_dificultad = df.groupby("dificultad")[['duracion']]
    print_tabulate(df_by_dificultad.max().sort_values(by=["duracion"], ascending=True))
    df_by_subject = df.groupby("subject")[['precio']]
    print('')
    print_tabulate(df_by_subject.median().sort_values(by=["precio"], ascending=True))
    print('')
    print_tabulate(df_by_subject.mean().sort_values(by=["precio"],ascending=True))
    return df
dficul = analysis(df)


def fraccion(df:pd.DataFrame)->pd.DataFrame:
    df['rese']=df['suscriptores']/df['resenias']
    print((df['suscriptores']))
    return df
dfrac = fraccion(df)

dfrachead = df.head()
print_tabulate(dfrachead)
print_tabulate(df.describe())




#Registros por pago
pd.value_counts(df['subject'])
plt.pie(pd.value_counts(df['gratuito o paga']), labels=['Paga','Gratuito'])
plt.title("Cursos gratuito o de paga")
plt.axis('equal')

plt.pie(pd.value_counts(df['subject']), labels=['Web Development','Business Finance','Musical Instruments','Graphic Design'])
plt.title("Conteo de cursos por subjejct")
plt.axis('equal')

df['rese'] = df['suscriptores'] / df['resenias']


#Boxplot
sns.boxplot(x=df['subject'], y=df['conferencias'])
plt.title("Boxplot de Conferencias")
plt.xticks(rotation=45)

#grafico
Tabla1=df.duracion.groupby(df.dificultad).mean().plot(kind='pie', cmap='Paired')
plt.title('Promedio de la duracion de los cursos por dificultad')
plt.ylabel('')
tabla2 = df.suscriptores.groupby(df.subject).sum().plot(kind='barh')
plt.title('Numero de suscriptores por categoria')
    
#Agrupamiento por fecha
df.columns
df_by_small = df.groupby(["fecha"])[["titulo del curso"]].aggregate(pd.DataFrame.count)
df_by_small.reset_index(inplace=True)
print_tabulate(df_by_small)
df_by_small.plot(x="fecha",y="titulo del curso", kind="scatter")
plt.ylabel("Numero de cursos")
plt.xlabel("Fecha")
plt.title("Dispersion de cursos por fecha")


# Correlación lineal entre las dos variables

corr_test = pearsonr(x = df['duracion'], y =  df['conferencias'])
print("Coeficiente de correlación de Pearson: ", corr_test[0])
print("P-value: ", corr_test[1])



def transform_variable(df: pd.DataFrame, x:str)->pd.Series:
    if isinstance(df[x][0], numbers.Number):
        return df[x] # type: pd.Series
    else:
        return pd.Series([i for i in range(0, len(df[x]))])

#Linear Models
def linear_regression(df: pd.DataFrame, x:str, y: str)->None:
    fixed_x = transform_variable(df, x)
    model= sm.OLS(df[y],sm.add_constant(fixed_x)).fit()
    print(model.summary())

    coef = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]['coef']
    df.plot(x=x,y=y, kind='scatter')
    plt.title('Regresion Lineal (Fecha, Num. Cursos)')
    plt.ylabel("Num. Cursos")
    plt.plot(df[x],[pd.DataFrame.mean(df[y]) for _ in fixed_x.items()], color='red')
    plt.plot(df[x],[ coef.values[1] * x + coef.values[0] for _, x in fixed_x.items()], color='black')
    plt.xticks(rotation=0)
    #plt.savefig(f'lr_{y}_{x}.png')
    #plt.close()

linear_regression(df_by_small, "fecha", "titulo del curso")

#----------------------------------------------------------------
########################################################################################    
def linear_regression(df: pd.DataFrame, x:str, y: str)->Dict[str, float]:
    fixed_x = transform_variable(df, x)
    model= sm.OLS(df[y],sm.add_constant(fixed_x), alpha=0.1).fit()
    bands = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]
    print_tabulate(pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0])
    coef = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]['coef']
    r_2_t = pd.read_html(model.summary().tables[0].as_html(),header=None,index_col=None)[0]
    return {'m': coef.values[1], 'b': coef.values[0], 'r2': r_2_t.values[0][3], 'r2_adj': r_2_t.values[1][3], 'low_band': bands['[0.025'][0], 'hi_band': bands['0.975]'][0]}

def plt_lr(df: pd.DataFrame, x:str, y: str, m: float, b: float, r2: float, r2_adj: float, low_band: float, hi_band: float, colors: Tuple[str,str]):
    fixed_x = transform_variable(df, x)
    df.plot(x=x,y=y, kind='scatter')
    plt.title("Forecasting Cursos por Meses")
    plt.plot(df[x],[ m * x + b for _, x in fixed_x.items()], color=colors[0])
    plt.fill_between(df[x],
                     [ m * x  + low_band for _, x in fixed_x.items()],
                     [ m * x + hi_band for _, x in fixed_x.items()], alpha=0.3, color=colors[1])

a = linear_regression(df_by_small, "fecha", "titulo del curso")
plt_lr(df=df_by_small, x="fecha", y="titulo del curso", colors=('black', 'red'), **a)

#######################################################################################

df_class = name_columnsdf('Udemy2.csv')
def limpiar_fecha(str_date: str)-> datetime:
    if(type(str_date)!=str):
        str_date=str(str_date)
    date_solo=str_date.replace('-','')
    return datetime.strptime(date_solo[0:8], '%Y%m%d')

df_class['fecha'] = df_class['fecha'].transform(limpiar_fecha)
df_class['ingresos']=df_class["precio"]*df_class["suscriptores"]
df_class.plot(x="fecha", y='ingresos', kind = 'scatter')

df_class.columns=['ID del curso', 'titulo del curso', 'pagadero', 'precio',
       'suscriptores', 'resenias', 'conferencias',
       'dificultad', 'duracion',
       'fecha', 'subject', 'ingresos']
df_class['pagadero']
#df_small = df_small.drop('titulo del curso', inplace=True,axis=1)
#print_tabulate(df_small.head(3))
df_class["pagadero"].unique()
df_subject = [True]
df_bf = df_class[df_class.pagadero.isin(df_subject)]
print_tabulate(df_bf)

df_bf["subject"].unique()
df_subject_2 = ["Business Finance"]
df_mi = df_bf[df_bf.subject.isin(df_subject_2)]
print_tabulate(df_mi)

df_mayor = df_mi[df_mi["ingresos"]>=100000]


df_by_ingresos = df_mayor.groupby(["fecha"])[["ingresos"]].aggregate(pd.DataFrame.sum)
df_by_ingresos.reset_index(inplace=True)
print(df_by_ingresos)
df_by_ingresos.plot(x="fecha",y="ingresos", kind="scatter")
plt.title("Ingresos Business Finance Mayores a $100,000")
df.plot(x='suscriptores', y='ingresos', kind='scatter')

df_by_ingresos = df_class.groupby(["duracion"])[["ingresos"]].aggregate(pd.DataFrame.sum)
df_by_ingresos.reset_index(inplace=True)
print(df_by_ingresos)
df_by_ingresos.plot(x="duracion",y="ingresos", kind="scatter")
plt.title("Ingresos percibidos por duracion de cursos")
df.plot(x='suscriptores', y='ingresos', kind='scatter')


