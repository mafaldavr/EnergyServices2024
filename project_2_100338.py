#BIBLIOTECAS NECESSÁRIAS
import dash
import dash_core_components as dcc
from dash import html
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import pickle
import os
from sklearn import  metrics
import numpy as np

#-------------------------------------

#CRIAR UMA DATAFRAME COM OS DADOS DE JANEIRO, FEVEREIRO E MARÇO DE 2019
data = pd.read_csv('2019.csv') #ler o ficheiro
data['Date'] = pd.to_datetime(data['Date']) #converter para datetime
data = data[data['Date'].dt.month != 4] #eliminar dados de abril
data.loc[data['Date'].dt.month == 1, 'Mês'] = 'Janeiro' #adicionar coluna com o mês
data.loc[data['Date'].dt.month == 2, 'Mês'] = 'Fevereiro'
data.loc[data['Date'].dt.month == 3, 'Mês'] = 'Março'
data = data.set_index('Date', drop = True) #definir como indíce a data
data['Hora (h)'] = data.index.hour #adicionar coluna com a hora
data['Dia de Semana'] = data.index.weekday // 5 #adicionar coluna que indica se é fim de semana
data['Consumo  da Hora Anterior (kWh)'] = data['North Tower (kWh)'].shift(1) #adicionar coluna com o consumo da hora anterior
data = data.dropna() #eliminar a primeira linha
data = data.rename(columns = {'North Tower (kWh)':'Consumo Atual (kWh)', 'temp_C':'Temperatura (\u00b0C)', 'HR':'Taxa de Humidade (%)', 'windSpeed_m/s':'Velocidade do Vento (m/s)', 'windGust_m/s':'Rajada de Vento (m/s)', 'pres_mbar':'Pressão Atmosférica (mbar)', 'solarRad_W/m2':'Radiação Solar (W/m²)', 'rain_mm/h':'Taxa de Precipitação (mm/h)', 'rain_day':'Precipitação Acumulada (mm/dia)'}) #Alterar o nome das colunas
data = data.iloc[:, [0,12,1,2,6,5,3,4,7,8,11,10,9]] #Alterar a posição das colunas

#-------------------------------------

#CRIAR OS DOIS POSSÍVEIS CONJUNTOS DE INPUTS E OUTPUTS
data_3 = data.drop(['Velocidade do Vento (m/s)', 'Rajada de Vento (m/s)', 'Pressão Atmosférica (mbar)', 'Taxa de Precipitação (mm/h)', 'Precipitação Acumulada (mm/dia)', 'Mês', 'Taxa de Humidade (%)', 'Radiação Solar (W/m²)'], axis=1) #eliminar as colunas que não são necessárias para prever o consumo
values_3 = data_3.values #guardar apenas os valores do dataframes
outputs_3 = values_3[:,0] #definir output como consumo por hora
inputs_3 = values_3[:,[1,2,3,4]] #definir inputs como condições meteorológicas, consumo na hora anterior, tipo de dia e hora
data_5 = data.drop(['Velocidade do Vento (m/s)', 'Rajada de Vento (m/s)', 'Pressão Atmosférica (mbar)', 'Taxa de Precipitação (mm/h)', 'Precipitação Acumulada (mm/dia)', 'Mês'], axis=1) #eliminar as colunas que não são necessárias para prever o consumo
values_5 = data_5.values #guardar apenas os valores do dataframes
outputs_5 = values_5[:,0] #definir output como consumo por hora
inputs_5 = values_5[:,[1,2,3,4,5,6]] #definir inputs como condições meteorológicas, consumo na hora anterior, tipo de dia e hora

#-------------------------------------

#FAZER LOAD DOS DIFERENTES MÉTODOS DISPONÍVEIS, CALCULAR AS PREVISÕES E RESPETIVOS ERROS
def load_model(model_filename):
    model_path = os.path.join('MODELS', model_filename)
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model
RFR_3 = load_model('model_RFR_3.pkl') #Regressor de Floresta Aleatória com 4 variáveis
predictions_RFR_3 = RFR_3.predict(inputs_3) #prever o cosumo energético
MAE_RFR_3 = metrics.mean_absolute_error(outputs_3,predictions_RFR_3) #calcular metricas
MBE_RFR_3 = np.mean(outputs_3-predictions_RFR_3)
MSE_RFR_3 = metrics.mean_squared_error(outputs_3,predictions_RFR_3)  
RMSE_RFR_3 = np.sqrt(metrics.mean_squared_error(outputs_3,predictions_RFR_3))
cvRMSE_RFR_3 = RMSE_RFR_3/np.mean(outputs_3)
NMBE_RFR_3 = MBE_RFR_3/np.mean(outputs_3)
values_RFR_3 = [MAE_RFR_3, MBE_RFR_3, MSE_RFR_3, RMSE_RFR_3, cvRMSE_RFR_3, NMBE_RFR_3]
RFR_5 = load_model('model_RFR_5.pkl') #Regressor de Floresta Aleatória com 6 variáveis
predictions_RFR_5 = RFR_5.predict(inputs_5) #prever o cosumo energético
MAE_RFR_5 = metrics.mean_absolute_error(outputs_5,predictions_RFR_5) #calcular metricas
MBE_RFR_5 = np.mean(outputs_5-predictions_RFR_5)
MSE_RFR_5 = metrics.mean_squared_error(outputs_5,predictions_RFR_5)  
RMSE_RFR_5 = np.sqrt(metrics.mean_squared_error(outputs_5,predictions_RFR_5))
cvRMSE_RFR_5 = RMSE_RFR_5/np.mean(outputs_5)
NMBE_RFR_5 = MBE_RFR_5/np.mean(outputs_5)
values_RFR_5 = [MAE_RFR_5, MBE_RFR_5, MSE_RFR_5, RMSE_RFR_5, cvRMSE_RFR_5, NMBE_RFR_5]
"""DTR_3 = load_model('model_DTR_3.pkl') #Regressor de Árvore de Decisão com 4 variáveis
predictions_DTR_3 = DTR_3.predict(inputs_3) #prever o cosumo energético
MAE_DTR_3 = metrics.mean_absolute_error(outputs_3,predictions_DTR_3) #calcular metricas
MBE_DTR_3 = np.mean(outputs_3-predictions_DTR_3)
MSE_DTR_3 = metrics.mean_squared_error(outputs_3,predictions_DTR_3)  
RMSE_DTR_3 = np.sqrt(metrics.mean_squared_error(outputs_3,predictions_DTR_3))
cvRMSE_DTR_3 = RMSE_DTR_3/np.mean(outputs_3)
NMBE_DTR_3 = MBE_DTR_3/np.mean(outputs_3)
values_DTR_3 = [MAE_DTR_3, MBE_DTR_3, MSE_DTR_3, RMSE_DTR_3, cvRMSE_DTR_3, NMBE_DTR_3]
DTR_5 = load_model('model_DTR_5.pkl') #Regressor de Árvore de Decisão com 6 variáveis
predictions_DTR_5 = DTR_5.predict(inputs_5) #prever o cosumo energético
MAE_DTR_5 = metrics.mean_absolute_error(outputs_5,predictions_DTR_5) #calcular metricas
MBE_DTR_5 = np.mean(outputs_5-predictions_DTR_5)
MSE_DTR_5 = metrics.mean_squared_error(outputs_5,predictions_DTR_5)  
RMSE_DTR_5 = np.sqrt(metrics.mean_squared_error(outputs_5,predictions_DTR_5))
cvRMSE_DTR_5 = RMSE_DTR_5/np.mean(outputs_5)
NMBE_DTR_5 = MBE_DTR_5/np.mean(outputs_5)
values_DTR_5 = [MAE_DTR_5, MBE_DTR_5, MSE_DTR_5, RMSE_DTR_5, cvRMSE_DTR_5, NMBE_DTR_5]"""
GBR_3 = load_model('model_GBR_3.pkl') #Regressor de Impulso Gradiente com 4 variáveis
predictions_GBR_3 = GBR_3.predict(inputs_3) #prever o cosumo energético
MAE_GBR_3 = metrics.mean_absolute_error(outputs_3,predictions_GBR_3) #calcular metricas
MBE_GBR_3 = np.mean(outputs_3-predictions_GBR_3)
MSE_GBR_3 = metrics.mean_squared_error(outputs_3,predictions_GBR_3)  
RMSE_GBR_3 = np.sqrt(metrics.mean_squared_error(outputs_3,predictions_GBR_3))
cvRMSE_GBR_3 = RMSE_GBR_3/np.mean(outputs_3)
NMBE_GBR_3 = MBE_GBR_3/np.mean(outputs_3)
values_GBR_3 = [MAE_GBR_3, MBE_GBR_3, MSE_GBR_3, RMSE_GBR_3, cvRMSE_GBR_3, NMBE_GBR_3]
GBR_5 = load_model('model_GBR_5.pkl') #Regressor de Impulso Gradiente com 6 variáveis
predictions_GBR_5 = GBR_5.predict(inputs_5) #prever o cosumo energético
MAE_GBR_5 = metrics.mean_absolute_error(outputs_5,predictions_GBR_5) #calcular metricas
MBE_GBR_5 = np.mean(outputs_5-predictions_GBR_5)
MSE_GBR_5 = metrics.mean_squared_error(outputs_5,predictions_GBR_5)  
RMSE_GBR_5 = np.sqrt(metrics.mean_squared_error(outputs_5,predictions_GBR_5))
cvRMSE_GBR_5 = RMSE_GBR_5/np.mean(outputs_5)
NMBE_GBR_5 = MBE_GBR_5/np.mean(outputs_5)
values_GBR_5 = [MAE_GBR_5, MBE_GBR_5, MSE_GBR_5, RMSE_GBR_5, cvRMSE_GBR_5, NMBE_GBR_5]
BR_3 = load_model('model_BR_3.pkl') #Regressor de Bagging com 4 variáveis
predictions_BR_3 = BR_3.predict(inputs_3) #prever o cosumo energético
MAE_BR_3 = metrics.mean_absolute_error(outputs_3,predictions_BR_3) #calcular metricas
MBE_BR_3 = np.mean(outputs_3-predictions_BR_3)
MSE_BR_3 = metrics.mean_squared_error(outputs_3,predictions_BR_3)  
RMSE_BR_3 = np.sqrt(metrics.mean_squared_error(outputs_3,predictions_BR_3))
cvRMSE_BR_3 = RMSE_BR_3/np.mean(outputs_3)
NMBE_BR_3 = MBE_BR_3/np.mean(outputs_3)
values_BR_3 = [MAE_BR_3, MBE_BR_3, MSE_BR_3, RMSE_BR_3, cvRMSE_BR_3, NMBE_BR_3]
BR_5 = load_model('model_BR_5.pkl') #Regressor de Bagging com 6 variáveis
predictions_BR_5 = BR_5.predict(inputs_5) #prever o cosumo energético
MAE_BR_5 = metrics.mean_absolute_error(outputs_5,predictions_BR_5) #calcular metricas
MBE_BR_5 = np.mean(outputs_5-predictions_BR_5)
MSE_BR_5 = metrics.mean_squared_error(outputs_5,predictions_BR_5)  
RMSE_BR_5 = np.sqrt(metrics.mean_squared_error(outputs_5,predictions_BR_5))
cvRMSE_BR_5 = RMSE_BR_5/np.mean(outputs_5)
NMBE_BR_5 = MBE_BR_5/np.mean(outputs_5)
values_BR_5 = [MAE_BR_5, MBE_BR_5, MSE_BR_5, RMSE_BR_5, cvRMSE_BR_5, NMBE_BR_5]
MLPR_3 = load_model('model_MLPR_3.pkl') #Regressor de Perceptron de Múltiplas Camadas com 4 variáveis
predictions_MLPR_3 = MLPR_3.predict(inputs_3) #prever o cosumo energético
MAE_MLPR_3 = metrics.mean_absolute_error(outputs_3,predictions_MLPR_3) #calcular metricas
MBE_MLPR_3 = np.mean(outputs_3-predictions_MLPR_3)
MSE_MLPR_3 = metrics.mean_squared_error(outputs_3,predictions_MLPR_3)  
RMSE_MLPR_3 = np.sqrt(metrics.mean_squared_error(outputs_3,predictions_MLPR_3))
cvRMSE_MLPR_3 = RMSE_MLPR_3/np.mean(outputs_3)
NMBE_MLPR_3 = MBE_MLPR_3/np.mean(outputs_3)
values_MLPR_3 = [MAE_MLPR_3, MBE_MLPR_3, MSE_MLPR_3, RMSE_MLPR_3, cvRMSE_MLPR_3, NMBE_MLPR_3]
MLPR_5 = load_model('model_MLPR_5.pkl') #Regressor de Perceptron de Múltiplas Camadas com 6 variáveis
predictions_MLPR_5 = MLPR_5.predict(inputs_5) #prever o cosumo energético
MAE_MLPR_5 = metrics.mean_absolute_error(outputs_5,predictions_MLPR_5) #calcular metricas
MBE_MLPR_5 = np.mean(outputs_5-predictions_MLPR_5)
MSE_MLPR_5 = metrics.mean_squared_error(outputs_5,predictions_MLPR_5)  
RMSE_MLPR_5 = np.sqrt(metrics.mean_squared_error(outputs_5,predictions_MLPR_5))
cvRMSE_MLPR_5 = RMSE_MLPR_5/np.mean(outputs_5)
NMBE_MLPR_5 = MBE_MLPR_5/np.mean(outputs_5)
values_MLPR_5 = [MAE_MLPR_5, MBE_MLPR_5, MSE_MLPR_5, RMSE_MLPR_5, cvRMSE_MLPR_5, NMBE_MLPR_5]
LR_3 = load_model('model_LR_3.pkl') #Regressão Linear com 4 variáveis
predictions_LR_3 = LR_3.predict(inputs_3) #prever o cosumo energético
MAE_LR_3 = metrics.mean_absolute_error(outputs_3,predictions_LR_3) #calcular metricas
MBE_LR_3 = np.mean(outputs_3-predictions_LR_3)
MSE_LR_3 = metrics.mean_squared_error(outputs_3,predictions_LR_3)  
RMSE_LR_3 = np.sqrt(metrics.mean_squared_error(outputs_3,predictions_LR_3))
cvRMSE_LR_3 = RMSE_LR_3/np.mean(outputs_3)
NMBE_LR_3 = MBE_LR_3/np.mean(outputs_3)
values_LR_3 = [MAE_LR_3, MBE_LR_3, MSE_LR_3, RMSE_LR_3, cvRMSE_LR_3, NMBE_LR_3]
LR_5 = load_model('model_LR_5.pkl')#Regressão Linear com 6 variáveis
predictions_LR_5 = LR_5.predict(inputs_5) #prever o cosumo energético
MAE_LR_5 = metrics.mean_absolute_error(outputs_5,predictions_LR_5) #calcular metricas
MBE_LR_5 = np.mean(outputs_5-predictions_LR_5)
MSE_LR_5 = metrics.mean_squared_error(outputs_5,predictions_LR_5)  
RMSE_LR_5 = np.sqrt(metrics.mean_squared_error(outputs_5,predictions_LR_5))
cvRMSE_LR_5 = RMSE_LR_5/np.mean(outputs_5)
NMBE_LR_5 = MBE_LR_5/np.mean(outputs_5)
values_LR_5 = [MAE_LR_5, MBE_LR_5, MSE_LR_5, RMSE_LR_5, cvRMSE_LR_5, NMBE_LR_5]

#------------------------------------- 

#DEFINIR DATAFRAMES COM AS MÉTRICAS
column_names = ['Erro Médio Absoluto', 'Erro Médio Quadrático', 'Raiz do Erro Médio Quadrático', 'Erro Médio de Viés', 'Erro Médio de Viés Normalizado', 'Coeficiente de Variação do Erro Médio Quadrático'] #dar nome às colunas
metrics_3 = pd.DataFrame(columns=column_names) #criar dataframe
metrics_3.loc['Regressor de Floresta Aleatória'] = values_RFR_3
#metrics_3.loc['Regressor de Árvore de Decisão'] = values_DTR_3
metrics_3.loc['Regressor de Impulso Gradiente'] = values_GBR_3
metrics_3.loc['Regressor de Bagging'] = values_BR_3
metrics_3.loc['Regressor de Perceptron de Múltiplas Camadas'] = values_MLPR_3
metrics_3.loc['Regressão Linear'] = values_LR_3
metrics_5 = pd.DataFrame(columns=column_names) #criar dataframe
metrics_5.loc['Regressor de Floresta Aleatória'] = values_RFR_5
#metrics_5.loc['Regressor de Árvore de Decisão'] = values_DTR_5
metrics_5.loc['Regressor de Impulso Gradiente'] = values_GBR_5
metrics_5.loc['Regressor de Bagging'] = values_BR_5
metrics_5.loc['Regressor de Perceptron de Múltiplas Camadas'] = values_MLPR_5
metrics_5.loc['Regressão Linear'] = values_LR_5

#------------------------------------- 

#CRIAR LISTAS COM AS POSSÍVEIS ALTERNATIVAS
available_months = ['Janeiro', 'Fevereiro', 'Março'] #criar lista com os meses disponíveis
available_data = ['Consumo Atual (kWh)', 'Consumo  da Hora Anterior (kWh)', 'Dia de Semana', 'Temperatura (\u00b0C)', 'Taxa de Humidade (%)', 'Radiação Solar (W/m²)', 'Pressão Atmosférica (mbar)', 'Velocidade do Vento (m/s)', 'Rajada de Vento (m/s)', 'Taxa de Precipitação (mm/h)', 'Precipitação Acumulada (mm/dia)'] #criar lista com as variáveis disponíveis  
available_view = ['Gráfico', 'Tabela'] #criar lista com visualizações disponíveis
available_views = ['Gráfico', 'Tabela', 'Outliers']
available_variables = ['Dia de Semana, Hora, Consumo  da Hora Anterior, Temperatura', 'Dia de Semana, Hora, Consumo  da Hora Anterior, Temperatura, Taxa de Humidade, Radiação Solar'] #criar lista com as variáveis disponíveis para enviar para o método
available_methods = ['Regressor de Floresta Aleatória', """'Regressor de Árvore de Decisão'""", 'Regressor de Impulso Gradiente', 'Regressor de Bagging', 'Regressor de Perceptron de Múltiplas Camadas', 'Regressão Linear'] #criar lista com os métodos disponíveis
available_metrics = ['Erro Médio Absoluto', 'Erro Médio Quadrático', 'Raiz do Erro Médio Quadrático', 'Erro Médio de Viés', 'Erro Médio de Viés Normalizado', 'Coeficiente de Variação do Erro Médio Quadrático'] #criar lista com as métricas disponíveis

#-------------------------------------

#CRIAR DASHBOARD
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([

#-------------------------------------    
    
#CRIAR UM CABEÇALHO ELEGANTE
    html.Div(style={'position':'relative', 'padding-top':'90px'},
    children=[
        html.Img(src='IMAGES/logo.jpg', style={'position':'absolute', 'top':'5%', 'left':'5%', 'width':'168px', 'height':'67px'}),
        html.Img(src='IMAGES/light.jpg', style={'position':'absolute', 'top':'5%', 'right':'5%', 'width':'67px','height': '67px'})]),
    html.H1('Previsão do Consumo Energético', style={'color':'#069EE1', 'text-align':'center', 'font-family':'Arial'}), #inserir títulos
    html.H2('Torre Norte do Instituto Superior Técnico', style={'color':'#42545E', 'text-align':'center', 'font-family':'Arial'}),
    html.H3('Mafalda Vila Rodrigues 100338', style={'color':'#42545E', 'text-align':'center', 'font-family':'Arial'}),
    
#-------------------------------------    
    
#CRIAR MENU QUE PERMITE ESCOLHER AS VARIÁVEIS, MÊS E FORMA DE VER
    html.H3('Dados de Consumo Energético e Condições Meteorlógicas', style={'color':'#069EE1', 'text-align':'center', 'margin-top': '50px', 'font-family':'Arial'}),
    html.Div([
    html.Div([ #criar lista que permite ao utilizador escolher o período temporal dos dados a visualizar
        html.Label("Selecione os Meses:"),
        dcc.Checklist(
            id = 'dados_meses',
            options = [{'label': i, 'value': i} for i in available_months],
            value = ['Janeiro'],
            labelStyle = {'display':'block'}
        ),
    ], style = {'vertical-align':'top', 'display': 'inline-block', 'width': '30%', 'font-family':'Arial'}),
    html.Div([ #criar lista que permite ao utilizador escolher as variáveis a visualizar
        html.Label("Selecione as Variáveis:"),
        dcc.Checklist(
            id = 'dados_variaveis',
            options = [{'label': i, 'value': i} for i in available_data],
            value = ['Consumo Atual (kWh)'],
            labelStyle = {'display':'block'}
        ),
    ], style={'vertical-align':'top', 'display': 'inline-block', 'width': '30%', 'font-family':'Arial'}),
    html.Div([ #criar lista que permite ao utilizador escolher a forma como vai visualizar os dados
        html.Label("Selecione a Visualização:"),
        dcc.RadioItems(
            id = 'dados_visualizacao',
            options = [{'label': i, 'value': i} for i in available_views],
            value = 'Gráfico',
            labelStyle = {'display':'block'}
        ),
    ], style = {'vertical-align':'top', 'display': 'inline-block', 'width': '30%', 'font-family':'Arial'}),
    html.Div(
        id='dados_resultados')
    ], style={'text-align':'center', 'margin-top': '20px'}),
    
#-------------------------------------    
    
#CRIAR MENU QUE PERMITE ESCOLHER O MÉTODO E VARIAVÉIS A USAR
    html.H3('Previsão do Consumo Energético', style={'color':'#069EE1', 'text-align':'center', 'margin-top': '50px', 'font-family':'Arial'}),
    html.Div([
    html.Div([ #criar lista que permite ao utilizador escolher as variáveis a usar
        html.Label("Selecione as Variáveis:"),
        dcc.RadioItems(
            id = 'metodos_variaveis',
            options = [{'label': i, 'value': i} for i in available_variables],
            value = 'Dia de Semana, Hora, Consumo  da Hora Anterior, Temperatura',
            labelStyle = {'display':'block'}
        ),
    ], style = {'vertical-align':'top', 'display': 'inline-block', 'width': '30%', 'font-family':'Arial'}),
    html.Div([ #criar lista que permite ao utilizador escolher o método a usar
        html.Label("Selecione o Método:"),
        dcc.RadioItems(
            id = 'metodos',
            options = [{'label': i, 'value': i} for i in available_methods],
            value = 'Regressor de Floresta Aleatória',
            labelStyle = {'display':'block'}
        ),
    ], style = {'vertical-align':'top', 'display': 'inline-block', 'width': '30%', 'font-family':'Arial'}),
    html.Div([ #criar lista que permite ao utilizador escolher o modo de ver
        html.Label("Selecione a Visualização:"),
        dcc.RadioItems(
            id = 'metodos_visualizacao',
            options = [{'label': i, 'value': i} for i in available_view],
            value = 'Gráfico',
            labelStyle = {'display':'block'}
        ),
    ], style = {'vertical-align':'top', 'display': 'inline-block', 'width': '30%', 'font-family':'Arial'}),
    html.Div(
        id='metodos_resultados')
    ], style={'text-align':'center', 'margin-top': '20px'}),
    
#-------------------------------------    
    
#CRIAR MENU QUE PERMITE ESCOLHER AS MÉTRICAS E MÉTODOS A VER 
    html.H3('Erros da Previsão do Consumo Energético', style={'color':'#069EE1', 'text-align':'center', 'margin-top': '50px', 'font-family':'Arial'}),
    html.Div([ 
    html.Div([ #criar lista que permite ao utilizador escolher as variáveis a usar
        html.Label("Selecione as Variáveis:"),
        dcc.RadioItems(
            id = 'metricas_variaveis',
            options = [{'label': i, 'value': i} for i in available_variables],
            value = 'Dia de Semana, Hora, Consumo  da Hora Anterior, Temperatura',
            labelStyle = {'display':'block'}
        ),
    ], style = {'vertical-align':'top', 'display': 'inline-block', 'width': '30%', 'font-family':'Arial'}),
    html.Div([ #criar lista que permite ao utilizador escolher os métodos das previsões a ver
        html.Label("Selecione os Métodos:"),
        dcc.Checklist(
            id = 'metricas_metodos',
            options = [{'label': i, 'value': i} for i in available_methods],
            value = ['Regressor de Floresta Aleatória'],
            labelStyle = {'display':'block'}
        ),
    ], style = {'vertical-align':'top', 'display': 'inline-block', 'width': '30%', 'font-family':'Arial'}),
    html.Div([ #criar lista que permite ao utilizador escolher as métricas das previsões a ver
        html.Label("Selecione as Métricas:"),
        dcc.Checklist(
            id = 'metricas',
            options = [{'label': i, 'value': i} for i in available_metrics],
            value = ['Erro Médio Absoluto'],
            labelStyle = {'display':'block'}
        ),
    ], style = {'vertical-align':'top', 'display': 'inline-block', 'width': '30%', 'font-family':'Arial'}),
    html.Div(
        id='metricas_resultados')
    ], style={'text-align':'center', 'margin-top': '20px'})
])        

#-------------------------------------    
    
#ATUALIZAR OS DADOS COM BASE NAS OPÇÕES SELECIONADAS
@app.callback( 
    dash.dependencies.Output('dados_resultados', 'children'),
    [dash.dependencies.Input('dados_meses', 'value'),
     dash.dependencies.Input('dados_variaveis', 'value'),
     dash.dependencies.Input('dados_visualizacao', 'value')])

#-------------------------------------    
    
#DESENHAR RESULTADOS COM BASE NAS OPÇÕES SELECIONADAS
def update_graph(selected_months, selected_variables, graph_choice):
    if graph_choice == 'Gráfico': #desenhar um gráfico para cada variável
        colors = ['#069EE1', '#E1A806', '#42545E']
        graphs = []
        for i, variable in enumerate(selected_variables):
            filtered_data = data[data['Mês'].isin(selected_months)]
            fig = px.line(filtered_data, x=filtered_data.index, y=variable, color='Mês', color_discrete_sequence=colors)
            fig.update_layout(xaxis_title='Data', font=dict(family='Arial'), plot_bgcolor='white', paper_bgcolor='white', xaxis=dict(gridcolor='lightgrey', gridwidth=1), yaxis=dict(gridcolor='lightgrey', gridwidth=1))
            graph = dcc.Graph(figure=fig)
            graphs.append(graph)
        return html.Div(graphs, style={'maxHeight':'500px', 'overflowY':'scroll', 'maxWidth': '90%', 'margin-top': '20px', 'margin':'auto'})
    elif graph_choice == 'Tabela': #desenhar tabela
        filtered_data = data[data['Mês'].isin(selected_months)]
        filtered_data = filtered_data[selected_variables]
        table = html.Div([
            html.Table([
                html.Tr([html.Th('Data', style={'borderBottom':'2px solid black', 'text-align':'center'})] +
                        [html.Th(col, style={'borderBottom':'2px solid black', 'text-align':'center'}) for col in filtered_data.columns]),
                *[html.Tr([html.Td(filtered_data.index[i], style={'borderBottom':'1px solid #dddddd', 'text-align':'center'})] +
                          [html.Td(filtered_data.iloc[i][col], style={'borderBottom':'1px solid #dddddd', 'text-align':'center'}) for col in filtered_data.columns])
                  for i in range(len(filtered_data))]
            ]), 
        ], style={'maxHeight':'500px', 'overflowY':'scroll', 'margin-top': '50px', 'maxWidth': '90%', 'margin':'auto', 'font-family':'Arial'})
        return table
    elif graph_choice == 'Outliers':
        colors = ['#069EE1', '#E1A806', '#42545E']
        box_plots = []
        for variable in selected_variables:
            filtered_data = data[data['Mês'].isin(selected_months)]
            fig = px.box(filtered_data, x=filtered_data.index.month_name(), y=variable, color='Mês', color_discrete_sequence=colors)
            fig.update_layout(font=dict(family='Arial'), plot_bgcolor='white', paper_bgcolor='white', xaxis=dict(title='Mês', gridcolor='lightgrey', gridwidth=1), yaxis=dict(title=variable, gridcolor='lightgrey', gridwidth=1))
            graph = dcc.Graph(figure=fig)
            box_plots.append(graph)
        return html.Div(box_plots, style={'maxHeight':'500px', 'overflowY':'scroll', 'maxWidth': '90%', 'margin-top': '20px', 'margin':'auto'})

#-------------------------------------   

#ATUALIZAR OS MÉTODOS COM BASE NAS OPÇÕES SELECIONADAS
@app.callback( 
    dash.dependencies.Output('metodos_resultados', 'children'),
    [dash.dependencies.Input('metodos_variaveis', 'value'),
     dash.dependencies.Input('metodos', 'value'),
     dash.dependencies.Input('metodos_visualizacao', 'value')])

#-------------------------------------  

#DESENHAR RESULTADOS COM BASE NAS OPÇÕES SELECIONADAS
def update_methods(selected_variables, selected_method, visualization_choice):
    if selected_variables == 'Dia de Semana, Hora, Consumo  da Hora Anterior, Temperatura': #selecionar o método a usar
        outputs = outputs_3
        xx = data_3.index
        predictions = None
        if selected_method == 'Regressor de Floresta Aleatória':
            predictions = predictions_RFR_3
        elif selected_method == 'Regressor de Árvore de Decisão':
            predictions = predictions_DTR_3
        elif selected_method == 'Regressor de Impulso Gradiente':
            predictions = predictions_GBR_3
        elif selected_method == 'Regressor de Bagging':
            predictions = predictions_BR_3
        elif selected_method == 'Regressor de Perceptron de Múltiplas Camadas':
            predictions = predictions_MLPR_3
        elif selected_method == 'Regressão Linear':
            predictions = predictions_LR_3
        errors = np.abs(outputs - predictions) / outputs * 100 #calcular erros
    elif selected_variables == 'Dia de Semana, Hora, Consumo  da Hora Anterior, Temperatura, Taxa de Humidade, Radiação Solar': #selecionar o método a usar
        outputs = outputs_5
        xx = data_5.index
        predictions = None
        if selected_method == 'Regressor de Floresta Aleatória':
            predictions = predictions_RFR_5
        elif selected_method == 'Regressor de Árvore de Decisão':
            predictions = predictions_DTR_5
        elif selected_method == 'Regressor de Impulso Gradiente':
            predictions = predictions_GBR_5
        elif selected_method == 'Regressor de Bagging':
            predictions = predictions_BR_5
        elif selected_method == 'Regressor de Perceptron de Múltiplas Camadas':
            predictions = predictions_MLPR_5
        elif selected_method == 'Regressão Linear':
            predictions = predictions_LR_5
        errors = np.abs(outputs - predictions) / outputs * 100 #calcular erros
    if visualization_choice == 'Gráfico': #desenhar gráficol
        fig_consumo = go.Figure()
        fig_consumo.add_trace(go.Scatter(x=xx, y=outputs, mode='lines', name='Consumo Real', line=dict(color='#069EE1')))
        fig_consumo.add_trace(go.Scatter(x=xx, y=predictions, mode='lines', name='Consumo Previsto', line=dict(color='#E1A806')))
        fig_consumo.update_layout(xaxis_title='Data', yaxis_title='Consumo (kWh)', font=dict(family='Arial'), plot_bgcolor='white', paper_bgcolor='white', xaxis=dict(gridcolor='lightgrey', gridwidth=1), yaxis=dict(gridcolor='lightgrey', gridwidth=1))
        graph_consumo = dcc.Graph(figure=fig_consumo)
        fig_error = go.Figure()
        fig_error.add_trace(go.Scatter(x=xx, y=errors, mode='lines', line=dict(color='#069EE1')))
        fig_error.update_layout(xaxis_title='Data', yaxis_title='Erro Relativo (%)', font=dict(family='Arial'), plot_bgcolor='white', paper_bgcolor='white', xaxis=dict(gridcolor='lightgrey', gridwidth=1), yaxis=dict(gridcolor='lightgrey', gridwidth=1))
        graph_error = dcc.Graph(figure=fig_error)
        return html.Div([html.Div(graph_consumo), html.Div(graph_error)], style={'maxHeight':'500px', 'overflowY':'scroll', 'maxWidth': '90%', 'margin-top': '20px', 'margin':'auto'})
    elif visualization_choice == 'Tabela': #desenhar tabela
        table_data = {'Data': xx, 'Consumo Real (kWh)': outputs, 'Consumo Previsto (kWh)': predictions, 'Erro Relativo (%)': errors}
        table_df = pd.DataFrame(table_data)
        table = html.Div([
            html.Table([
                html.Tr([html.Th(col, style={'borderBottom':'2px solid black', 'text-align':'center'}) for col in table_df.columns]),
                *[html.Tr([html.Td(table_df.iloc[i][col], style={'borderBottom':'1px solid #dddddd', 'text-align':'center'}) for col in table_df.columns])
                  for i in range(len(table_df))]
            ]), 
        ], style={'maxHeight':'500px', 'overflowY':'scroll', 'margin-top': '50px', 'maxWidth': '90%', 'margin':'auto', 'font-family':'Arial'})
        return table

#-------------------------------------   
 
#ATUALIZAR AS MÉTRICAS COM BASE NAS OPÇÕES SELECIONADAS
@app.callback( 
    dash.dependencies.Output('metricas_resultados', 'children'),
    [dash.dependencies.Input('metricas', 'value'),
     dash.dependencies.Input('metricas_variaveis', 'value'),
     dash.dependencies.Input('metricas_metodos', 'value')])

#-------------------------------------  

#DESENHAR RESULTADOS COM BASE NAS OPÇÕES SELECIONADAS
def update_metrics(selected_metrics, selected_variables, selected_methods):
    if selected_variables == 'Dia de Semana, Hora, Consumo  da Hora Anterior, Temperatura': #selecionar o método a usar
        metrics = metrics_3
    elif selected_variables == 'Dia de Semana, Hora, Consumo  da Hora Anterior, Temperatura, Taxa de Humidade, Radiação Solar': #selecionar o método a usar
        metrics = metrics_5
    filtered_metrics = metrics.loc[selected_methods, selected_metrics]
    table = html.Div([ #desenhar tabela
        html.Table([
            html.Tr([html.Th('Método', style={'borderBottom':'2px solid black', 'text-align':'center'})] +
                    [html.Th(col, style={'borderBottom':'2px solid black', 'text-align':'center'}) for col in filtered_metrics.columns]),
            *[html.Tr([html.Td(filtered_metrics.index[i], style={'borderBottom':'1px solid #dddddd', 'text-align':'center'})] +
                      [html.Td(filtered_metrics.iloc[i][col], style={'borderBottom':'1px solid #dddddd', 'text-align':'center'}) for col in filtered_metrics.columns])
              for i in range(len(filtered_metrics))]
        ], style={'marginTop': '50px'}), 
    ], style={'maxHeight':'500px', 'overflowY':'scroll', 'maxWidth': '90%', 'margin':'auto', 'font-family':'Arial', 'margin-bottom':'50px'})    
    return table
    
#-------------------------------------

#CORRER O DASHBOARD
if __name__ == '__main__':
    app.run_server(debug=False)
