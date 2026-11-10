import pandas as pd
import numpy as np

def calcular_regressao_imoveis(caminho_arquivo):
    dados = pd.read_csv(caminho_arquivo, sep=';', skipinitialspace=True)
    
    dados.columns = dados.columns.str.strip()
    
    lista_areas = dados["area_util_m2"].to_list()
    lista_quartos = dados["quantidade_quartos"].to_list()
    
    coluna_preco_limpa = dados["preco_venda"].astype(str)
    coluna_preco_limpa = coluna_preco_limpa.str.replace("R$", "", regex=False)
    coluna_preco_limpa = coluna_preco_limpa.str.replace(".", "", regex=False)
    coluna_preco_limpa = coluna_preco_limpa.str.replace(",", ".", regex=False).str.strip()
    
    dados["preco_venda"] = coluna_preco_limpa.astype(float)
    lista_precos = dados["preco_venda"].to_list()

    coluna_uns = np.ones(len(lista_areas))
    
    X = np.column_stack((coluna_uns, lista_areas, lista_quartos))
    y = np.array(lista_precos)

    matriz_quadrada = X.T @ X
    matriz_inversa = np.linalg.inv(matriz_quadrada)
    coeficientes_beta = matriz_inversa @ X.T @ y

    return coeficientes_beta.tolist()

coeficientes_finais = calcular_regressao_imoveis("imoveis_lar_perfeito.csv")
print(coeficientes_finais)
