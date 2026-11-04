import pandas as pd

df = pd.read_csv("day.csv")

print("========== BASE DE DADOS ==========")
print(df)

print("\n========== INFORMAÇÕES ==========")
df.info()

print("\n========== ESTATÍSTICAS ==========")
print(df.describe())


from scipy.stats import norm

print("\n========== EXERCÍCIO 02 ==========")

media = df["cnt"].mean()
mediana = df["cnt"].median()
desvio = df["cnt"].std(ddof=0)

# a) Probabilidade de ser menor que a média
a = norm.cdf(media, media, desvio)

# b) Probabilidade de ser maior que a mediana
b = norm.sf(mediana, media, desvio)

print(f"Média: {media:.2f}")
print(f"Mediana: {mediana:.2f}")
print(f"Desvio padrão: {desvio:.2f}")

print(f"\na) Probabilidade: {a:.4f} ({a:.2%})")
print(f"b) Probabilidade: {b:.4f} ({b:.2%})")


from math import sqrt
from scipy.stats import norm

print("\n========== EXERCÍCIO 03 ==========")

dados = df["cnt"]

media_pop = dados.mean()
desvio_pop = dados.std(ddof=0)

# a) Teste de hipótese - média menor
amostra_a = dados.sample(n=20, random_state=42)
media_a = amostra_a.mean()

zc_a = (media_a - media_pop) / (desvio_pop / sqrt(20))
za_a = norm.ppf(0.06)

print("\n========== ITEM A ==========")
print("Média populacional:", media_pop)
print("Média amostral:", media_a)
print("Z calculado:", zc_a)
print("Z crítico:", za_a)

if zc_a < za_a:
    print("Decisão: Rejeita H0")
else:
    print("Decisão: Não rejeita H0")


# b) Teste de hipótese - média maior
amostra_b = dados.sample(n=25, random_state=43)
media_b = amostra_b.mean()

zc_b = (media_b - media_pop) / (desvio_pop / sqrt(25))
za_b = norm.ppf(1 - 0.09)

print("\n========== ITEM B ==========")
print("Média populacional:", media_pop)
print("Média amostral:", media_b)
print("Z calculado:", zc_b)
print("Z crítico:", za_b)

if zc_b > za_b:
    print("Decisão: Rejeita H0")
else:
    print("Decisão: Não rejeita H0")


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

print("\n========== EXERCÍCIO 04 ==========")

# a) Gráfico de dispersão
sns.scatterplot(x="temp", y="cnt", data=df)
plt.title("Dispersão: Temperatura x Aluguéis")
plt.xlabel("Temperatura Normalizada")
plt.ylabel("Quantidade de Bicicletas Alugadas")
plt.show()

# b) Covariância e Correlação de Pearson
covariancia = np.cov(df["temp"], df["cnt"])[0, 1]
correlacao = np.corrcoef(df["temp"], df["cnt"])[0, 1]

print("\nCovariância:", covariancia)
print("Correlação de Pearson:", correlacao)

# c) Gráfico de Correlação de Pearson
sns.heatmap(df[["temp", "cnt"]].corr(), annot=True, cmap="coolwarm")
plt.title("Gráfico de Correlação Linear de Pearson")
plt.show()

# d) Regressão Linear Simples
X = sm.add_constant(df["temp"])
modelo = sm.OLS(df["cnt"], X).fit()

sns.regplot(x="temp", y="cnt", data=df, ci=None, line_kws={"color": "red"})
plt.title("Dispersão com Reta de Regressão")
plt.xlabel("Temperatura Normalizada")
plt.ylabel("Quantidade de Bicicletas Alugadas")
plt.show()

# Equação da regressão
b0, b1 = modelo.params

print("\nEquação da Regressão Linear Simples:")
print(f"Y = {b0:.4f} + ({b1:.4f} * X)")

# e) Adequação do modelo
print("\n========== ADEQUAÇÃO DO MODELO ==========")
print(modelo.summary())

print("R²:", modelo.rsquared)
print("Teste F:", modelo.fvalue)
print("P-valor do Teste F:", modelo.f_pvalue)

# f) Previsão
nova_temperatura = 0.5

previsao = b0 + b1 * nova_temperatura

print("\n========== PREVISÃO ==========")
print("Temperatura normalizada:", nova_temperatura)
print(f"Quantidade prevista: {previsao:.0f} bicicletas")


print("\n========== EXERCÍCIO 05 ==========")

# Seleção das variáveis quantitativas
dados = df[["temp", "hum", "windspeed", "cnt"]]

# a) Gráfico de dispersão
sns.pairplot(dados, diag_kind=None)
plt.show()

# b) Matrizes de Covariância e Correlação
print("\n========== MATRIZ DE COVARIÂNCIA ==========")
print(dados.cov())

print("\n========== MATRIZ DE CORRELAÇÃO ==========")
print(dados.corr())

# c) Gráfico de Correlação de Pearson
sns.heatmap(dados.corr(), annot=True, cmap="coolwarm")
plt.title("Gráfico de Correlação Linear de Pearson")
plt.show()

# d) Regressão Linear Múltipla
modelo_multiplo = sm.OLS(
    dados["cnt"], sm.add_constant(dados[["temp", "hum", "windspeed"]])
).fit()

print("\n========== REGRESSÃO LINEAR MÚLTIPLA ==========")
print(modelo_multiplo.summary())

# Equação da regressão
b0, b1, b2, b3 = modelo_multiplo.params

print("\nEquação da Regressão Linear Múltipla:")
print(f"Y = {b0:.4f} + ({b1:.4f} * X1)" f" + ({b2:.4f} * X2) + ({b3:.4f} * X3)")

# e) Adequação do modelo
print("\n========== ADEQUAÇÃO DO MODELO ==========")
print("R²:", modelo_multiplo.rsquared)
print("Teste F:", modelo_multiplo.fvalue)
print("P-valor do Teste F:", modelo_multiplo.f_pvalue)

# f) Previsão utilizando três variáveis explicativas
nova_temperatura = 0.5
nova_umidade = 0.6
novo_vento = 0.2

previsao_multipla = b0 + b1 * nova_temperatura + b2 * nova_umidade + b3 * novo_vento

print("\n========== PREVISÃO ==========")
print("Temperatura:", nova_temperatura)
print("Umidade:", nova_umidade)
print("Velocidade do vento:", novo_vento)

print(f"Quantidade prevista: {previsao_multipla:.0f} bicicletas")


from sklearn.metrics import confusion_matrix, accuracy_score

print("\n========== EXERCÍCIO 06 ==========")

# Criando a variável dependente binária
mediana = df["cnt"].median()

df["demanda_alta"] = (df["cnt"] > mediana).astype(int)

# a) Regressão Logística Binária
X = sm.add_constant(df[["temp", "hum", "windspeed"]])
y = df["demanda_alta"]

modelo_logistico = sm.Logit(y, X).fit()

print("\n========== MODELO LOGÍSTICO ==========")
print(modelo_logistico.summary())

# Equação da Regressão Logística
b0, b1, b2, b3 = modelo_logistico.params

print("\nEquação da Regressão Logística Binária:")
print(
    f"P(Y=1) = 1 / (1 + e^-("
    f"{b0:.4f} + ({b1:.4f} * X1)"
    f" + ({b2:.4f} * X2)"
    f" + ({b3:.4f} * X3)))"
)

# b) Matriz de Confusão e Acurácia
probabilidades = modelo_logistico.predict(X)

classes = (probabilidades > 0.5).astype(int)

matriz = confusion_matrix(y, classes)
acuracia = accuracy_score(y, classes)

print("\n========== MATRIZ DE CONFUSÃO ==========")
print(matriz)

print("\n========== ACURÁCIA ==========")
print(f"Acurácia: {acuracia:.2%}")

# c) Classificação de uma nova observação
novo = pd.DataFrame({"const": [1], "temp": [0.5], "hum": [0.6], "windspeed": [0.2]})

prob = modelo_logistico.predict(novo)
classe_nova = (prob > 0.5).astype(int)

print("\n========== CLASSIFICAÇÃO ==========")
print(f"Probabilidade de demanda alta: {prob.iloc[0]:.2%}")
print("Classe prevista:", classe_nova.iloc[0])

if classe_nova.iloc[0] == 1:
    print("Classificação: Demanda Alta")
else:
    print("Classificação: Demanda Baixa")
