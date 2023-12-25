import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Leitura dos dados do CSV
df = pd.read_csv("dataset.csv")

# Criando colunas de valores
df.insert(loc=1, column='IDADE_VALUE', value=df['IDADE'].map({'≤30': 0, '31...40': 1, '>40': 2}))
df.insert(loc=3, column='RENDA_VALUE', value=df['RENDA'].map({'Baixa': 0, 'Média': 1, 'Alta': 2}))
df.insert(loc=5, column='ESTUDANTE_VALUE', value=df['ESTUDANTE'].map({'Não': 0, 'Sim': 1}))
df.insert(loc=7, column='CREDITO_VALUE', value=df['CREDITO'].map({'Bom': 0, 'Excelente': 1}))

# Dados de treinamento
X = df[['IDADE_VALUE', 'RENDA_VALUE', 'ESTUDANTE_VALUE', 'CREDITO_VALUE']]
y = df['CLASSE']

# Divisão em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o classificador
clf = KNeighborsClassifier(n_neighbors=3)

# Treinando o modelo
clf.fit(X_treino, y_treino)

# Dados para predição
dados_pred = {
    'IDADE_VALUE': [0],
    'RENDA_VALUE': [1],
    'ESTUDANTE_VALUE': [1],
    'CREDITO_VALUE': [0]
}

# Criando um DataFrame para os dados de predição
dados_pred = pd.DataFrame(dados_pred)

# Fazendo a predição
predicoes = clf.predict(dados_pred)

# Exibindo a predição
print("Predição para os dados de predição:", predicoes)
