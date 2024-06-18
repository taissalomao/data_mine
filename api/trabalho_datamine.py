import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def plotar_resultados(y_true, y_pred, title):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
        'Score': [accuracy, precision, recall, f1]
    })

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Metric', y='Score', data=metrics)
    plt.ylim(0, 1)
    plt.title(title)
    plt.show()

# Carregando o dataset
data = pd.read_csv('./web-page-phishing.csv')

# Separando features e target
X = data.drop('phishing', axis=1)
y = data['phishing']

# Treinando o modelo de Detecção de Novidades (Isolation Forest)
modelo_IF = IsolationForest(contamination=0.1, random_state=42).fit(X[y == 0])

# Analisando o comportamento de cada URL
anormalidades = modelo_IF.predict(X)

# Marcando URLs anormais como potenciais phishing
X['anormalidade'] = anormalidades

# Treinando o modelo de Modelagem Preditiva (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X.drop('anormalidade', axis=1), y, test_size=0.2, random_state=42)
modelo_RF = RandomForestClassifier(random_state=42).fit(X_train, y_train)

# Salvando os modelos
joblib.dump(modelo_IF, 'modelo_IF.pkl')
joblib.dump(modelo_RF, 'modelo_RF.pkl')

# Fazendo a predição para o conjunto de teste
y_pred = modelo_RF.predict(X_test)

# Avaliando a performance do modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Precisão:", precision_score(y_test, y_pred))
print("Revocação:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

# Integrando as estratégias

# Predição inicial com o modelo de Modelagem Preditiva
predicao_RF = modelo_RF.predict(X.drop('anormalidade', axis=1))

# Refinando a predição com base na Detecção de Novidades
predicao_final = predicao_RF.copy()
predicao_final[X['anormalidade'] == -1] = 1

# Avaliando a performance da estratégia integrada
print("\nAcurácia (estratégia integrada):", accuracy_score(y, predicao_final))
print("Precisão (estratégia integrada):", precision_score(y, predicao_final))
print("Revocação (estratégia integrada):", recall_score(y, predicao_final))
print("F1-score (estratégia integrada):", f1_score(y, predicao_final))

plotar_resultados(y, predicao_final, 'Estratégia Integrada')

# def extrair_features(url):
#     features = {
#         'url_length': len(url),
#         'n_dots': url.count('.'),
#         'n_hypens': url.count('-'),
#         'n_underline': url.count('_'),
#         'n_slash': url.count('/'),
#         'n_questionmark': url.count('?'),
#         'n_equal': url.count('='),
#         'n_at': url.count('@'),
#         'n_and': url.count('&'),
#         'n_exclamation': url.count('!'),
#         'n_space': url.count(' '),
#         'n_tilde': url.count('~'),
#         'n_comma': url.count(','),
#         'n_plus': url.count('+'),
#         'n_asterisk': url.count('*'),
#         'n_hastag': url.count('#'),
#         'n_dollar': url.count('$'),
#         'n_percent': url.count('%'),
#         'n_redirection': url.count('//'),
#     }
    
#     return features

# # Carregue os modelos treinados
# modelo_IF = joblib.load('modelo_IF.pkl')
# modelo_RF = joblib.load('modelo_RF.pkl')

# # Função para classificar URL
# def classificar_url(url):
#     features = extrair_features(url)
#     data = pd.DataFrame(features, index=[0])

#     anormalidade = modelo_IF.predict(data)
#     if anormalidade == -1:
#         predicao = 1
#     else:
#         predicao = modelo_RF.predict(data)

#     return predicao

# # Exemplo de uso
# url = "https://cutt.ly/fgbet2" 
# predicao = classificar_url(url)

# if predicao == 1:
#     print(f"{url} - Phishing")
# else:
#     print(f"{url} - Legítimo")
