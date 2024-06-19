import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route('/')
def index():
    return render_template('index.html')

# Função para extrair features de uma URL
def extrair_features(url):
    features = {
        'url_length': len(url),
        'n_dots': url.count('.'),
        'n_hypens': url.count('-'),
        'n_underline': url.count('_'),
        'n_slash': url.count('/'),
        'n_questionmark': url.count('?'),
        'n_equal': url.count('='),
        'n_at': url.count('@'),
        'n_and': url.count('&'),
        'n_exclamation': url.count('!'),
        'n_space': url.count(' '),
        'n_tilde': url.count('~'),
        'n_comma': url.count(','),
        'n_plus': url.count('+'),
        'n_asterisk': url.count('*'),
        'n_hastag': url.count('#'),
        'n_dollar': url.count('$'),
        'n_percent': url.count('%'),
        'n_redirection': url.count('//'),
    }
    return features

# Caminho dos modelos treinados
modelo_if_path = os.path.join(os.path.dirname(__file__), 'modelo_IF.pkl')
modelo_rf_path = os.path.join(os.path.dirname(__file__), 'modelo_RF.pkl')

modelo_IF = joblib.load(modelo_if_path)
modelo_RF = joblib.load(modelo_rf_path)

# Função para classificar URL
def classificar_url(url):
    
    features = extrair_features(url)
    data = pd.DataFrame(features, index=[0])
    anormalidade = modelo_IF.predict(data)
    # Se a URL for anormal, classifique como phishing
    if anormalidade == -1:
        predicao = 1
    else:
        # Preveja usando o modelo Random Forest
        predicao = modelo_RF.predict(data)
    # Retorne a classificação
    return predicao

@app.route('/classify', methods=['POST'])
def classify_url():
    if request.is_json:  
        json_data = request.json
        if 'url' in json_data:  
            url = json_data['url']
            predicao = classificar_url(url)
            if predicao == 1:
                resultado = {'url': url, 'phishing': True}
            else:
                resultado = {'url': url, 'phishing': False}
            return jsonify(resultado)
        else:
            return jsonify({'error': 'No URL provided in JSON data'})
    elif 'file' in request.files:  
        file = request.files['file']
        urls = file.read().decode('utf-8').splitlines()
        resultados = []
        for url in urls:
            predicao = classificar_url(url)
            if predicao == 1:
                resultados.append({'url': url, 'phishing': True})
            else:
                resultados.append({'url': url, 'phishing': False})
        return jsonify(resultados)
    else:
        return jsonify({'error': 'No JSON data or file provided'})

if __name__ == '__main__':
    app.run(debug=True)
