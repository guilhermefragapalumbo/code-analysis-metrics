from flask import Flask, jsonify, request
from flask_cors import CORS
import callModelo

app = Flask(__name__)
CORS(app)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


@app.route('/', methods=['POST'])
def receive_data():
    data = request.get_json()
    #print(data)
    result = callModelo.predict_price(data)
    result = result.astype(float)
    result_list = result.tolist()  # converte para lista
    response = {'price': result_list}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='localhost',port=8081)

