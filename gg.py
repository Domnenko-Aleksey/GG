import sys
import json
from flask import Flask, request, abort, jsonify
sys.path.append('classes')
import config
from Core import Core
import fit  # Обучает модель
import recom  # Выдаёт рекомендации
import popular  # Выдаёт 100 популярных каналов в порядке убывания популярности
# import popular  # Выдаёт популярные стримы

app = Flask(__name__)

CORE = Core()  # Инициируем класс, загружаем модель
CORE.config = config

@app.route("/", methods=['POST', 'GET'])
def route_model():
    CORE.req = request.values
    # Проверка наличия ключа
    if 'key' not in CORE.req or 'act' not in CORE.req:
        answer = {'status': 'err', 'message': 'Bad request'}
        return json.dumps(answer)
    
    # Проверка значения ключа
    if CORE.req['key'] != CORE.config.model_key:
        answer = {'status': 'err', 'message': 'Bad secret key'}
        return json.dumps(answer)

    acts_dict = {
        'recom': recom.recom, 
        'popular': popular.popular,
        'fit': fit.fit,
    }

    if CORE.req['act'] not in acts_dict:
    	abort(404)

    resp = acts_dict[CORE.req['act']](CORE)

    print('RESPONSE:', resp)

    return resp

    # response = jsonify(resp)  # {'some': 'data'}
    # response.headers.add('Access-Control-Allow-Origin', '*')
    # response.headers.add('Access-Control-Allow-Methods', 'GET')
    # return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
