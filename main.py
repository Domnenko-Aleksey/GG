import sys
import json
from flask import Flask, request, abort
sys.path.append('classes')
import config
from Core import Core
from recom import recom
from popular import popular

app = Flask(__name__)

CORE = Core()  # Инициируем класс, загружаем модель
CORE.config = config

@app.route("/", methods=['POST', 'GET'])
def route_model():

    CORE.req = request.values
    # Проверка наличия ключа
    if 'key' not in CORE.req or 'act' not in CORE.req:
        answer = {'status': 'err', 'message' :'Bad request'}
        return json.dumps(answer)
    
    # Проверка значения ключа
    if CORE.req['key'] != CORE.config.secret_key:
        answer = {'status': 'err', 'message' :'Bad secret key'}
        return json.dumps(answer)

    acts_dict = {
        'recom': recom.recom, 
        'popular': popular.popular,
    }

    if CORE.req['act'] not in acts_dict:
    	abort(404)

    print('-------', request.values)
    print('=====', request.values['act'])
    # answer = {'status': 'ok', 'message' :''}
    # return json.dumps(answer)

    return acts_dict[CORE.req['act']](CORE)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
