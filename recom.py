import json

# Получаем рекомендации для пользователя
def recom(CORE):
    # Проверяем и при необходимости корректируем входные данные
    if 'top' not in CORE.req:   
        top = CORE.config.top
    else:
        try:
            top = int(CORE.req['top'])
        except:
            answer = {'status': 'err', 'message' :'Bad secret "top" key'}
            return json.dumps(answer)

    if top < 1 or top > 100:
        top = CORE.config.top

    user_id = CORE.req['user_id']

    chanells = CORE.model.predict(user_id)  # Получаем данные по API
    answer = {'status': 'ok', 'user_id': user_id, 'recom': chanells[0:top]}

    return json.dumps(answer)