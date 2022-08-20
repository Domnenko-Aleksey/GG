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

    if top < 1 or top > 10:
        top = CORE.config.top


    user_id = CORE.req['user_id']
    chanells, scotes = CORE.get_recom(user_id, top=top)
    print('--- RECOM ---', chanells)
    answer = {'status': 'ok', 'user_id': user_id, 'recom': chanells}
    return json.dumps(answer)