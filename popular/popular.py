import json

def popular(CORE):
    print('--- POPULAR ---')
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

    chanells, scotes = CORE.get_recom('0', top=top)
    print('--- RECOM ---', chanells)
    answer = {'status': 'ok', 'popular': chanells}
    return json.dumps(answer)