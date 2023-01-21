import json

# Получаем рекомендации для пользователя
def popular(CORE):
    # Проверяем и при необходимости корректируем входные данные
    answer = {'status': 'ok', 'popular': CORE.model.popular_list}

    return json.dumps(answer)