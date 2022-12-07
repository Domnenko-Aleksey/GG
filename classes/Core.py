import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs


class Core:
    def __init__(self):
        self.db = None  # Объект подключения к БД
        self.req = None  # Запрос
        self.model = None  # Тут будем хранить модель

    
    '''
    # Получить рекомендацию по id пользователя
    def get_recom(self, id, top=5):
        scores, chanells = self.model([str(id)])
        chanells = chanells.numpy().astype('str').tolist()[0][0:top]
        return chanells, scores


    # ======= РАБОТА С БАЗАМИ ДАННЫХ ======= 
    # Получаем данные по просмотру каналов за установленный период
    def get_data(self, period=30):
        last_time = time.time() + period * 60 * 60 * 24
        sql =   "SELECT id, chanell_id,	user_id	"
        sql +=  "FROM duration*** "
        sql +=  "WHERE chanell_id != '' AND user_id != '' "
        sql +=  "AND time > %s"
        self.db.execute(sql, (last_time))
        return self.db.fetchall()
    '''
