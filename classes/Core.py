import os
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

class Core:
    def __init__(self):
        self.db = False  # Объект подключения к БД
        self.req = False  # Запрос
        self.model = False  # Модель TFRS
        self.load_model()  # Загружаем модель TFRS


    # Загрузка модели TFRS
    def load_model(self):
        self.model = tf.saved_model.load('model')

    
    # Получить рекомендацию по id пользователя
    def get_recom(self, id, top=5):
        scores, chanells = self.model([str(id)])
        chanells = chanells.numpy().astype('str').tolist()[0][0:top]
        return chanells, scores

