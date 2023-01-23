import os
import time
import datetime
import requests
import pandas as pd
import numpy as np
from typing import Dict, Text
# from pymongo import MongoClient

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs


class ModelTfrs():
    def __init__(self, config):
        self.config = config  # Данные из конфигурационного файла
        self.data_duration = []  # Временное хранилище списка получаемых данных
        self.df_duration = None  # Тут размещаем данные 'duration' в формате Pandas, полученные результате работы метода __get_data_duration
        self.df_website = None  # Тут размещаем данные 'website' в формате Pandas полученные результате работы метода __get_data_website
        self.log = []  # Лог операций
        self.model = None  # Тут будет храниться модель
        self.run_time = 0
        self.files_path = self.config.path + '/ModelTfrs'
        self.popular_list = []  # Тут будет храниться список 100 популярных каналов

        # Создаём папку для файлов модели
        if not os.path.isdir(self.files_path):
            os.makedirs(self.files_path, mode=0o755, exist_ok=True)


    # === ОБУЧЕНИЕ МОДЕЛИ ===
    def fit(self, days=30):
        # Удаляем файл логов
        if os.path.isfile(self.files_path + '/status.log'): 
            os.remove(self.files_path + '/status.log')

        self.run_time = 0  # Обнуляем время выполнения

        # Получаем данные `duration` за указанное количество дней `days`
        answer = self.__get_data_duration(days)
        if answer['status'] != 'OK':
            return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнени

        # Получаем данные `website` за указанное количество дней `days`
        # self.__get_data_website(days) 

        # Обработка данных
        answer = self.__data_processing()
        if answer['status'] != 'OK':
            return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение

        # Обучение модели
        answer = self.__fit_model()
        if answer['status'] != 'OK':
            return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение

        return answer


    # Предсказание - переопределяем в наследуемом классе
    def predict(self, user_id):
        # Создаём модель, которая использует необработанные функции запроса, и
        index = tfrs.layers.factorized_top_k.BruteForce(self.model.user_model)
        # рекомендует фильмы из всего набора данных фильмов.
        index.index_from_dataset(
            tf.data.Dataset.zip((self.chanells_tensor.batch(10000), self.chanells_tensor.batch(10000).map(self.model.movie_model)))
        )

        # Получить рекомендации для пользователя с индексом user_id, например: "38082".
        _, titles = index(tf.constant([user_id]))
        answer = titles.numpy().astype('str').tolist()

        return answer[0]


    # === ПОЛУЧЕНИЕ ДАННЫХ "DURATION" ПО API ЗА НУЖНОЕ КОЛИЧЕСТВО ДНЕЙ ===
    def __get_data_duration(self,  days=30):

        start_time = time.time()
        from_time = int(time.time()) - int(days)*86400
        from_time_url = f'&from={from_time}'
        to_time = int(time.time())
        from_to_time = f'&to={to_time}'

        # --- Соединение с GG Api и получение данных ---
        for i in range(1, 1000):
            url = self.config.gg_api_url + from_time_url + from_to_time + '&page=' + str(i)
            try:
                api_req = requests.get(url)
                if int(api_req.status_code) != 200:
                    answer = {
                        'status': 'ERROR', 
                        'message': f'Error connecting to goodgame.ru API. Server API response code: {api_req.status_code}'
                    }
                    self.__logging(answer)
                    break
                else:
                    # Добавляем данные в наш список
                    data_list = api_req.json()
                    self.data_duration.extend(data_list)
                    # print(i, len(self.data_duration), len(data_list))
                    if len(data_list) < self.config.gg_pagination_step:  # Шаг пагинации
                        self.df_duration = pd.DataFrame.from_dict(self.data_duration)
                        break
            except:
                answer = {
                    'status': 'ERROR', 
                    'messare': 'Error connecting to goodgame.ru API'
                }
                self.__logging(answer)
                break

        # --- Формирование ответа и запись в лог ---
        delta_time = time.time() - start_time
        if (len(self.data_duration) > 0):
            answer = {
                'status': 'OK', 
                'message': f'Data received, number of rows: {len(self.data_duration)}, execution time: {round(delta_time, 4)}s'
            }
            self.__logging(answer)
        else:
            answer = {
                'status': 'ERROR', 
                'message': f'No data, execution time: {round(delta_time, 4)}s'
            }
            self.__logging(answer)
        self.data_duration = []  # Очищаем буфер данных

        self.run_time += delta_time

        return answer 



    # === ОБРАБОТКА ДАННЫХ ===
    def __data_processing(self):
        start_time = time.time()

        df = self.df_duration
        df.dropna(inplace=True)

        # --- Находим самые популярные каналы - составляем список из 100 каналов ---
        self.popular_list = df.groupby(['channel_id']).size().sort_values(ascending=False)[0:100].tolist()

        # --- Удаляем стримы с небольшим количество просмотров (20% от среднего) ---
        # Находим среднее число просмотров стрима
        view_mean = df.groupby(['channel_id']).size().mean()

        # Создадим новый датасет только с подсчетом рейтинга, чтобы исключить некоторые
        view_counter = pd.DataFrame({'Count' : df.groupby(['channel_id']).size()}).reset_index()
        k_1 = view_counter.shape[0]

        view_counter = view_counter.loc[view_counter['Count'] > view_mean/5]  # 1/5
        k_2 = view_counter.shape[0]

        reducer = df['channel_id'].isin(view_counter['channel_id'])
        df_2 = df[reducer]

        self.df_duration = df_2.drop_duplicates()

        delta_time = time.time() - start_time
        self.run_time += delta_time
        answer = {
            'status': 'OK', 
            'message': f'Data processed, number of rows: {self.df_duration.shape[0]}, execution time: {round(delta_time, 4)}s'
        }
        self.__logging(answer)

        return answer


    # === ЗАПИСЬ В ЛОГ ===
    def __logging(self, answer):
        with open(self.files_path + '/status.log', 'a') as file:
            now = datetime.datetime.now()
            d = f'{now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second}'
            text = f"{d}, {answer['status']}: {answer['message']} \n"
            file.write(text)



    # === ОБУЧЕНИЕ МОДЕЛИ ===
    def __fit_model(self):
        start_time = time.time()

        # --- Получаем tf датасет уникальных пользователей ---
        users_id_arr = self.df_duration['user_id'].unique().astype('bytes')

        # --- Получаем tf датасет уникальных каналов ---
        chanells_id_arr = self.df_duration['channel_id'].unique().astype('bytes')
        self.chanells_tensor = tf.data.Dataset.from_tensor_slices(chanells_id_arr)

        # --- Получаем tf датасет ratings ---
        ratings_ser = self.df_duration['user_id']
        df_ratings = pd.DataFrame(ratings_ser)
        df_ratings['channel_id'] = self.df_duration['channel_id']
        ratings_b_arr = df_ratings.to_numpy().astype('bytes')
        ratings = tf.data.Dataset.from_tensor_slices({"channel_id": ratings_b_arr[:, 1], "user_id": ratings_b_arr[:, 0]})

        # --- Перемешиваем данные ---
        tf.random.set_seed(42)
        shuffled = ratings.shuffle(df_ratings.shape[0], seed=42, reshuffle_each_iteration=False)

        # --- Рассчитываем длинну train и test данных в зависимости от длинны выбоки ---
        if len(shuffled) > 6000 :
            train_num = 5000
            test_num = 1000
        else:
            train_num = round(len(shuffled) * 0.8)
            test_num = len(shuffled) - train_num

        # --- Разбиваем выборку на учебную и тестовую ---
        train = shuffled.take(train_num)
        test = shuffled.skip(train_num).take(test_num)

        # --- РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА ---
        embedding_dimension = 64  # Размер вектора сжатого признакового пространства

        # - Башня запросов -
        user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=users_id_arr, mask_token=None),
            tf.keras.layers.Embedding(len(users_id_arr) + 1, embedding_dimension)
        ])

        # - Башня кандидатов -
        movie_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=chanells_id_arr, mask_token=None),
            tf.keras.layers.Embedding(len(chanells_id_arr) + 1, embedding_dimension)
        ])

        # - Метрики -
        # Вычисляет показатели для K лучших кандидатов, обнаруженных моделью поиска
        metrics = tfrs.metrics.FactorizedTopK(
            # Применяем к батчу данных chanells_tensor нашу модель кандидата
            candidates = self.chanells_tensor.batch(10000).map(movie_model)
        )

        # - Ошибки -
        task = tfrs.tasks.Retrieval(
            metrics=metrics
        )

        self.model = CreateModel(user_model, movie_model, task)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

        cached_train = train.shuffle(100_000).batch(32).cache()
        cached_test = test.batch(16).cache()

        # Обучим модель
        history = self.model.fit(cached_train, epochs=18, verbose=0)

        delta_time = time.time() - start_time
        self.run_time += delta_time
        message =   'Model training completed, '
        message +=  f'top_1: {history.history["factorized_top_k/top_1_categorical_accuracy"][-1]:.4f}, '
        message +=  f'top_5: {history.history["factorized_top_k/top_5_categorical_accuracy"][-1]:.4f}, '
        message +=  f'top_10: {history.history["factorized_top_k/top_10_categorical_accuracy"][-1]:.4f}, '
        message +=  f'top_50: {history.history["factorized_top_k/top_50_categorical_accuracy"][-1]:.4f}, '
        message +=  f'top_100: {history.history["factorized_top_k/top_100_categorical_accuracy"][-1]:.4f}, '
        message +=  f'total execution time: {round(self.run_time)}s '
        answer = {
            'status': 'OK', 
            'message': message
        }
        self.__logging(answer)

        return answer





# ======= СОЗДАЁМ МОДЕЛЬ =======
class CreateModel(tfrs.Model):
    def __init__(self, user_model, movie_model, task):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task


    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # Выбираем пользовательские функции и передаем их в пользовательскую модель.
        user_embeddings = self.user_model(features["user_id"])
        # И выберите функции фильма и передайте их в модель фильма,
        # вернуть вложения.
        positive_movie_embeddings = self.movie_model(features["channel_id"])

        # Вычисляет потери и метрики.
        return self.task(user_embeddings, positive_movie_embeddings)