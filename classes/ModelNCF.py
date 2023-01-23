from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import time
import requests 

#кластеризация
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#tf/keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
import os


class ModelNCF():
  def __init__(self, config):
        # print('MODEL NFC -> INIT')
        #super(ModelNCF, self).__init__(**kwargs)
        self.config = config  # Данные из конфигурационного файла
        self.data_duration = []  # Временное хранилище списка получаемых данных
        self.df_duration = False  # Данные 'duration' в формате Pandas
        self.df_website = False  # Данные 'website' в формате Pandas
        self.log = []  # Лог операций

        self.k = 10
        self.embedding_size = 50 #размер внутреннего пространства
        self.batch_size = 64 #размер batch_size
        self.epochs = 100 #кол-во эпох
        self.seed = 42
        self.path = '/content/drive/My Drive/internship/production'

        self.df_duration = None  # Тут размещаем данные 'duration' в формате Pandas, полученные результате работы метода __get_data_duration
        self.df_website = None  # Тут размещаем данные 'website' в формате Pandas полученные результате работы метода __get_data_website
        self.df = None
        self.log = []  # Лог операций
        self.model = None  # Тут будет храниться модель

        self.run_time = 0
        self.files_path = self.config.path + '/ModelNCF'


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
        answer = self.__fit_model(self.embedding_size)
        if answer['status'] != 'OK':
            return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение

        return answer

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


    # === ПОЛУЧЕНИЕ ДАННЫХ "WEBSITE" MONGODB ЗА НУЖНОЕ КОЛИЧЕСТВО ДНЕЙ ===
  def __get_data_website(self, days=30):
        # print('MODEL -> GET DATA')

        start_time = time.time()

        client = MongoClient(self.config.mongo_client_website)
        db = client['stats']
        website = db['website']

        # Отбор по дням
        timestamp = (int(time.time()) - int(days) * 86400)*1000
        results = website.find({"timestamp":{"$gte":timestamp}})
        res = [r for r in results]
        self.df_website = pd.DataFrame(list(res))
        # print(self.df_website.head())

        # print('--- ДАННЫЕ "WEBSITE" ПОЛУЧЕНЫ ---')
        delta_time = time.time() - start_time
        # print(f'РАЗМЕР ДАТАФРЕЙМА: {self.df_website.shape}, ВРЕМЯ ВЫПОЛНЕНИЯ: {round(delta_time, 4)}')

        answer = {
            'status': 'OK', 
            'message': f'Data received, number of rows: {self.df_website.shape[0]}, execution time: {round(delta_time, 4)}s'
        }
        self.__logging(answer)
        return answer

    # Метод обработки данных (на данный момент в качестве теста грузим датасет с google drive)
  def __data_processing(self):
        '''
        Ваша реализация обработки данных
        '''
        #функционал get_data перенесем в data_processing
        start_time= time.time()

        df = self.df_duration
        df['channel_id'] = pd.to_numeric(df['channel_id'], errors='coerce') 
        #df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')
        df = df.dropna()

        # --- Находим самые популярные каналы - составляем список из 100 каналов ---
        self.popular_list = df.groupby(['channel_id']).size().sort_values(ascending=False)[0:100].tolist()

        #кластеризация

        x = np.nan_to_num(df)
        Clust_dataset=StandardScaler().fit_transform(x)
        clasterNum = 11 #в соответствии с проведенным исследованием
        k_means = KMeans(init="k-means++", n_clusters=clasterNum, n_init=12)
        k_means.fit(Clust_dataset)
        labels=k_means.labels_
        df_labels = pd.DataFrame(labels, columns=['cluster'])
        indexes = df.index
        df_labels.index = indexes
        df_clust = pd.merge(left=df, right=df_labels, left_index=True, right_index=True)
        df_clust = df_clust.drop(columns=['duration'])

        #готовим данные для формирования финального датасета для обучения модели
        
        df_clust = df_clust.reindex(columns=['user_id', 'channel_id', 'cluster', 'time',]) # time  channel_id  user_id  duration
        df_clust.rename(columns={'user_id':'userId', 'channel_id':'itemId', 'cluster':'rating', 'time':'timestamp'}, inplace = True)
        df_clust['userId'] = df_clust['userId'].astype(int)
        df_clust['rating'] = df_clust['rating'].astype(int)
        df_clust['timestamp'] = df_clust['timestamp'].astype(int)
        df_clust = df_clust.reset_index()
        df = df_clust.copy()
        user_ids = df['userId'].unique().tolist() #делаем список пользователей
        self.user2user_encoded = {x: i for i, x in enumerate(user_ids)} #делаем словарь, где каждому ID пользователя присваиваем индекс 
        userencoded2user = {i: x for i, x in enumerate(user_ids)} #переводим ранее сформированный словарь userId-index в формат index-ID
        streamer_ids = df['itemId'].unique().tolist() #делаем список стримеров
        self.streamer2streamer_encoded = {x: i for i, x in enumerate(streamer_ids)} #делаем словарь, где каждому ID стримера присваиваем индекс 
        self.streamer_encoded2streamer = {i: x for i, x in enumerate(streamer_ids)} #переводим ранее сформированный словарь streamerId-index в формат index-streamerId

        #сформируем новый датафрейм с учетом индексов streamerId и userId
        df['user'] = df['userId'].map(self.user2user_encoded) 
        df['streamer'] = df['itemId'].map(self.streamer2streamer_encoded)

        self.num_users = len(self.user2user_encoded) #кол-во пользователей
        self.num_streamers = len(self.streamer_encoded2streamer) #кол-во стримеров
        df['rating'] = df['rating'].values.astype(np.float32) #переводим во float32 значение рейтинга
        df['userId'] = df['userId'].values.astype(np.float32) #переводим во float32 значение userID
        df['itemId'] = df['itemId'].values.astype(np.float32) #переводим во float32 значение itemId
        self.df_duration = df
        delta_time = time.time() - start_time
        self.run_time += delta_time 

        answer = {
            'status': 'OK',
            'message': f'Data processed, number of rows: {df.shape[0]}, execution time: {round(delta_time, 4)}s', #self.df.shape[0]
            'data': self.df_duration,
            'num_users': self.num_users,
            'num_streamers': self.num_streamers,
            'streamer2streamer_encoded': self.streamer2streamer_encoded,
            'user2user_encoded': self.user2user_encoded,
            'streamer_encoded2streamer': self.streamer_encoded2streamer,
        }
        
        self.__logging(answer)

        return answer
        
  def __fit_model(self, embedding_size):
        start_time = time.time()
        x_train, x_val, y_train, y_val = self.__train_split()
        self.model = CreateModel(self.num_users, self.num_streamers, embedding_size=self.embedding_size)
        self.model.compile(
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                        )
        history = self.model.fit(x_train,
                        y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        verbose=1,
                        validation_data=[x_val, y_val],
                        )
        
        message =   'Model training completed, '

        delta_time = time.time() - start_time
        self.run_time += delta_time
              

        answer = {
                  'status': 'OK', 
                  'message': message,
              }

        self.__logging(answer)
              
        return answer

  def model_save(model, path):
        model.save(path)
        return # print('модель сохранена')

  def model_load(path):
        return keras.models.load_model(path)

    # === ЗАПИСЬ В ЛОГ ===
  def __logging(self, answer):
        with open(self.files_path + '/status.log', 'a') as file:
            now = datetime.datetime.now()
            d = f'{now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second}'
            text = f"{d}, {answer['status']}: {answer['message']} "
            file.write(text)



    #функция вывода топовых стримеров для пользователя 
  def predict(self, user_id):
        data = self.df_duration
        model = self.model

        #по непонятным причинам не определяется наличия user_id в столбце с юзерами
        #создадим вспомогательный список всех юзеров в колонке userId, через него будем опредедять новый пользователь или нет
        l_temp = []
        for i in data.userId:
          l_temp.append(i)

        k = 10 #задаем количество выводимых рекомендованных стримеров
        if user_id in l_temp: #если пользователь заходил и смотрел что-то, предсказываем рекомендации
          streamers_watched_by_user = data[data.userId == user_id] #стримеры, которых смотрел пользователь
          #стримеры, которых не смотрел пользователь
          streamers_not_watched = data[~data["itemId"].isin(streamers_watched_by_user.itemId.values)]["itemId"]
          streamers_not_watched = list(set(streamers_not_watched).intersection(set(self.streamer2streamer_encoded.keys())))
          streamers_not_watched = [[self.streamer2streamer_encoded.get(x)] for x in streamers_not_watched]

          user_encoder = self.user2user_encoded.get(user_id) #вычисляем индекс userId
          user_streamer_array = np.hstack(([[user_encoder]] * len(streamers_not_watched), streamers_not_watched)) #формируем массив стримеров, которых юзер не смотрел
          ratings = model.predict(user_streamer_array).flatten() #делаем предсказание рейтинга
          top_ratings_indices = ratings.argsort()[-k:][::-1] #выбираем 10 индексов стримеров для "нашего" пользователя, с наивысшим предсказанным рейтингом

          #переводим индекс в streamerId
          recommended_streamer_ids = [self.streamer_encoded2streamer.get(streamers_not_watched[x][0]) for x in top_ratings_indices] 
          #формируем список топ 10 стримеров
          answer = []
          for i in recommended_streamer_ids:
            answer.append(str(i))

        else: #если пользователь новый, показываем 10 наиболее популярных каналов
          #формируем список популярных каналов
          answer = [] 
          #temp = data.groupby(data.columns.streamerId.to_list(), as_index=False).size()[:10]
          temp = data.groupby(data.itemId.to_list(), as_index=False).size()[:k]

          for i in range(len(temp)):
            answer.append(temp['index'].iloc[i])
        return answer       
       
  def __train_split(self):
        df = self.df_duration
        df = df.sample(frac=1, random_state=42) #перемешаем датафрейм
        x = df[['user', 'streamer']].values #формируем массив пар user-streamer
        df['rating'] = df['rating'].values.astype(np.float32) #переводим во float32 значение рейтинга
        #минимальное и максимальное значение рейтинга для дальнейшей нормализации:
        min_rating = min(df['rating'])
        max_rating = max(df['rating'])
        #формируем выход нейросети с нормализацией от 0 до 1 для лучшего качества обучения
        y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values 
        #разбивка на тестовую и проверочную выборку
        train_indices = int(0.9 * df.shape[0])
        x_train, x_val, y_train, y_val = (
            x[:train_indices],
            x[train_indices:],
            y[:train_indices],
            y[train_indices:],
        )


        return x_train, x_val, y_train, y_val  


# ======= СОЗДАЁМ МОДЕЛЬ =======
class CreateModel(keras.Model):
  def __init__(self, num_users, num_streamers, embedding_size, **kwargs): #, embedding_size, batch_size, epochs, **kwargs
        super(CreateModel, self).__init__(**kwargs)
        self.num_streamers = num_streamers
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.streamer_embedding = layers.Embedding(
            num_streamers,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.streamer_bias = layers.Embedding(num_streamers, 1)
  
  def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        streamer_vector = self.streamer_embedding(inputs[:, 1])
        streamer_bias = self.streamer_bias(inputs[:, 1])
        dot_user_streamer = tf.tensordot(user_vector, streamer_vector, 2)
        #добавим все компоненты включая bias
        x = dot_user_streamer + user_bias + streamer_bias
        #активационная функция сигмоида с выходом от 0 до 1
        return tf.nn.sigmoid(x)