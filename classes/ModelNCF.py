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


class ModelNCF():
  def __init__(self, config, path, num_users, num_streamers, embedding_size, batch_size, epochs, **kwargs):
        print('MODEL NFC -> INIT')
        super(RecommenderNet, self).__init__(**kwargs)
        self.config = config  # Данные из конфигурационного файла
        self.data_duration = []  # Временное хранилище списка получаемых данных
        self.df_duration = False  # Данные 'duration' в формате Pandas
        self.df_website = False  # Данные 'website' в формате Pandas
        self.log = []  # Лог операций
        self.path = path
        self.top_k = k
        self.num_users = num_users #кол-во пользователей
        self.num_streamers = num_streamers #кол-во стримеров
        self.embedding_size = embedding_size #размер внутреннего пространства
        self.batch_size = batch_size #размер batch_size
        self.epochs = epochs #кол-во эпох

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

    # === ОБУЧЕНИЕ МОДЕЛИ ===
  def fit(self, days=30):
        # Удаляем файл логов
        if os.path.isfile(self.config.log_file): 
            os.remove(self.config.log_file)

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

             # === ПОЛУЧЕНИЕ ДАННЫХ "DURATION" ПО API ЗА НУЖНОЕ КОЛИЧЕСТВО ДНЕЙ ===
  def __get_data_duration(self, model_key, act, days, top, rand, gg_key, gg_api_url, gg_pagination_step):
        print('MODEL TFRS -> GET DATA')

        # !!! БАЗА НЕ АКТУАЛЬНАЯ - ДОБАВЛЯЕМ КОСТЫЛИ ДЛЯ СМЕЩЕНИЯ ПО СРОКАМ НА1 ГОД
        crutch = 86400 * 365

        start_time = time.time()
        from_time = int(time.time()) - int(days)*86400 - crutch
        from_time_url = f'&from={from_time}'
        to_time = int(time.time()) - crutch
        from_to_time = f'&to={to_time}'

        # --- Соединение с GG Api и получение данных ---
        for i in range(1, 1000):
            url = gg_api_url + from_time_url + from_to_time + '&page=' + str(i)
            try:
                api_req = requests.get(url)
                if int(api_req.status_code) != 200:
                    answer = {
                        'status': 'ERR', 
                        'message': f'Error connecting to goodgame.ru API. Server API response code: {api_req.status_code}'
                    }
                    break
                else:
                    # Добавляем данные в наш список
                    data_list = api_req.json()
                    self.data_duration.extend(data_list)
                    print(i, len(self.data_duration), len(data_list))
                    if len(data_list) < self.gg_pagination_step:  # Шаг пагинации
                        print(self.data_duration[0:5])
                        self.df_duration = pd.DataFrame.from_dict(self.data_duration)
                        self.data_duration = []
                        break
            except:
                answer = {
                    'status': 'ERR', 
                    'messare': 'Error connecting to goodgame.ru API'
                }
                break

        # --- Формирование ответа ---
        delta_time = time.time() - start_time
        if (len(self.data_duration) > 0):
            answer = {
                'status': 'OK', 
                'message': f'Data received, number of rows: {len(self.data_duration)}, execution time: {round(delta_time, 4)}s'
            }
        else:
            answer = {
                'status': 'ERR', 
                'message': f'No data, execution time: {round(delta_time, 4)}s'
            }

        print('--- ДАННЫЕ "DURATION" ПОЛУЧЕНЫ ---')
        print(f'РАЗМЕР ДАННЫХ: {self.df_duration.shape}, ВРЕМЯ ВЫПОЛНЕНИЯ: {round(delta_time, 4)}')
        self.log.append(f"{answer['status']}. {answer['message']}")
        return answer 


    # === ПОЛУЧЕНИЕ ДАННЫХ "WEBSITE" MONGODB ЗА НУЖНОЕ КОЛИЧЕСТВО ДНЕЙ ===
  def __get_data_website(self, days=30):
        print('MODEL -> GET DATA')

        start_time = time.time()

        client = MongoClient(self.config.mongo_client_website)
        db = client['stats']
        website = db['website']

        # Отбор по дням
        timestamp = (int(time.time()) - int(days) * 86400)*1000
        results = website.find({"timestamp":{"$gte":timestamp}})
        res = [r for r in results]
        self.df_website = pd.DataFrame(list(res))
        print(self.df_website.head())

        print('--- ДАННЫЕ "WEBSITE" ПОЛУЧЕНЫ ---')
        delta_time = time.time() - start_time
        print(f'РАЗМЕР ДАТАФРЕЙМА: {self.df_website.shape}, ВРЕМЯ ВЫПОЛНЕНИЯ: {round(delta_time, 4)}')

        answer = {
            'status': 'OK', 
            'message': f'Data received, number of rows: {self.df_website.shape[0]}, execution time: {round(delta_time, 4)}s'
        }
        self.__logging(answer)
        return answer

    # Метод обработки данных (на данный момент в качестве теста грузим датасет с google drive)
  def data_processing(self):
        '''
        Ваша реализация обработки данных
        '''
        #функционал get_data перенесем в data_processing
        
        '''df = pd.read_csv(path, names=['time', 'channelId', 'userId', 'duration'])  # Тут ваше значение датафрейма (compression='gzip', )
        df = df.drop(df.index[0]) 
        df['channelId'] = pd.to_numeric(df['channelId'], errors='coerce')
        df['userId'] = pd.to_numeric(df['userId'], errors='coerce')
        df = df.dropna()'''

        df = self.df_duration
        df.dropna(inplace=True)

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
        df_clust = df_clust.reindex(columns=['userId', 'channelId', 'cluster', 'time',])
        df_clust.rename(columns={'userId':'userId', 'channelId':'itemId', 'cluster':'rating', 'time':'timestamp'}, inplace = True)
        df_clust['userId'] = df_clust['userId'].astype(int)
        df_clust['rating'] = df_clust['rating'].astype(int)
        df_clust['timestamp'] = df_clust['timestamp'].astype(int)
        df_clust = df_clust.reset_index()
        df = df_clust.copy()
        user_ids = df['userId'].unique().tolist() #делаем список пользователей
        user2user_encoded = {x: i for i, x in enumerate(user_ids)} #делаем словарь, где каждому ID пользователя присваиваем индекс 
        userencoded2user = {i: x for i, x in enumerate(user_ids)} #переводим ранее сформированный словарь userId-index в формат index-ID
        streamer_ids = df['itemId'].unique().tolist() #делаем список стримеров
        streamer2streamer_encoded = {x: i for i, x in enumerate(streamer_ids)} #делаем словарь, где каждому ID стримера присваиваем индекс 
        streamer_encoded2streamer = {i: x for i, x in enumerate(streamer_ids)} #переводим ранее сформированный словарь streamerId-index в формат index-streamerId

        #сформируем новый датафрейм с учетом индексов streamerId и userId
        df['user'] = df['userId'].map(user2user_encoded) 
        df['streamer'] = df['itemId'].map(streamer2streamer_encoded)

        num_users = len(user2user_encoded) #кол-во пользователей
        num_streamers = len(streamer_encoded2streamer) #кол-во стримеров
        df['rating'] = df['rating'].values.astype(np.float32) #переводим во float32 значение рейтинга
        df['userId'] = df['userId'].values.astype(np.float32) #переводим во float32 значение userID
        df['itemId'] = df['itemId'].values.astype(np.float32) #переводим во float32 значение itemId

        answer = {
            'status': 'OK',
            'message': f'Data processed, number of rows: {self.df.shape[0]}, execution time: {round(delta_time, 4)}s',
            'data': df,
            'num_users': num_users,
            'num_streamers': num_streamers,
            'streamer2streamer_encoded': streamer2streamer_encoded,
            'user2user_encoded': user2user_encoded,
            'streamer_encoded2streamer': streamer_encoded2streamer,
        }

        self.__logging(answer)

        return answer


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

  def __fit_model(model, x_train, y_train, batch_size, epochs, validation_data):
        history = model.fit(x_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=validation_data,
                  )
        self.__logging(answer)
        message =   'Model training completed, '
        message +=  f'top_1: {history.history["factorized_top_k/top_1_categorical_accuracy"][-1]:.4f}, '
        message +=  f'top_5: {history.history["factorized_top_k/top_5_categorical_accuracy"][-1]:.4f}, '
        message +=  f'top_10: {history.history["factorized_top_k/top_10_categorical_accuracy"][-1]:.4f}, '
        message +=  f'top_50: {history.history["factorized_top_k/top_50_categorical_accuracy"][-1]:.4f}, '
        message +=  f'top_100: {history.history["factorized_top_k/top_100_categorical_accuracy"][-1]:.4f}, '
        message +=  f'total execution time: {round(self.run_time)}s '

        answer = {
            'status': 'OK', 
            'message': message,
            'history': history,
        }
        
        return answer

  def model_save(model, path):
        model.save(path)
        return print('модель сохранена')

  def model_load(path):
        return keras.models.load_model(path)

    # === ЗАПИСЬ В ЛОГ ===
  def __logging(self, answer):
        with open(self.config.log_file, 'a') as file:
            now = datetime.datetime.now()
            d = f'{now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second}'
            text = f"{d}, {answer['status']}: {answer['message']} "
            file.write(text)

    #функция вывода топовых стримеров для пользователя 
  def predict(model, data, user_id, streamer2streamer_encoded, user2user_encoded, streamer_encoded2streamer, k):
        streamers_watched_by_user = data[data.userId == user_id] #стримеры, которых смотрел пользователь

        #стримеры, которых не смотрел пользователь
        streamers_not_watched = data[~data["itemId"].isin(streamers_watched_by_user.itemId.values)]["itemId"]
        streamers_not_watched = list(set(streamers_not_watched).intersection(set(streamer2streamer_encoded.keys())))
        streamers_not_watched = [[streamer2streamer_encoded.get(x)] for x in streamers_not_watched]

        user_encoder = user2user_encoded.get(user_id) #вычисляем индекс userId
        user_movie_array = np.hstack(([[user_encoder]] * len(streamers_not_watched), streamers_not_watched)) #формируем массив стримеров, которых юзер не смотрел
        ratings = model.predict(user_movie_array).flatten() #делаем предсказание рейтинга
        top_ratings_indices = ratings.argsort()[-k:][::-1] #выбираем 10 индексов стримеров для "нашего" пользователя, с наивысшим предсказанным рейтингом

        #переводим индекс в streamerId
        recommended_streamer_ids = [streamer_encoded2streamer.get(streamers_not_watched[x][0]) for x in top_ratings_indices] 

        #формируем словарь userId - top 10 streamersId
        #result = {}
        #result[user_id] = recommended_streamer_ids

        #формируем список
        answer = []
        for i in recommended_streamer_ids:
          answer.append(str(i))

        return answer       
      
  def train_split(data):
      data = data.sample(frac=1, random_state=42) #перемешаем датафрейм
      x = data[['user', 'streamer']].values #формируем массив пар user-streamer
      data['rating'] = data['rating'].values.astype(np.float32) #переводим во float32 значение рейтинга
      #минимальное и максимальное значение рейтинга для дальнейшей нормализации:
      min_rating = min(data['rating'])
      max_rating = max(data['rating'])
      #формируем выход нейросети с нормализацией от 0 до 1 для лучшего качества обучения
      y = data['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values 
      #разбивка на тестовую и проверочную выборку
      train_indices = int(0.9 * data.shape[0])
      x_train, x_val, y_train, y_val = (
          x[:train_indices],
          x[train_indices:],
          y[:train_indices],
          y[train_indices:],
      )


      return x_train, x_val, y_train, y_val

# ModelNCF.fit(days=30)

