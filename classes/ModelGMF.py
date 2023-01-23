import os
import time
import datetime
import requests
import pandas as pd
import numpy as np
import math

from typing import Dict, Text, Tuple
from pymongo import MongoClient

import tensorflow as tf                 
#import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs  
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import collections


class ModelGMF():

      def __init__(self, config):
          # print('MODEL GMF -> INIT')
          self.config = config          # Данные из конфигурационного файла
          self.data_duration = []       # Временное хранилище списка получаемых данных
          self.df_duration = None       # Тут размещаем данные 'duration' в формате Pandas, полученные в результате работы метода __get_data_duration
          #self.df_website = None       # Тут размещаем данные 'website' в формате Pandas полученные результате работы метода __get_data_website
                                        # Данные должны быть сохранены в self.df_duration, т.к. используются в предсказании
          self.pop_channels_rec = []    # Хранилище списка популярных каналов (для новичков). Сохраним recommend_num каналов [`str`]
          self.ind2name = {}            # Словарь     индекс : channel_id
          self.name2ind = {}            # Словарь channel_id : индекс
          self.df_users = None          # Тут размещаем данные 'df_users' в формате Pandas, полученные в результате обработки df_duration
          self.channel_corr_mat = None  # Матрица обработанных данных (косинусных расстояний между channel_id)

          self.log = []                 # Лог операций
          self.model = None             # Тут будет храниться модель (излишне, т.к. нужна будет только на случай дообучения, что вряд ли нужно)
          self.run_time = 0
          self.recommend_num = 50       # Скорректировать при необходимости!!!

          self.files_path = self.config.path + '/ModelGMF'

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

          # Обработка данных `duration`
          answer = self.__data_processing_duration()
          if answer['status'] != 'OK':
              return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение

          # Получаем данные `website` за указанное количество дней `days`
          # self.__get_data_website(days)
          # if answer['status'] != 'OK':
          #     return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнени

          # Обработка данных `website`
          # answer = self.__data_processing_website()
          # if answer['status'] != 'OK':
          #     return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение        

          # Обучение модели
          answer = self.__fit_model()
          if answer['status'] != 'OK':
              return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение

          return answer



      # === ПОЛУЧЕНИЕ ДАННЫХ "DURATION" ПО API ЗА НУЖНОЕ КОЛИЧЕСТВО ДНЕЙ ===
      def __get_data_duration(self,  days=30):
          # print('MODEL GMF -> GET DATA')

          start_time = time.time()
          from_time = int(time.time()) - int(days) * 86400
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
                          # print(self.data_duration[0:5])
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

          # print('--- ДАННЫЕ "DURATION" ПОЛУЧЕНЫ ---')
          # print(f'РАЗМЕР ДАННЫХ: {self.df_duration.shape}, ВРЕМЯ ВЫПОЛНЕНИЯ: {round(delta_time, 4)}')
          # print(self.df_duration.head(10))
          return answer 


      # === ПОЛУЧЕНИЕ ДАННЫХ "WEBSITE" MONGODB ЗА НУЖНОЕ КОЛИЧЕСТВО ДНЕЙ ===
      def __get_data_website(self, days=30):
          # print('MODEL -> GET DATA')

          start_time = time.time()

          client = MongoClient(self.config.mongo_client_website)
          db = client['stats']
          website = db['website']

          # Отбор по дням
          timestamp = (int(time.time()) - int(days) * 86400) * 1000
          results = website.find({"timestamp": {"$gte":timestamp}})
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



      # === ОБРАБОТКА ДАННЫХ `Duration`===
      def __data_processing_duration(self):
          # print('MODEL -> DATA PROCESSING')
          start_time = time.time()

          df = self.df_duration
          df.dropna(inplace=True)

          # --- Находим самые популярные каналы - составляем список из 100 каналов ---
          self.popular_list = df.groupby(['channel_id']).size().sort_values(ascending=False)[0:100].tolist()

          # --- Удаляем стримы с небольшим количество просмотров (20% от среднего) ---
          # Находим среднее число просмотров стрима
          view_mean = df.groupby(['channel_id']).size().mean()
          # print('СРЕДНЕЕ КОЛИЧЕСТВО ПРОСМОТРОВ', view_mean)

          # Создадим новый датасет только с подсчетом рейтинга, чтобы исключить некоторые
          view_counter = pd.DataFrame({'Count' : df.groupby(['channel_id']).size()}).reset_index()
          k_1 = view_counter.shape[0]

          view_counter = view_counter.loc[view_counter['Count'] > view_mean/5]  # 1/5
          k_2 = view_counter.shape[0]

          # print(f'КАНАЛОВ ДО ФИЛЬТРАЦИИ: {k_1}, ПОСЛЕ: {k_2}')

          reducer = df['channel_id'].isin(view_counter['channel_id'])
          df_2 = df[reducer]
          # print(f'СТРИМОВ ДО ФИЛЬТРАЦИИ: {df.shape[0]}, ПОСЛЕ: {df_2.shape[0]}')

          self.df_duration = df_2.drop_duplicates()
          # print(f'ПОСЛЕ УДАЛЕНИЯ ДУБЛИКАТОВ: {self.df_duration.shape[0]}')

          delta_time = time.time() - start_time
          self.run_time += delta_time
          answer = {
              'status': 'OK', 
              'message': f'Data processed, number of rows: {self.df_duration.shape[0]}, execution time: {round(delta_time, 4)}s'
          }
          self.__logging(answer)

          return answer



      # === ОБРАБОТКА ДАННЫХ `WEBSITE`===
      # На выходе должна быть база данных df_duration
      # Дописать (при необходимости)



      # === ЗАПИСЬ В ЛОГ ===
      def __logging(self, answer):
          with open(self.files_path + '/status.log', 'a') as file:
              now = datetime.datetime.now()
              d = f'{now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second}'
              text = f"{d}, {answer['status']}: {answer['message']} \n"
              file.write(text)



      # === ОБУЧЕНИЕ МОДЕЛИ ===
      def __fit_model(self):
          # print('MODEL -> FIT MODEL')
          start_time = time.time()

          '''
          Готовим две части рекомендательной системы:
            1 - Рекомендации новичкам на основе рейтинга каналов
            2 - Рекомендации от обученной нейронки
          '''

          # --- часть 1 ---
          # Определим популярные каналы (для рекомендации новичкам)
          # Для этого создадим новый DataFrame `result` (будет нужен для обучения модели)
          result = self.df_duration.groupby(['channel_id','user_id']).sum().reset_index()
          result.drop(columns=['time'],axis = 1, inplace=True)
          
          # Добавим столбец продолжительности по диапазонам, для каждого пользователя относительно данного канала
          # Так как значения в столбце duration слишком велики, разобьём их по диапазонам по логарифимеской шкале
          result['rating'] = pd.DataFrame(np.log((result['duration'] - 1)/60000))

          # Затем рассчитаем средневзвешенный ретинг каждого канала на основе предпочтений users
          def weighted_rating(v,m,R,C):
              '''
              Calculate the weighted rating
              
              Args:
              v -> average rating for each channel (float)
              m -> minimum votes required to be classified as popular (float)
              R -> average rating for the channel (pd.Series)
              C -> average rating for the whole dataset (pd.Series)
              
              Returns:
              pd.Series
              '''
              return ( (v / (v + m)) * R) + ( (m / (v + m)) * C )

          def assign_popular_based_score(rating_df, user_col, channel_col, rating_col):
              '''
              Assigned popular based score (based on the IMDB weighted average).
              
              Args:
              result -> pd.DataFrame contains ['channel_id', 'rating'] for each user.
              
              Returns
              popular_channels -> pd.DataFrame contains channel and weighted rating.
              '''
              
              # pre processing
              vote_count = (
                  rating_df
                  .groupby(channel_col,as_index=False)
                  .agg( {user_col:'count', rating_col:'mean'} )
                  )
              vote_count.columns = [channel_col, 'vote_count', 'avg_rating']
              
              # calcuate input parameters
              C = np.mean(vote_count['avg_rating'])
              m = np.percentile(vote_count['vote_count'], 80)               # Персентиль от общего количества значений рейтинга
              vote_count = vote_count[vote_count['vote_count'] >= m]
              R = vote_count['avg_rating']
              v = vote_count['vote_count']
              vote_count['weighted_rating'] = weighted_rating(v,m,R,C)
              
              # post processing
              popular_channels = vote_count.loc[:,[channel_col, 'vote_count', 'avg_rating', 'weighted_rating']]
              
              return popular_channels

          # init constant
          USER_COL = 'user_id'
          CHANNEL_COL = 'channel_id'
          RATING_COL = 'rating'

          # calcualte popularity based
          pop_channels = assign_popular_based_score(result, USER_COL, CHANNEL_COL, RATING_COL)
          pop_channels = pop_channels.sort_values('weighted_rating', ascending = False)

          # Сохраним список популярных каналов для рекомендации новым пользователям в количестве self.recommend_num
          self.pop_channels_rec = pop_channels['channel_id'].iloc[:self.recommend_num].astype(str).tolist()
          # print(f'ПОДГОТОВЛЕН СПИСОК ПОПУЛЯРНЫХ КАНАЛОВ ДЛЯ НОВИЧКОВ В КОЛИЧЕСТВЕ: {len(self.pop_channels_rec)}')


          # --- часть 2 ---
          # Готовим данные для обучения модели
          # Пронумеруем уникальные наименования каналов и юзеров
          result['channel_indx'] = pd.factorize(result['channel_id'])[0]
          result['user_indx'] = pd.factorize(result['user_id'])[0]

          # Составим массивы по уникальным каналам и юзерам. Имена сохраним как строковые
          result.sort_values('channel_indx', ascending=True, inplace=True)
          df_channels = pd.DataFrame(result.sort_values('channel_indx', ascending=True)['channel_id'].unique())
          df_channels.columns = ['channel_id']
          df_channels['channel_id'] = df_channels['channel_id'].astype(str)

          result.sort_values('user_indx', ascending=True, inplace=True)
          self.df_users = pd.DataFrame(result.sort_values('user_indx', ascending=True)['user_id'].unique())
          self.df_users.columns = ['user_id']
          self.df_users['user_id'] = self.df_users['user_id'].astype(str)   # Сохраняем базу по юзерам

          # Для дальнейшего использования соберём словари
          self.ind2name = df_channels.to_dict()['channel_id']                      # Словарь     индекс : channel_id
          self.name2ind = {v:k for k,v in self.ind2name.items()}                   # Словарь channel_id : индекс

          # Преобразуем индексы юзеров и каналов в строковой тип
          result['user_indx'] = result['user_indx'].astype(str)
          result['channel_indx'] = result['channel_indx'].astype(str)


          # ----------------
          # Обучение модели проводим, если новые данные
          def df_to_ds(df):

              # Конвертируем pd.DataFrame в tf.data.Dataset
              ds = tf.data.Dataset.from_tensor_slices(
                  (dict(df[['user_indx','channel_indx']]), df['rating']))
              
              # Конвертируем Tuple[Dict[Text, tf.Tensor], tf.Tensor] в Dict[Text, tf.Tensor]
              ds = ds.map(lambda x, y: {
                  'user_indx' : x['user_indx'],
                  'channel_indx' : x['channel_indx'],
                  'rating' : y
                  })

              return ds.batch(256)

          class RankingModel(keras.Model):

              def __init__(self, users_db, user_indx, channel_indx, embedding_size):
                  super().__init__()
                  
                  # user model
                  input = keras.Input(shape=(), dtype=tf.string)
                  x = keras.layers.StringLookup(
                      vocabulary = users_db.index.astype(str).to_numpy(), mask_token = None
                      )(input)
                  output = keras.layers.Embedding(
                      input_dim = len(user_indx) + 1,
                      output_dim = embedding_size,
                      name = 'embedding'
                      )(x)
                  self.user_model = keras.Model(inputs = input,
                                                outputs = output,
                                                name = 'user_model')
                  
                  # channel model
                  input = keras.Input(shape=(), dtype=tf.string)
                  x = keras.layers.StringLookup(
                      vocabulary = df_channels.index.astype(str).to_numpy(), mask_token = None
                      )(input)
                  output = keras.layers.Embedding(
                      input_dim = len(channel_indx) + 1,
                      output_dim = embedding_size,
                      name = 'embedding'
                      )(x)
                  self.channel_model = keras.Model(inputs = input,
                                            outputs = output,
                                            name = 'channel_model')

                  # rating model
                  user_input = keras.Input(shape=(embedding_size,), name='user_emb')
                  channel_input = keras.Input(shape=(embedding_size,), name='channel_emb')
                  x = keras.layers.Concatenate(axis=1)([user_input, channel_input])
                  x = keras.layers.Dense(256, activation = 'relu')(x)
                  x = keras.layers.Dense(64, activation = 'relu')(x)
                  output = keras.layers.Dense(1)(x)
                  
                  self.rating_model = keras.Model(
                      inputs = {
                          'user_indx' : user_input,
                          'channel_indx' : channel_input
                          },
                      outputs = output,
                      name = 'rating_model'
                      )

              def call(self, inputs: Dict[Text, tf.Tensor]) -> tf.Tensor:

                  user_emb = self.user_model(inputs['user_indx'])
                  channel_emb = self.channel_model(inputs['channel_indx'])

                  prediction = self.rating_model({
                      'user_indx' : user_emb,
                      'channel_indx' : channel_emb
                      })
                  
                  return prediction

          class GMFModel(tfrs.models.Model):

              def __init__(self, users_db, user_indx, channel_indx, embedding_size):
                  super().__init__()
                  self.ranking_model = RankingModel(users_db, user_indx, channel_indx, embedding_size)
                  self.task = tfrs.tasks.Ranking(
                      loss = keras.losses.MeanSquaredError(),
                      metrics = [keras.metrics.RootMeanSquaredError()]
                      )
              
              def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
                  
                  return self.ranking_model(
                      {
                      'user_indx' : features['user_indx'], 
                      'channel_indx' : features['channel_indx']
                      })

              def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

                  return self.task(labels = features.pop('rating'),
                                  predictions = self.ranking_model(features))

          # Предобработка исходных данных - разбивка на обучающую и проверучную выбрки
          train, test = train_test_split(result[['user_indx','channel_indx','rating']], train_size = .8, random_state=42)
          train, val_ds = train_test_split(train, train_size = .8, random_state=None)

          # Переводим датасеты в представление tf.data.Dataset
          train, test, val_ds = df_to_ds(train), df_to_ds(test),  df_to_ds(val_ds)
          
          # ----------------
          ## Инициализация модели
          embedding_size = 64
          self.model = GMFModel(self.df_users, self.df_users.index.astype(str),
                          df_channels.index.astype(str),
                          embedding_size)
          self.model.compile(
              optimizer = keras.optimizers.Adagrad(learning_rate = .1)
              )

          ## Обучение модели
          history = self.model.fit(train, epochs=30, verbose=1, validation_data=val_ds, validation_steps=10)
          # Оценка качества на тестовой выборке
          eval_result = self.model.evaluate(test, return_dict=True, verbose=0)
          # print("\n--- Оценка качества обучения модели на тестовой выборке: ---")
          # print(eval_result)

          # Извлекаем скрытое пространство обученной модели
          channel_emb = self.model.ranking_model.channel_model.layers[-1].get_weights()[0]
          # Сохраняем матрицу обработанных данных (косинусных расстояний между channel_id)
          self.channel_corr_mat = cosine_similarity(channel_emb)

          # ----------------
          delta_time = time.time() - start_time
          self.run_time += delta_time
          message = 'Model training completed, '
          message += f'total execution time: {round(self.run_time)}s '
          answer = {
              'status': 'OK',
              'message': message
          }
          self.__logging(answer)

          return answer



      # === ПРЕДСКАЗАНИЕ МОДЕЛИ ===
      def predict(self, user_id):
          # где `user_id` - строковой тип

          def top_k_channels(channel_id, top_k, corr_mat, map_name):
              
              # Сортировка значений по увеличению и выбор первых top_k элементов
              top_channels = corr_mat[channel_id,:].argsort()[-top_k-1:][::-1] 
              top_channels = [map_name[e] for e in top_channels][1:]
              return top_channels

          # Для указанного юзера смотрим по website (df_copy) историю его взаимодействия
          if user_id in self.df_users['user_id'].unique():
              # Если такой пользователь есть,
              # Выбираем всю историю его взаимодействий (сортируем по истории)
              df = self.df_duration[self.df_duration['user_id']==int(user_id)][['time', 'channel_id']].sort_values('time', ascending=False)

              # Берём последний просмотренный им канал
              ch_id = str(df['channel_id'].iloc[0])

              # Рекомендуем похожие каналы в количестве self.recommend_num
              # На всякий случай будем обрабатывать ошибки
              try:
                  answer = top_k_channels(self.name2ind[ch_id],
                                          top_k = self.recommend_num,
                                          corr_mat = self.channel_corr_mat,
                                          map_name = self.ind2name)
              except:
                  # Непредвиденная ошибка - выдаём заготовленный список популярных каналов
                  answer = self.pop_channels_rec

          else:
              # Если такого пользователя нет, то выдаём заготовленный список популярных каналов
              answer = self.pop_channels_rec
              ## print("Новый пользователь")

          # В итоговом выводе - список с рекомендациями (channel_id) тип `str`
          return answer