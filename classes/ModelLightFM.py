import os
import pandas as pd
import numpy  as np
import time
import datetime
import requests
from pymongo import MongoClient
from lightfm import LightFM
from scipy.sparse import coo_matrix


class ModelLightFM():

      def __init__(self, config):
          # print('MODEL LightFM -> INIT')
          self.config = config  # Данные из конфигурационного файла
          self.data_duration = []  # Временное хранилище списка получаемых данных
          self.df_duration = None  # Тут размещаем данные 'duration' в формате Pandas, полученные результате работы метода __get_data_duration
          self.df_website = None  # Тут размещаем данные 'website' в формате Pandas полученные результате работы метода __get_data_website
          self.log = []  # Лог операций
          self.model = None  # Тут будет храниться модель
          self.run_time = 0
          self.files_path = self.config.path
          self.channel_key = [] # Список id каналов (стримеров) из базы данных
          self.channel_value = [] # Список id каналов (стримеров) с нумерацией по порядку
          self.channel_dict = {} # Словарь id каналов (стримеров)
          self.channel_dict_rev = {} # Словарь id каналов (стримеров) наоборот
          self.user_key = [] # Список id юзеров из базы данных
          self.user_value = [] # Список id юзеров с нумерацией по порядку
          self.user_dict = {} # Словарь id юзеров
          self.user_dict_rev = {} # Словарь id юзеров наоборот
          self.coo_data = None # coo матрица обработанных данных
          self.popular_channels = [] # Список популярных каналов (стримеров)

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

          # Обработка данных duration
          answer = self.__data_processing_duration()
          if answer['status'] != 'OK':
              return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение

          # Получаем данные `website` за указанное количество дней `days`
          # self.__get_data_website(days)
          # if answer['status'] != 'OK':
          #     return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнени

          # Обработка данных website
          # answer = self.__data_processing_website()
          # if answer['status'] != 'OK':
          #     return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение

          # Обучение модели
          answer = self.__fit_model()
          if answer['status'] != 'OK':
              return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение

          return answer

      # === ПОЛУЧЕНИЕ ДАННЫХ "DURATION" ПО API ЗА НУЖНОЕ КОЛИЧЕСТВО ДНЕЙ ===
      def __get_data_duration(self, days=30):
          # print('MODEL LightFM -> GET DATA')

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
          results = website.find({"timestamp": {"$gte": timestamp}})
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

      # === ЗАПИСЬ В ЛОГ ===
      def __logging(self, answer):
          with open(self.files_path + '/status.log', 'a') as file:
              now = datetime.datetime.now()
              d = f'{now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second}'
              text = f"{d}, {answer['status']}: {answer['message']} \n"
              file.write(text)

      # === ОБРАБОТКА ДАННЫХ WEBSITE===
      def __data_processing_website(self):
          # print('MODEL -> DATA PROCESSING')
          start_time = time.time()

          website = self.df_website

          website['streamId'] = website['streamer'].fillna(website['streamId'])
          website.drop(columns=['user','streamer'], inplace=True)
          website.dropna(subset=['event'], inplace=True)
          website['game'] = website['game'].fillna('unknoun_game')
          website['message'] = website['message'].fillna('unknoun_message')
          website['streamStarted'] = website['streamStarted'].fillna(0)
          website[['streamId','streamStarted','userId']] = website[['streamId','streamStarted','userId']].astype('int64')
          website = website.loc[website.userId != 0]
          website.drop(columns=['message'], inplace=True)
          website = website.loc[website.event != 'chat-message']
          website = website.loc[website.event != 'stream-follow']
          website = website.loc[website.streamStarted != 0]
          view = website.loc[website.event == 'stream-view']
          view.drop(columns=['event'], inplace=True)
          view = view.reindex(columns=['game', 'streamId', 'userId', 'streamStarted', 'timestamp'])
          # dt = pd.to_datetime(view.timestamp, unit='ms').min()
          # view = view.loc[view.streamStarted > int(datetime.timestamp(datetime(dt.year, dt.month, dt.day, 0, 0, 0)))]
          view_agg = view.groupby(['userId',
                                   'streamId',
                                   'game',
                                   'streamStarted'], as_index=False).agg({'timestamp':['min','max']})
          view_agg.columns = ['_'.join(col).rstrip('_') for col in view_agg.columns.values]
          view_agg['duration'] = view_agg.timestamp_max - view_agg.timestamp_min
          view_agg.drop(columns=['timestamp_max'], inplace=True)
          view_agg = view_agg.rename(columns={'timestamp_min': 'timestamp'})
          view_agg = view_agg.loc[view_agg.duration >= 60_000]
          view_agg.duration = view_agg.duration // 1000
          view_agg.index = np.arange(len(view_agg))

          self.user_key = pd.unique(view_agg.userId)
          self.user_value = [i for i in range(len(self.user_key))]
          self.user_dict = dict(zip(self.user_key, self.user_value))
          self.user_dict_rev = dict(zip(self.user_value, self.user_key))
          view_agg['us_id'] = view_agg['userId'].map(self.user_dict)

          self.channel_key = pd.unique(view_agg.streamId)
          self.channel_value = [i for i in range(len(self.channel_key))]
          self.channel_dict = dict(zip(self.channel_key, self.channel_value))
          self.channel_dict_rev = dict(zip(self.channel_value, self.channel_key))
          view_agg['ch_id'] = view_agg['streamId'].map(self.channel_dict)

          self.coo_data = coo_matrix((view_agg.duration, (view_agg.us_id, view_agg.ch_id)))

          self.popular_channels = list(view_agg.sort_values(by = 'duration', ascending = False))

          delta_time = time.time() - start_time
          self.run_time += delta_time
          answer = {
              'status': 'OK',
              'message': f'Data processed, number of rows: {self.df_duration.shape[0]}, execution time: {round(delta_time, 4)}s'
          }
          self.__logging(answer)

          # print('--- ДАННЫЕ "WEBSITE" ОБРАБОТАНЫ ---')

          return answer

      # === ОБРАБОТКА ДАННЫХ DURATION===
      def __data_processing_duration(self):
          # print('MODEL -> DATA PROCESSING')
          start_time = time.time()

          # --- Находим самые популярные каналы - составляем список из 100 каналов ---
          self.popular_list = self.df_duration.groupby(['channel_id']).size().sort_values(ascending=False)[0:100].tolist()

          durations = self.df_duration

          # Удаление строк с пустыми ячейками
          durations.dropna(axis = 0, inplace=True)

          # Удаление аномально высоких user_id
          user_id_max = 1e+7 # 100.000.000
          durations.drop(durations[durations.user_id > user_id_max].index, axis=0, inplace=True)

          # Длительность просмотра более 8 часов приравнять к 8 часам - 28.800.000 ms
          duration_max = int(np.timedelta64(8, 'h') / np.timedelta64(1, 'ms'))
          durations.loc[durations['duration'] > duration_max, 'duration'] = duration_max

          # Длительность просмотра менее 1 минуты удалить
          duration_min = int(pd.Timedelta(1, "m") / np.timedelta64(1, 'ms'))
          durations.drop(durations[durations.duration < duration_min].index, axis=0, inplace=True)

          # Создание нового столбца ch_id с id каналов
          self.channel_key   = pd.unique(durations.channel_id) # Ключи из названий уникальных id каналов
          self.channel_value = [i for i in range(len(self.channel_key))] # Значения по порядку для уникальных id каналов
          self.channel_dict  = dict(zip(self.channel_key, self.channel_value)) # Словарь старых и новых id каналов
          self.channel_dict_rev = dict(zip(self.channel_value, self.channel_key)) # Словарь новых и старых id каналов
          durations['ch_id'] = durations['channel_id'].map(self.channel_dict) # Присвоение новых значений

          # Создание нового столбца us_id с id юзеров
          self.user_key   = pd.unique(durations.user_id) # Ключи из названий уникальных id юзеров
          self.user_value = [i for i in range(len(self.user_key))] # Значения по порядку для уникальных id юзеров
          self.user_dict  = dict(zip(self.user_key, self.user_value)) # Словарь старых и новых id юзеров
          self.user_dict_rev = dict(zip(self.user_value, self.user_key)) # Словарь новых и старых id юзеров
          durations['us_id'] = durations['user_id'].map(self.user_dict) # Присвоение новых значений

          # Группировка значений по столбцам channel_id и user_id
          durations = durations.groupby(['user_id',
                                         'channel_id']).agg({'time':'max',
                                                             'duration':'sum',
                                                             'ch_id':'max',
                                                             'us_id':'max'})

          # sorted_channels = durations.sort_values(by = 'duration', ascending = False)
          # self.popular_channels = [self.channel_dict_rev[id] for id in sorted_channels.ch_id]
          groupby_channels = durations.groupby(['ch_id']).agg({'duration':'sum', 'ch_id':'max'})
          sorted_channels = groupby_channels.sort_values(by = 'duration', ascending = False)
          self.popular_channels = [self.channel_dict_rev[id] for id in sorted_channels.ch_id]

          self.coo_data = coo_matrix((durations.duration, (durations.us_id, durations.ch_id)))

          delta_time = time.time() - start_time
          self.run_time += delta_time
          answer = {
              'status': 'OK',
              'message': f'Data processed, number of rows: {self.df_duration.shape[0]}, execution time: {round(delta_time, 4)}s'
          }
          self.__logging(answer)

          # print('--- ДАННЫЕ "DURATION" ОБРАБОТАНЫ ---')

          return answer

      def __fit_model(self):
          # print('MODEL -> FIT MODEL')
          start_time = time.time()

          self.model = LightFM(no_components=32, loss='warp')
          self.model.fit(self.coo_data, epochs=4, verbose=True)

          # Сохранение модели в файл
          # with open(modelpath + 'model.pickle', 'wb') as fle:
          #     pickle.dump(model, fle)

          delta_time = time.time() - start_time
          self.run_time += delta_time
          message = 'Model training completed, '
          message += f'total execution time: {round(self.run_time)}s '
          answer = {
              'status': 'OK',
              'message': message
          }
          self.__logging(answer)

          # print('--- МОДЕЛЬ ОБУЧЕНА ---')

          return answer

      # Загрузка модели из файла
      # def __load_model(self, modelpath):
      #
      #     with open(modelpath + 'model.pickle', 'rb') as fle:
      #         model = pickle.load(fle)
      #
      #     # print('--- МОДЕЛЬ ЗАГРУЖЕНА ---')
      #
      #     return model

      def predict(self, user_id):
          n = 10
          user_id = int(user_id)
          try:
              col = self.channel_value
              us_id = self.user_dict[user_id]
              row = np.array([us_id for _ in range(len(col))])
              pred = self.model.predict(row, col, num_threads=os.cpu_count())
              list_pred = list(zip(col, pred))
              sort_pred = sorted(list_pred, reverse=True, key=lambda x: x[1])
              rec_ch_id = [first for first, second in sort_pred][:n]
              rec_channel_ids = [self.channel_dict_rev[id] for id in rec_ch_id]
          except KeyError:
              rec_channel_ids = self.popular_channels[:n]
          except ValueError:
              rec_channel_ids = self.popular_channels[:n]

          return rec_channel_ids