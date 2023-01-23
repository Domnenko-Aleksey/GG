import os
import sys
import time
import datetime
from datetime import datetime
import requests
import pandas as pd
import numpy as np
from typing import Dict, Text
from pymongo import MongoClient

import torch, fastai
from fastai.collab import *
from fastai.tabular.all import *

from recommenders.models.fastai.fastai_utils import cartesian_product, score

# print("System version: {}".format(sys.version))
# print("Pandas version: {}".format(pd.__version__))
# print("Fast AI version: {}".format(fastai.__version__))
# print("Torch version: {}".format(torch.__version__))
# print("Cuda Available: {}".format(torch.cuda.is_available()))
# print("CuDNN Enabled: {}".format(torch.backends.cudnn.enabled))


class ModelFastai:
    def __init__(self, config):
        # print('MODEL Fastai -> INIT')
        self.config = config  # Данные из конфигурационного файла
        self.data_duration = []  # Временное хранилище списка получаемых данных
        self.df_duration = None  # Тут размещаем данные 'duration' в формате Pandas, полученные результате работы метода __get_data_duration
        self.df_final = None # Пост-обработанные данные 'duration' в формате Pandas, полученные результате работы метода __data_processing
        # self.df_website = None  # Тут размещаем данные 'website' в формате Pandas полученные результате работы метода __get_data_website
        self.log = []  # Лог операций
        self.model = None  # Тут будет храниться модель
        self.run_time = 0
        self.files_path = self.config.path + '/ModelFastai'

        # Создаём папку для файлов модели
        if not os.path.isdir(self.files_path):
            os.mkdir(self.files_path, mode=0o755)


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
        answer = self.fit_model()
        if answer['status'] != 'OK':
            return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение

        # # Расчет рекомендаций
        # answer = self.compute_predicts()
        # if answer['status'] != 'OK':
        #     return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение

        return answer


    # Предсказание - переопределяем в наследуемом классе
    def predict(self, user_id):
        start_time = time.time()
        # self.model = load_learner(self.files_path+'/model.pkl')
        n = 10 # self.config.top # сколько каналов рекомендуем 
        # Get all users and items that the model knows /content/корень/ModelFastai/model.pkl

        _, total_items = self.model.classes.values()
        total_items = total_items[1:]
        total_users = [str(user_id)]

        users_items = cartesian_product(np.array(total_users),np.array(total_items))
        users_items = pd.DataFrame(users_items, columns=['userID','itemID'])

        dl_test = self.model.dls.test_dl(users_items, with_labels=True)
        preds = self.model.get_preds(dl=dl_test, with_decoded=True)
        users_items['preds'] = preds[2]
        # users_items.sort_values(by='preds', ascending=False).head(10)

        delta_time = time.time() - start_time

        # print(f'predict execution time: {round(delta_time, 4)}s')

        channel_list=users_items[users_items["userID"]==str(user_id)].sort_values(by='preds', ascending=False).head(n)

        return channel_list['itemID'].tolist()


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

        self.df_final = df_2.drop_duplicates()
        # # print(f'ПОСЛЕ УДАЛЕНИЯ ДУБЛИКАТОВ: {self.df_final.shape[0]}')

        # Смена имени колонок для последущей подачи в модель
        self.df_final=self.df_final.groupby(['user_id', 'channel_id']).sum().reset_index()
        self.df_final['userID']=self.df_final['user_id'].astype(str)
        self.df_final['itemID']=self.df_final['channel_id'].astype(str)
        self.df_final=self.df_final.drop(columns=['user_id'],axis = 1)
        self.df_final=self.df_final.drop(columns=['channel_id'],axis = 1)

        # Подсчет и создание рейтинга

        self.df_final['rating'] = np.where(self.df_final['duration'] < 10*1000*60, 1.0, 
                                          # первый множитель = колво минут просмотра. 
                                  np.where(self.df_final['duration'] < 30*1000*60, 2.0, 
                                          # через запятую = присваиваемый рейтинг.
                                  np.where(self.df_final['duration'] < 60*1000*60, 3.0,
                                          # через вторую запятую пишем, что будет при ложном условии
                                  np.where(self.df_final['duration'] < 120*1000*60,4.0,5.0))))

        # # print(f'колво каналов с рейтингом 5 = {len(self.df_final[self.df_final["rating"]==5.0])}')
        # # print(f'колво каналов с рейтингом 4 = {len(self.df_final[self.df_final["rating"]==4.0])}')
        # # print(f'колво каналов с рейтингом 3 = {len(self.df_final[self.df_final["rating"]==3.0])}')
        # # print(f'колво каналов с рейтингом 2 = {len(self.df_final[self.df_final["rating"]==2.0])}')
        # # print(f'колво каналов с рейтингом 1 = {len(self.df_final[self.df_final["rating"]==1.0])}')

        # чистка лишних столбцов
        self.df_final=self.df_final.drop(columns=['duration'],axis = 1)
        self.df_final=self.df_final.drop(columns=['time'],axis = 1)

        delta_time = time.time() - start_time
        self.run_time += delta_time

        answer = {
            'status': 'OK', 
            'message': f'Data processed, number of rows: {self.df_final.shape[0]}, execution time: {round(delta_time, 4)}s'
        }
        self.__logging(answer)
        return answer


    # === ЗАПИСЬ В ЛОГ ===
    def __logging(self, answer):
        with open(self.files_path + '/status.log', 'a') as file:
            now = datetime.now()
            d = f'{now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second}'
            text = f"{d}, {answer['status']}: {answer['message']} \n"
            file.write(text)



    # === ОБУЧЕНИЕ МОДЕЛИ ===
    def fit_model(self):
        # print('MODEL Fastai -> FIT MODEL')
        start_time = time.time()

        dls = CollabDataLoaders.from_df(self.df_final)

        # embs = get_emb_sz(dls)

        # model = DotProductBias(n_users, n_movies, 50)
        # learn = Learner(dls, model, loss_func=MSELossFlat())
        # learn.fit_one_cycle(5, 5e-3, wd=0.1)

        # инициализация модели
        self.model = collab_learner(dls, n_factors=64, y_range=[0,5.5], wd=1e-1)
        # # print(self.model.model) # справка по модели

        # старт обучения
        self.model.fit_one_cycle(15, 5e-3, wd=0.1)

        # экспорт модели 
        self.model.export(self.files_path+'/model.pkl')

        delta_time = time.time() - start_time
        self.run_time += delta_time

        answer = {
            'status': 'OK', 
            'message': f"Took {round(self.run_time)} seconds for training."
        }
        self.__logging(answer)

        return answer

    # def __cartesian_product(*arrays):
    #     """Compute the Cartesian product in fastai algo. This is a helper function.

    #     Args:
    #         arrays (tuple of numpy.ndarray): Input arrays

    #     Returns:
    #         numpy.ndarray: product

    #     """
    #     la = len(arrays)
    #     dtype = np.result_type(*arrays)
    #     arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    #     for i, a in enumerate(np.ix_(*arrays)):
    #         arr[..., i] = a
    #     return arr.reshape(-1, la)

    # заранее считает все предикты user*item и сохраняет в csv 
    # пока без надобности
    # def compute_predicts(self):
    #     # print('MODEL Fastai -> COMPUTE PREDICTS')
    #     start_time = time.time()

    #     self.model = load_learner(self.files_path, 'model.pkl')

    #     n = self.config.top # сколько каналов рекомендуем 

    #     # Get all users and items that the model knows

    #     total_users, total_items = self.model.data.train_ds.x.classes.values()
    #     total_items = total_items[1:]
    #     total_users = total_users[1:]

    #     users_items = cartesian_product(np.array(total_users),np.array(total_items))
    #     users_items = pd.DataFrame(users_items, columns=[USER,ITEM])

    #     with Timer() as test_time:
    #         top_k_scores = score(self.model, 
    #                             test_df=users_items,
    #                             user_col=USER, 
    #                             item_col=ITEM, 
    #                             prediction_col=PREDICTION,
    #                             top_k=n)

    #     # # print("Took {} seconds to compute {} predictions.".format(test_time, len(users_items)))
    #     # channel_list=top_k_scores[top_k_scores["userID"]==str(user_id)].sort_values(by='prediction', ascending=False)

    #     # # print(channel_list['itemID'].tolist())
    #     top_k_scores.to_csv(self.files_path+'/users_items_predicts.csv')

    #     delta_time = time.time() - start_time
    #     self.run_time += delta_time

    #     answer = {
    #         'status': 'OK', 
    #         'message': f"Took {round(self.run_time)} seconds for predicting."
    #     }
    #     self.__logging(answer)

    #     return answer