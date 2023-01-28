import pickle
import requests
import time
import os

import pandas as pd
import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LambdaCallback

from sklearn import cluster, mixture
import warnings


class NNHS():  # Nearest Neighbors in a Hidden Space
    def __init__(self, config):
        self.config = config
        self.path = config.path
        self.days = 120
        self.sup = 2 * 24 * 60 * 60  # сек
        self.inf = 4 * 60  # сек
        self.streamers_population_graphs = []
        self.streamers_mean_online = np.array([])
        self.count_views = []
        self.unique_streamers = []
        self.durations = pd.DataFrame()
        self.popular_list = []
        self.len_recommends_list = 20  # Колличество рекоммендуемых каналов
        # ++++++++++++++++++++++++++++++++++
        self.model = None
        self.modelencoder = None
        self.modelcluster = None
        self.M = 3
        self.X_train = []
        self.streamers2users = []
        self.unique_users = []
        self.XYZ_dataset = []
        # +++++++++++++++++++++++++++++++++++
        self.klasterer = None
        self.clusters = 40
        self.cluster_dataset = []
        self.clusters_XYZ = []
        self.centroid_clusters = {}
        # +++++++++++++++++++++++++++++++++++
        self.cur_epoch = 0
        self.df_rec = pd.DataFrame()
        self.df_cold_rec = pd.DataFrame()

    # Подгрузка базы duration в память

    def get_data(self):
        from_time = int(time.time()) - int(self.days) * 86400
        from_time_url = f'&from={from_time}'
        to_time = int(time.time())
        from_to_time = f'&to={to_time}'
        data_duration = []

        # --- Соединение с GG Api и получение данных ---
        for i in range(1, self.config.gg_pagination_step):
            url = self.config.gg_api_url + from_time_url + from_to_time + '&page=' + str(i)
            try:
                api_req = requests.get(url)
                if int(api_req.status_code) != 200:
                    result = {
                        'status': 'error',
                        'message': 'Check your API connection',  # Сообщение, необязательныый атрибут
                        'data': pd.DataFrame()
                        # Необработанные предварительно данные в формате dataframe, эти данные подаются на вход метода
                    }
                    return result
                else:
                    # Добавляем данные в наш список
                    data_list = api_req.json()
                    data_duration.extend(data_list)
                    if len(
                            data_list) < self.config.gg_pagination_step or i == self.config.gg_pagination_step - 1:  # Шаг пагинации
                        df_duration = pd.DataFrame.from_dict(data_duration)
                        self.durations = df_duration
                        result = {
                            'status': 'success',
                            'message': 'Dataset has been installed',  # Сообщение, необязательныый атрибут
                            'data': df_duration
                            # Необработанные предварительно данные в формате dataframe, эти данные подаются на вход метода
                        }
                        return result
            except Exception as e:
                result = {
                    'status': 'error',
                    'message': e,  # Сообщение, необязательныый атрибут
                    'data': df_duration
                    # Необработанные предварительно данные в формате dataframe, эти данные подаются на вход метода
                }
                return result

    # Приведение изначального датасета к нормальной форме

    def data_processing(self, df_input):
        self.durations = df_input

        if self.durations.empty:
            result = {
                'status': 'error',
                'message': 'The Dataset is empty. Please try install duration again and repeat this operation',
                # Сообщение, необязательныый атрибут
                'data': self.durations  # Обработанный датафрейм
            }
            return result
        else:
            # Удаление выбросов
            count0 = self.durations.shape[0]
            count1 = count0
            self.durations = self.durations[
                self.durations['duration'] < self.sup * 1000]  # Меньше точной верхней границы
            self.durations = self.durations[
                self.durations['duration'] > self.inf * 1000]  # Больше точной нижней границы

            # Изменение формата отображения времени в базе
            self.durations['duration'] = pd.to_datetime(self.durations['time'] + (self.durations['duration'] / 1000),
                                                        unit='s')
            self.durations['time'] = pd.to_datetime(self.durations['time'], unit='s')

            # Сортировака по увеличению времени захода на стрим (прямой временной проядок)
            self.durations.columns = ['join_time', 'ID_streamer', 'ID_user', 'close_time']
            self.durations = self.durations.sort_values(by=['join_time']).reset_index()
            self.durations.drop(columns=['index'], inplace=True)
            # print(f'Стримеров было: {pd.unique(self.durations["ID_streamer"]).shape}')
            # Удаление нецифровых индексов стримеров
            valid_streamers_id = []
            for streamer_id in pd.unique(self.durations['ID_streamer']):
                try:
                    valid_streamers_id.append(str(int(streamer_id)))
                except:
                    pass
            self.durations = self.durations.loc[self.durations['ID_streamer'].isin(valid_streamers_id)]
            count1 -= self.durations.shape[0]
            # print(f'Стримеров стало: {pd.unique(self.durations["ID_streamer"]).shape}')
            # print(f'Было удалено {count1} из {count0} записей с аномальными значениями.')

            result = {
                'status': 'success',
                'message': 'Dataset has been processed',  # Сообщение, необязательныый атрибут
                'data': self.durations  # Обработанный датафрейм
            }
            return result

    # Функция расчёта базы с динамикой онлайна во времени для каждого стримера

    def _streamer_population(self):
        # set_streamers = set(pd.unique(db['ID_streamer']))
        unique_streamers = pd.unique(self.durations['ID_streamer'])
        db_population = []
        progress = 0
        for streamer_id in unique_streamers:
            progress += 1
            #if progress % 10 == 0:
                # print(f'{progress}/{len(unique_streamers)}')
            loc_db = self.durations[self.durations['ID_streamer'] == streamer_id]

            db_join = pd.DataFrame(loc_db['join_time'])
            db_join.columns = ['time']
            db_join['join/close'] = np.ones([loc_db.shape[0]], dtype='uint8')

            db_close = pd.DataFrame(loc_db['close_time'])
            db_close.columns = ['time']
            db_close['join/close'] = np.ones([loc_db.shape[0]], dtype='uint8') * -1

            db_concatenate_join_close = pd.concat([db_join, db_close], axis=0).sort_values(by=['time'])

            count = 0
            population = []
            n = 0
            for i in db_concatenate_join_close['join/close'].values:
                count += i
                population.append([db_concatenate_join_close['time'].iloc[n], count])
                n += 1
            db_population.append([streamer_id, population])

        return db_population

    # Функция расчёта временной зависимости онлайна стримеров
    def _calculate_streamer_population_graphs(self, refresh_data=False, filepath=None):
        assert self.durations.shape[0] > 0  # Для совершения этого действия необходимо заполнить durations
        self.streamers_population_graphs = self._streamer_population()
        return None

    # Создание базы со средним онлайном для каждого стримера
    def _mean_population(self):
        if len(self.streamers_population_graphs) <= 0:
            self._calculate_streamer_population_graphs()

        streamers_population = []
        for i in self.streamers_population_graphs:
            mean_population = np.mean([i[1][j][1] for j in range(len(i[1]))])
            streamers_population.append([i[0], round(mean_population, 3)])

        # Сортируем стримеров по среднему онлайну
        streamers_population = sorted(streamers_population, key=lambda x: x[1], reverse=True)
        self.streamers_mean_online = np.array([i[0] for i in streamers_population])
        return None

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _count_views_calculate(self):
        assert self.durations.shape[0] > 0

        self.count_views = self.durations.groupby('ID_user').count()
        self.count_views.drop(columns=['join_time', 'close_time'], inplace=True)
        self.count_views.columns = ['count_views']
        self.unique_users = self.count_views.sort_values(by='count_views', ascending=False).loc[
            self.count_views.sort_values(by='count_views', ascending=False)['count_views'] > self.M].index
        return None

    # Функция преобразования индексов стримеров в вектор one hot encoder
    def _to_ohe(self, data, length):
        np_array = np.zeros(length, dtype='uint8')
        for i in data:
            np_array[i] = 1
        return np_array

    # Создание датасета для автоенкодера
    def _get_dataset2AE(self):
        self._count_views_calculate()
        self.unique_streamers = pd.unique(
            self.durations[self.durations['ID_user'].isin(self.unique_users)]['ID_streamer'])
        new_db = []
        db_array_streamers = []
        for user in self.unique_users:
            unique_str = pd.unique(self.durations[self.durations['ID_user'] == user]['ID_streamer'])
            new_db.append(self._to_ohe(
                [pd.DataFrame(self.unique_streamers)[self.unique_streamers == unique_str[i]].index[0] for i in
                 range(len(unique_str))], self.unique_streamers.shape[0]))
            db_array_streamers.append([user,
                                       [self.unique_streamers[np.where(self.unique_streamers == unique_str[i])[0][0]]
                                        for i in range(len(unique_str))]])
        self.X_train = np.array(new_db)
        self.streamers2users = db_array_streamers
        return None

    def create_model(self):
        self._get_dataset2AE()
        try:
            activation = 'relu'
            latent_dim = 32
            first_layer = 256
            second_layer = 1024
            # Кодировщик
            input_encoder = Input(len(self.X_train[0]))

            coder = Dense(first_layer, activation=activation)(input_encoder)
            coder = Dense(second_layer, activation=activation)(coder)
            output_encoder = Dense(latent_dim, activation=activation)(coder)

            encoder_model = Model(input_encoder, output_encoder)

            # Декодер
            input_decoder = Input(latent_dim)

            decoder = Dense(second_layer, activation=activation)(input_decoder)
            decoder = Dense(first_layer, activation=activation)(decoder)
            output_decoder = Dense(len(self.X_train[0]), activation=activation)(decoder)

            decoder_model = Model(input_decoder, output_decoder)

            # Автокодировщик
            model = Model(input_encoder, decoder_model(encoder_model(input_encoder)))
            model.summary()

            # Компиляция
            model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
            self.model = model
            self.modelencoder = encoder_model
            result = {
                'status': 'success',
                'message': 'Autoencoder has been created',
                'data': [model, encoder_model]}  # Возвращает данные для построения графика обучения модели
            return result
        except Exception as e:
            result = {
                'status': 'error',
                'message': e,
                'data': [None, None]}  # Возвращает данные для построения графика обучения модели
            return result

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _fit_klaster(self):
        self.klasterer = mixture.GaussianMixture(n_components=self.clusters, covariance_type='full')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message="the number of connected components of the " +
                                            "connectivity matrix is [0-9]{1,2}" +
                                            " > 1. Completing it to avoid stopping the tree early.",
                                    category=UserWarning)

            warnings.filterwarnings("ignore",
                                    message="Graph is not fully connected, spectral embedding" +
                                            " may not work as expected.",
                                    category=UserWarning)

            self.klasterer.fit(self.XYZ_dataset)

        # print('Тренировка кластеризатора завершена')

        if hasattr(self.klasterer, 'labels_'):
            self.cluster_dataset = pd.DataFrame(self.klasterer.labels_.astype(int), columns=['clasters'])
        else:
            self.cluster_dataset = pd.DataFrame(self.klasterer.predict(self.XYZ_dataset), columns=['clasters'])
        self._get_centroids()
        return None

    def _distance(self, first_point, second_point):
        if len(first_point.shape) > 1:
            return np.sqrt(np.sum((first_point - second_point) ** 2, axis=1))
        else:
            return np.sqrt(np.sum((first_point - second_point) ** 2))

    def _get_centroids(self):
        self.clusters_XYZ = pd.DataFrame(pd.Series([self.XYZ_dataset[i] for i in range(len(self.XYZ_dataset))]),
                                         columns=['XYZ']).join(pd.DataFrame(self.cluster_dataset, columns=['clasters']))
        unique_clusters = pd.unique(self.cluster_dataset['clasters'])

        for cluster in unique_clusters:
            array_XYZ = np.array(
                [self.clusters_XYZ[self.clusters_XYZ.clasters == cluster]['XYZ'].values[i].tolist() for i in
                 range(self.clusters_XYZ[self.clusters_XYZ.clasters == cluster].shape[0])])
            self.centroid_clusters[int(cluster)] = np.mean(array_XYZ, axis=0)
        self.centroid_clusters = dict(sorted((self.centroid_clusters.items())))
        return None

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _create_DF(self):
        # print('начало сбора базы рекомендаций')
        # Формирование основы датасета
        data = pd.DataFrame(self.streamers2users, columns=['ID_user', 'drop'])
        data.drop(columns=['drop'], inplace=True)
        # Создание Фреймов из кластеров и координат
        df_XYZ = pd.DataFrame(pd.Series([self.XYZ_dataset[i] for i in range(len(self.XYZ_dataset))]), columns=['XYZ'])
        # Устанавливаем юзеров как индексы для координат
        df_XYZ = df_XYZ.join(data)
        df_XYZ = df_XYZ.set_index('ID_user')

        # Создание датасета стримеров, которых смотрит юзер
        # df_streamers = get_dataset(df_durations)
        # Соединение Юзеров и Кластеров
        df_res = data.join(self.cluster_dataset)
        df_res = df_res.set_index('ID_user')
        df_streamers = pd.DataFrame(self.streamers2users, columns=['ID_user', 'array_streamers']).set_index('ID_user')
        count_views = self.durations[['ID_user', 'ID_streamer']].groupby('ID_user').count()
        count_views.columns = ['count_views']
        # Соединие по индексам юзеров, кластеров и координат
        df_res = df_res.join(df_streamers, on='ID_user')
        df_res = df_res.join(df_XYZ)
        df_res = df_res.join(count_views)

        # Удаление незаполненных полей
        self.df_rec = df_res.dropna()
        # print('База данных для рекомендаций собрана')
        return None

    def _create_DF_cold_start(self):
        # print('начало сбора базы рекомендаций для холодных пользоваетелй')
        # Создание датасета с юзерами и соединение его с датасетом кластеров
        data = pd.DataFrame(self.streamers2users, columns=['ID_user', 'drop'])
        data.drop(columns=['drop'], inplace=True)
        df_res = data.join(self.cluster_dataset)

        # Установка индексов юзеров в индексы датасета
        df_res = df_res.set_index('ID_user')
        df_streamers = pd.DataFrame(self.streamers2users, columns=['ID_user', 'ID_streamers']).set_index('ID_user')

        # Соединение Кластеров и Стримеров
        df_res = df_res.join(df_streamers, on='ID_user')
        df_res.dropna(inplace=True)

        # формирование единых массивов стримеров для каждого кластера
        df_res = df_res.groupby(['clasters']).sum()

        values_res = df_res.values

        # Загрузка датасета со стримерами по популярности
        self._mean_population()  # Стримеры сортируются по популярности: self.streamers_mean_online

        new_arrays = []
        for cluster in range(self.clusters):
            streamers_in_cluster_1 = np.array([[i] for i in set(values_res[cluster][0])])
            streamers_in_cluster_2 = np.array([i for i in set(values_res[cluster][0])])
            indexes = np.where(self.streamers_mean_online == streamers_in_cluster_1)[1].argsort()
            new_arrays.append(streamers_in_cluster_2[indexes])

        df_res = pd.DataFrame(np.array(new_arrays, dtype='object'), columns=['array_streamers'])
        self.df_cold_rec = df_res
        # print('База данных для холодный рекомендаций собрана')
        return None

        # Пользовательский колбек

    def _ae_on_epoch_end(self, epoch, logs):
        # print('________________________')
        # print(f'*** ЭПОХА: {epoch + 1}, loss: {logs["loss"]} ***')
        # print('________________________')
        self.cur_epoch = epoch

    def fit(self, days=120):
        self.days = days
        # Загрузка датасета
        df_input = self.get_data()['data']
        # Препроцессинг датасета
        self.data_processing(df_input)
        # Создание модели автоенкодера
        self.create_model()

        # Обучение
        self.cur_epoch = 0
        try:
            # Обучение распределения пользователей в n мерном пространстве
            # Компиляция пользовательского колбека
            ae_callback = LambdaCallback(on_epoch_end=self._ae_on_epoch_end)

            epochs = 200
            verbose = 0
            history = []

            history = self.model.fit(self.X_train, self.X_train,
                                     epochs=epochs,
                                     batch_size=256,
                                     shuffle=True,
                                     callbacks=[ae_callback],
                                     verbose=verbose)

            # Создаём скрытое n мерное пространство для всех пользователей
            self.XYZ_dataset = self.modelencoder.predict(self.X_train)

            # Обучение кластеризатора
            self._fit_klaster()
            # Формирование таблиц для формирования предсказания
            self._create_DF()
            self._create_DF_cold_start()

            # Формирование списка популярных каналов
            j = 0
            for i in range(len(self.df_cold_rec.iloc[j][0])):
                for j in range(self.clusters):
                    if (len(self.popular_list) < self.len_recommends_list):
                        try:
                            if self.df_cold_rec.iloc[j][0][i] not in self.popular_list:
                                self.popular_list.append(self.df_cold_rec.iloc[j][0][i])
                        except:
                            continue
                    else:
                        break

            result = {
                'status': 'success',
                'message': 'All models have been fitted',
                'epochs_count': epochs,  # Количество эпох обучения
                'epochs_current': self.cur_epoch,  # Текущую эпоху обучения
                # 'data': history  # 'data': history #  Object of type History is not JSON serializable  Возвращает данные для построения графика обучения модели
            }
            return result
        except Exception as e:

            result = {
                'status': 'error',
                'message': e,
                'epochs_count': epochs,  # Количество эпох обучения
                'epochs_current': self.cur_epoch,  # Текущую эпоху обучения
                # 'data': history #  Object of type History is not JSON serializable # Возвращает данные для построения графика обучения модели
            }  
            return result

    def save_model(self):
        try:
            self.model.save('model_ae')
            with open('klasterer.pickle', 'wb') as f:
                pickle.dump(self.klasterer, f)
            with open('centroid_clusters.pickle', 'wb') as f:
                pickle.dump(self.centroid_clusters, f)
            self.df_rec.to_json('df_rec.json')
            self.df_cold_rec.to_json('df_cold_rec.json')
            result = {
                'status': 'success',
                'message': ''
            }
            return result
        except Exception as e:
            # print(e)
            result = {
                'status': 'error',
                'message': e
            }
            return result

    def load_model(self):
        try:
            loaded_model = load_model('model_ae')
            with open('klasterer.pickle', 'rb') as f:
                klasterer = pickle.load(f)
            with open('centroid_clusters.pickle', 'rb') as f:
                centroid_clusters = pickle.load(f)
            df_rec = pd.read_json(os.path.join(self.path, 'df_rec.json'))
            df_cold_rec = pd.read_json(os.path.join(self.path, 'df_cold_rec.json'))
            return [loaded_model, klasterer, df_rec, df_cold_rec, centroid_clusters]
        except Exception as e:
            # print(f'Ошибка: {e}')
            return [None, None, None, None]

    def predict(self, user_id):  # Индекс юзера, для котороге делается предсказание
        # self.df_rec = model[2]
        # self.df_cold_rec = model[3]
        # self.centroid_clusters = model[4]
        m = 4
        db_copy = self.df_rec[self.df_rec['count_views'] > m]

        centroids = np.array(list(self.centroid_clusters.values()))
        set_streamers = []
        coordinates = []
        if user_id in db_copy.index:
            # Тёплый пользователь
            user_raw = self.df_rec[self.df_rec.index == user_id]
            user_cluster = user_raw['clasters'].values[0]
            user_coordinates = user_raw['XYZ'].values[0]
            indexes = np.argsort(self._distance(centroids, user_coordinates))

            for cluster in indexes[:5]:
                db_part = db_copy[db_copy['clasters'] == cluster]
                coordinates = coordinates + db_part['XYZ'].values.tolist()
                set_streamers = set_streamers + db_part['array_streamers'].values.tolist()

            coordinates = np.array(coordinates)
            indexes_2 = self._distance(coordinates, user_coordinates)
            # indexes_2 = indexes_2[np.where(indexes_2 != 0)]
            indexes_2 = np.argsort(indexes_2)

            set_streamers = np.array(set_streamers, dtype='object')[indexes_2]

            chanels_list = []

            for arr_streamers in set_streamers:
                for streamer in arr_streamers:
                    if streamer not in chanels_list:
                        chanels_list.append(streamer)
            return chanels_list[:self.len_recommends_list]
        else:
            # Холодный пользователь
            chanels_list = []
            j = 0
            for i in range(len(self.df_cold_rec.iloc[j][0])):
                for j in range(self.clusters):
                    if (len(chanels_list) < self.len_recommends_list):
                        try:
                            if self.df_cold_rec.iloc[j][0][i] not in chanels_list:
                                chanels_list.append(self.df_cold_rec.iloc[j][0][i])
                        except:
                            continue
                    else:
                        return chanels_list