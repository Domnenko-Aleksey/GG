#проверенная в колабе версия
#https://colab.research.google.com/drive/15NlqBzOFXjogQWEF5t6yw1eEMmQdSy_C?usp=sharing

class ModelCollaborativeF():
    def __init__(self, config):
        print('MODEL Collaborative Filtration -> INIT')
        self.config = config  # Данные из конфигурационного файла
        self.data_duration = []  # Временное хранилище списка получаемых данных
        self.df_duration = None  # Тут размещаем данные 'duration' в формате Pandas, полученные результате работы метода __get_data_duration
        self.df_website = None  # Тут размещаем данные 'website' в формате Pandas полученные результате работы метода __get_data_website
        self.log = []  # Лог операций
        self.model = None  # Тут будет храниться модель
        self.run_time = 0
        self.files_path = self.config.path + '/ModelCollaborativeF'
        self.popular_list = []  # Тут будет храниться список 100 популярных каналов

        # Создаём папку для файлов модели
        if not os.path.isdir(self.files_path):
            os.makedirs(self.files_path, mode=0o755, exist_ok=True)


    # === ОБУЧЕНИЕ МОДЕЛИ ===
    def fit(self, days=30):
        # Удаляем файл логов
        if os.path.isfile(self.files_path + '/sataus.log'): 
            os.remove(self.files_path + '/sataus.log')

        self.run_time = 0  # Обнуляем время выполнения

        # Получаем данные `duration` за указанное количество дней `days`
        answer = self.__get_data_website(days)
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
    def __get_data_duration(self,  days=30):
        print('MODEL CollaborativeF -> GET DATA duration')

        # !!! БАЗА НЕ АКТУАЛЬНАЯ - ДОБАВЛЯЕМ КОСТЫЛИ ДЛЯ СМЕЩЕНИЯ ПО СРОКАМ НА1 ГОД
        crutch = 86400 * 365

        start_time = time.time()
        from_time = int(time.time()) - int(days)*86400 - crutch
        from_time_url = f'&from={from_time}'
        to_time = int(time.time()) - crutch
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
                    print(i, len(self.data_duration), len(data_list))
                    if len(data_list) < self.config.gg_pagination_step:  # Шаг пагинации
                        print(self.data_duration[0:5])
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

        print('--- ДАННЫЕ "DURATION" ПОЛУЧЕНЫ ---')
        print(f'РАЗМЕР ДАННЫХ: {self.df_duration.shape}, ВРЕМЯ ВЫПОЛНЕНИЯ: {round(delta_time, 4)}')
        print(self.df_duration.head(10))
        return answer 


   # === ПОЛУЧЕНИЕ ДАННЫХ "WEBSITE" MONGODB ЗА НУЖНОЕ КОЛИЧЕСТВО ДНЕЙ ===
    def __get_data_website(self, days=3):
        print('MODEL CollaborativeF -> GET DATA websites')
        start_time = time.time()

        client = MongoClient(self.config.mongo_client_website)
        db = client['stats']
        website = db['website']

        # Отбор по дням
        timestamp = (int(time.time()) - int(days) * 86400) * 1000
        results = website.find({"timestamp": {"$gte": timestamp}})
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



    # === ОБРАБОТКА ДАННЫХ ===
    def __data_processing(self):
        print('MODEL CollaborativeF -> DATA PROCESSING')
        start_time = time.time()

        df_dirty = self.df_website
        df_dirty=df_dirty.loc[~df_dirty.event.isna()]                                               # убираем строки с nan в столбце event
        #df_dirty['streamId'] = df_dirty['streamId'].fillna(df_dirty['streamer'])                    # заполняем streamId значением streamer
        #df_dirty['userId'] = df_dirty['userId'].fillna(df_dirty['user'])                            # заполняем userId значением user
        df_dirty = df_dirty.drop(['_id','ip','[object Object]'], axis=1)                                       # удаляем столбцы streamer и user
        df_dirty = df_dirty.dropna(subset=['userId']) 
        df_dirty = df_dirty.reset_index(drop=True)
        df_dirty.timestamp = pd.to_datetime(df_dirty.timestamp, unit='ms', dayfirst=True)           # переводим timestamp из timestamp в datime
        df_dirty.streamStarted = pd.to_datetime(df_dirty.streamStarted, unit='s', dayfirst=True)    # переводим streamStarted из timestamp в datime
        df_dirty=df_dirty.drop(np.where(df_dirty.event.str.contains('stream-follow'))[0])           # очищаем общий датафрейм от событий stream-follow
        df_dirty['sessionId'] = df_dirty.apply(lambda row: str(row['streamId']) + str(row['userId']), axis=1) # создаем уникальную метку для определения пользовательского просмотра. Меткой является склейка streamId и UserId
        Session_list=df_dirty.sessionId.unique()
        #print('Количество сессий', len(Session_list))
        User_list=df_dirty.userId.unique()

# определяем продолжительность просмотров (выполняется ДОЛГО)
        df_result = pd.DataFrame(columns=['userId', 'streamId', 'sessionId', 'duration'])
        df_result.set_index('userId')
        i_time = time.time()
        for i in range(len(User_list)):  
          df_user = df_dirty[df_dirty['userId'].str.contains(User_list[i])]
          cs_for_user = df_user['sessionId'].unique()
          for k in range(len(cs_for_user)):
            df_cs_for_user = df_user[df_user['sessionId'] == cs_for_user[k]]
            cs_start = df_cs_for_user['timestamp'].min()
            #print('cs_start', cs_start)
            cs_end = df_cs_for_user['timestamp'].max()
            #print('cs_end', cs_end)
            cs_duration = cs_end - cs_start 
            cs_date = cs_start.date()
            row = {'userId':User_list[i], 'streamId':df_cs_for_user['streamId'].unique()[0], 'sessionId':cs_for_user[k], 'duration':cs_duration}
            df_result.loc[len(df_result.index)] = row
# это для мониторинга процесса
            if k/250 == k // 250: print('сеанс:',k , 'из', len(cs_for_user))                    
          if i > 0: print('пользователь:',i , 'из', len(User_list), 'ETA:', (((time.time() - i_time)/i)*len(User_list))/60/60, 'часов')

        df_result['duration'] = df_result['duration'] / np.timedelta64(1, 'ms') #datetime конвертируем в количество часов

# теперь надо учесть влияние событий subscribe, message и donate
# для этого длительность стрима, во время которого произошел subscribe умножаем на S
# длительность стрима, во время которого произошел message умножаем на M
# длительность стрима, во время которого произошел donate умножаем на D
        S = 1.5
        M = 1.01
        D = 2

        cur_time = time.time()

        subscribe_list = df_dirty.loc[df_dirty.event=='stream-subscribe'].sessionId.unique()
        message_list = df_dirty.loc[df_dirty.event=='chat-message'].sessionId.unique()
        donat_list = df_dirty.loc[df_dirty.event=='stream-donat'].sessionId.unique()

        for i in range(len(subscribe_list)):
          res=df_dirty.loc[((df_dirty.event=='stream-subscribe') & (df_dirty.sessionId==subscribe_list[i]))]
          sId=res.sessionId.loc[res.index[0]]
          df_result.loc[df_result.sessionId==sId, 'duration'] *= S
          if i/10 == i//10: print('stream-subscribe', i, 'from', len(subscribe_list))

        for i in range(len(message_list)):
          res=df_dirty.loc[((df_dirty.event=='chat-message') & (df_dirty.sessionId==message_list[i]))]
          sId=res.sessionId.loc[res.index[0]]
          df_result.loc[df_result.sessionId==sId, 'duration'] *= M
          if i/50 == i//50: print('chat-message', i, 'from', len(message_list))

        for i in range(len(donat_list)):
          res=df_dirty.loc[((df_dirty.event=='stream-donat') & (df_dirty.sessionId==donat_list[i]))]
          sId=res.sessionId.loc[res.index[0]]
          df_result.loc[df_result.sessionId==sId, 'duration'] *= D
          if i/10 == i//10: print('stream-donat', i, 'from', len(donat_list))

        print(time.time() - cur_time, 'c.')


        df_result.rename(columns={'Длительность сеанса': 'duration', 'streamId': 'Название стрима', 'userId': 'Имя пользователя'}, inplace=True)
        
        streams_list = df_result['Название стрима'].unique()
        print('Длина списка стримов', len(streams_list))
        users_list = df_result['Имя пользователя'].unique()
        print('Длина списка пользователей', len(users_list))
# для оптимизации скорости работы рек.системы принято решение 
# не учитывать пользователей, которые смотрели не более Х стримов
# не учитывать стримы, которые смотрели менее Y раз
# не учитывать просмотры длительностью менее Z миллисекунд
        X = 3
        Y = 5
        Z = 5*60000 # 5 секунд

        r = df_result.groupby('Имя пользователя').filter(lambda d: len(d) > X)
        print(r.shape)
        r = r.groupby('Название стрима').filter(lambda d: len(d) > Y)
        print(r.shape)
        r = r[~(r['duration'] < Z)]
        print(r.shape)

        print('Количество пользователей:', len(r['Имя пользователя'].unique()),'\nКоличество стримов:', len(r['Название стрима'].unique()))

        result=r.groupby(['Название стрима','Имя пользователя']).sum().reset_index()
        if not os.path.exists(self.files_path): os.makedirs(self.files_path)       
        result.to_csv(self.files_path + '/ggr_result_limited.csv', index=False, header=False)

        self.df_website = result.copy()
        
        # cписок рекомендаций
        self.popular_list = result.groupby(['Название стрима'])['duration'].sum().sort_values(ascending=False)[0:100].index.values.tolist()
        
        delta_time = time.time() - start_time
        self.run_time += delta_time
        answer = {
            'status': 'OK', 
            'message': f'Data processed, number of rows: {self.df_website.shape[0]}, execution time: {round(delta_time, 4)}s'
        }
        self.__logging(answer)

        return answer


    # === ЗАПИСЬ В ЛОГ ===
    def __logging(self, answer):
        with open(self.files_path + '/sataus.log', 'a') as file:
            now = datetime.datetime.now()
            d = f'{now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second}'
            text = f"{d}, {answer['status']}: {answer['message']} "
            file.write(text)


    # === ОБУЧЕНИЕ МОДЕЛИ ===
    def __fit_model(self, days=30):
        print('MODEL Collaborative Filtration -> Fit')
        start_time = time.time()

        #df = self.df_website
        #print(df)
        #input()


        
        def ReadFile (filename = self.files_path + "/ggr_result_limited.csv"):
            f = open (filename)
            r = csv.reader (f)
            mentions = dict()
            for line in r:
                user    = line[1]
                product = line[0]
                rate    = float(line[2])
                if not user in mentions:
                    mentions[user] = dict()
                mentions[user][product] = rate
            f.close()
            return mentions

        def distCosine (vecA, vecB):
            def dotProduct (vecA, vecB):
                d = 0.0
                for dim in vecA:
                    if dim in vecB:
                        d += vecA[dim]*vecB[dim]
                return d
            return dotProduct (vecA,vecB) / math.sqrt(dotProduct(vecA,vecA)) / math.sqrt(dotProduct(vecB,vecB))

        def makeRecommendation (userID, userRates, nBestUsers, nBestProducts):
            if userRates.get(userID) != None:         # если пользователь найден
              matches = [(u, distCosine(userRates[userID], userRates[u])) for u in userRates if u != userID]
              bestMatches = sorted(matches, key=itemgetter(1), reverse=True)[:nBestUsers]
              sim = dict()
              sim_all = sum([x[1] for x in bestMatches])
              bestMatches = dict([x for x in bestMatches if x[1] > 0.0])
              for relatedUser in bestMatches:
                for product in userRates[relatedUser]:
                    if not product in userRates[userID]:
                        if not product in sim:
                            sim[product] = 0.0 
                        sim[product] += userRates[relatedUser][product] * bestMatches[relatedUser] 
              for product in sim:
                sim[product] /= sim_all
              sim_l=list(sim.items())
              bestProducts = sorted(sim_l, key=itemgetter(1), reverse=True)[:nBestProducts]
            else:                                   # если пользователь не найден
              resultdict = {}                       # строим словарь с рейтингом всех стримеров без учета конкретных пользовательских предпочтений
              for user in userRates:
                for product in userRates[user]:
                  try:
                    resultdict[product] += userRates[user][product]   # складываем значения
                  except KeyError:                                    # если ключа еще нет - создаем
                    resultdict[product] = userRates[user][product] 
              bp_l=list(resultdict.items())
              bestProducts = sorted(bp_l, key=itemgetter(1), reverse=True)[:nBestProducts]
            return [(x[0], x[1]) for x in bestProducts]        
# функция учета реакции пользователя на сделанные рекомендации (кликнул - успех, проигнорировал - неудача)
# идея - предрасcчитать рекомендации, сохранить в виде "стрим", "коэффициент интереса" в порядке убывания коэффициента
# "коэффициент интереса" понижать на некий коэффициент в случае игнора выданной рекомендации. 
# Например, пополамить при каждой неудаче
# в этом случае приоритет выдачи конкретной рекомендации при накоплении неудач будет экспоненциально стремиться к нулю

# вопрос - хотим ли мы возврата стрима в рекомендации через какое-то время? 
# если да, то надо добавить механизм "устаревания" понижающего коэффициента

# предрассчитанные рекомендации сложить в файлы с именами вида userid.csv (или БД?)
# в эти же файлы складывать результаты учета реакции пользователя

# предрасчет рекомендаций по всем пользователям
# 40 минут

# Количество пользователей: 
        X = 3172 
# Количество стримов: 
        Y = 1087

        RF = ReadFile()
        i = 1
        b_time = time.time()
        for users in RF:
          cur_time = time.time()
          rec = makeRecommendation (users, RF, X, Y)
          if not os.path.exists(self.files_path + '/CR/'): os.makedirs(self.files_path + '/CR/') 
          filename = self.files_path + '/CR/'+users+'.csv'
          with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(rec)  
          print(users, ',', i, 'из', len(RF), 'количество строк-рекомендаций', len(rec), ', время вычисления: ', time.time() - cur_time, 'c.', 'ETA:', ((time.time()-b_time)/60)/60*(len(RF)/i), 'часов')
          i += 1

# предрасчет рекомендации для тех, о ком не известно ничего
        rec = makeRecommendation ('', RF, X, Y)
        if not os.path.exists(self.files_path + '/CR_for_unknown/'): os.makedirs(self.files_path + '/CR_for_unknown/')
        filename = self.files_path + '/CR_for_unknown/user.csv'
        with open(filename, 'w', newline='') as csv_file:
          csv_writer = csv.writer(csv_file)
          csv_writer.writerows(rec)  

# удаление файлов старше 2-х суток
        path = self.files_path + '/CR/'
        now = time.time()
        for filename in os.listdir(path):
            if os.path.getmtime(os.path.join(path, filename)) < now - 2 * 86400:
                if os.path.isfile(os.path.join(path, filename)):
                    print(filename)
                    os.remove(os.path.join(path, filename))

        delta_time = time.time() - start_time
        self.run_time += delta_time
        message =   'Model computing completed, '
        message +=  f'total execution time: {round(self.run_time)}s '
        answer = {
            'status': 'OK', 
            'message': message
        }
        self.__logging(answer)

        return answer


    # Предсказание - переопределяем в наследуемом классе
    def predict(self, user_id):
        username = user_id
        number = 10
        if os.path.isfile(self.files_path + '/CR/'+username+'.csv'): 
           filename = self.files_path + '/CR/'+username+'.csv'
           unknown_user = False
        else: 
           filename = self.files_path + '/CR_for_unknown/user.csv'
           unknown_user = True

        f = open (filename)
        r = csv.reader (f)
        mentions = dict()
        for line in r:
            product = line[0]
            rate    = float(line[1])
            if not product in mentions:
                mentions[product] = dict()
            mentions[product] = rate #, coeff
        f.close()
        rec_products = dict()
        aaa13 = []
        for i in range(number):
          aaa13.append(list(mentions.items())[i][0])
        rec_products[username] = aaa13
        if unknown_user == True: 
          fn = self.files_path + '/CR/'+username+'.csv' #если пользователя нет - создаем ему новый файл из общих рекомендаций для всех
    #resorted_result = sorted(mentions.items(), key=itemgetter(1), reverse=True) #пересортируем словарь
          with open(fn, 'w', newline='') as csv_file: #записываем результат в файл для последующего использования
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(mentions.items())  


        answer = aaa13

        return answer




# ======= НЕ СОЗДАЁМ МОДЕЛЬ, поскольку в моем подходе ее не существует =======

class CreateModel(ModelCollaborativeF):
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
