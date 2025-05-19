import json

import dill

import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI ()
with open ( 'best_model_pipeline.pkl', 'rb' ) as f:
    best_model_forest, metadata = dill.load ( f )

expected_columns = getattr(metadata, 'expected_columns', None)

class Form ( BaseModel ):
    utm_source:       str
    utm_medium:       str
    utm_campaign:     str
    utm_adcontent:    str
    utm_keyword:      str
    device_category:  str
    device_os:        str
    device_brand:     str
    device_model:     str
    device_browser:   str
    device_screen_resolution: str
    geo_country:      str
    geo_city:         str



class Prediction ( BaseModel ):
     pred: str


@app.get ( '/status' )
def status():
    return "I'm OK"

@app.get ( '/version' )
def version():
    return metadata

@app.post ( '/predict', response_model=Prediction )
def predict(form: Form):
    input_data = form.dict()
    df = pd.DataFrame([input_data])

    # Добавление признака "квартал" (1-4)
    df['quarter'] = (datetime.now().month - 1) // 3 + 1
    df['visit_number'] = 1
    # Месяц
    df['month'] = datetime.now().month
    # 0=понедельник, 6=воскресенье
    df['day_of_week'] = datetime.now().weekday()
    # Добавление признака "номер недели в месяце" (1-4)
    df['week_in_month'] = (datetime.now().day - 1) // 7 + 1
    # Час
    df['hour']=datetime.now().hour
    # Создаем числовой признак (кодировку)
    df['time_of_day'] = (
        df['hour']
        .apply(lambda x:
               0 if 0 <= x < 6 else  # Ночь (0-6)
               1 if 6 <= x < 12 else  # Утро (6-12)
               2 if 12 <= x < 18 else  # День (12-18)
               3  # Вечер (18-24)
               )
    )

    # Безопасное вычисление пикселей с обработкой возможных ошибок
    df['screen_pixels'] = (
        df['device_screen_resolution']
        .str.split('x')
        .apply(lambda x: int(x[0]) * int(x[1]) if x and len(x) == 2 else None)
        .fillna(0)  #  значение по умолчанию
    )

    df.drop(['device_screen_resolution'], axis=1, inplace=True)

    #  Предсказание зависит от даты и времени  Для теста положительного  предсказания можно использовать:
    # df['visit_number'] = 1
    # df['month'] = 6
    # df['day_of_week'] = 0
    # df['week_in_month'] = 2
    # df['hour'] = 9
    # df['time_of_day'] = 1

    y = best_model_forest.predict(df)
    return {'pred': str(y[0])}
