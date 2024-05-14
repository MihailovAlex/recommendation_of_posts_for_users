import hashlib
from loguru import logger
from typing import List
from fastapi import FastAPI, Depends

import os
import pickle

import catboost
import pandas as pd

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey,func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

from pydantic import BaseModel
from schema import PostGet
from datetime import datetime


'''  Database  '''

SQLALCHEMY_DATABASE_URL = ''

engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_size=30, max_overflow=20)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


'''  table post'''

class Post(Base):
    __tablename__ = 'post'
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)


if __name__ == "__main__":
    session = SessionLocal()
    result = session.query(Post).filter(Post.topic == 'business').order_by(Post.id.desc()).limit(10).all()
    list_id = [r.id for r in result]
    print(list_id)


'''  table user'''

class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    gender = Column(Integer)
    age = Column(Integer)
    country = Column(String)
    city = Column(String)
    exp_group = Column(Integer)
    os = Column(String)
    source = Column(String)


if __name__ == "__main__":
    session = SessionLocal()
    result = (session
              .query(User.country, User.os, func.count("*"))
              .filter(User.exp_group == 3)
              .group_by(User.country, User.os)
              .having(func.count("*") > 100)
              .order_by(func.count("*").desc())
              .all()
              )
    print(result)


'''  table feed'''

class Feed(Base):
    __tablename__ = 'feed_action'
    user_id = Column(Integer, ForeignKey(User.id), primary_key=True)
    user = relationship("User")
    post_id = Column(Integer, ForeignKey(Post.id), primary_key=True)
    post = relationship("Post")
    action = Column(String)
    time = Column(DateTime)
    target = Column(Integer)

    
'''  schema  '''

class UserGet(BaseModel):
    id: int
    gender: int
    age: int
    country: str
    city: str
    exp_group: int
    os: str
    source: str

    class Config:
        orm_mode = True


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

        
class FeedGet(BaseModel):
    user_id: int
    user: UserGet
    post_id: int
    post: PostGet
    action: str
    time: datetime
    target: int

    class Config:
        orm_mode = True


class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]
        
        
'''  App  '''

app = FastAPI()

def get_db():
    with SessionLocal() as db:
        return db


""" Загрузка модели (для двух моделей) """

# Функция для проверок в системе
def get_model_path(path: str, exp_group:str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально.
        if exp_group == 'control':
            MODEL_PATH = '/workdir/user_input/model_control'
        elif exp_group == 'test':
                MODEL_PATH = '/workdir/user_input/model_test'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models(exp_group):
    model_path = get_model_path("/my/super/path", exp_group)
    model = pickle.load(open(model_path, 'rb'))
    return model


""" Выгрузка признаков """

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features() -> pd.DataFrame:
    query = 'SELECT * FROM "aleksa-mihajlov_features_lesson_22"'
    return batch_load_sql(query)

post_data = pd.read_sql(
    """SELECT * FROM public.post_text_df;
    """,     engine)


'''создаем get_exp_group(user_id: int) -> str, которая будет определять, в какую группу попал пользователь.'''

SALT = 'mih'
CONTROL_PERCENTAGE = 50

def get_exp_group(user_id: int) -> str:
    hash_value = hashlib.md5((str(user_id)+SALT).encode()).hexdigest()
    group = 'control' if int(hash_value, 16) % 100 < CONTROL_PERCENTAGE else 'test'
    return group


"""Отбор призников по запросу"""

df = load_features()

def get_rec_feed_control(id: int, time:datetime, model, limit: int= 5):
    pred_df = df[(df['user_id'] == id)    ]
    df_dop = pred_df.copy()
    pred_df = pred_df.drop(['user_id', 'post_id', 'index'], axis=1)
    pred_proba = model.predict_proba(pred_df)
    df_dop['pred_proba'] = pred_proba[:, 1]
    df_dop = df_dop.sort_values(by='pred_proba', ascending=False).head(limit)
    list_posts = df_dop['post_id'].to_list()

    return [
        PostGet(**{
            'id': i,
            'text' : post_data[post_data.post_id == i].text.values[0],
            'topic': post_data[post_data.post_id == i].topic.values[0]
        })  for i in list_posts
    ]


def get_rec_feed_test(id: int, time:datetime, model, limit: int= 5):
    pred_df = df[(df['user_id'] == id)    ]
    df_dop = pred_df.copy()
    pred_df = pred_df.drop(['user_id', 'post_id', 'index'], axis=1)
    pred_proba = model.predict_proba(pred_df)
    df_dop['pred_proba'] = pred_proba[:, 1]
    df_dop = df_dop.sort_values(by='pred_proba', ascending=False).head(limit)
    list_posts = df_dop['post_id'].to_list()

    return [
        PostGet(**{
            'id': i,
            'text' : post_data[post_data.post_id == i].text.values[0],
            'topic': post_data[post_data.post_id == i].topic.values[0]
        })  for i in list_posts
    ]


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 5) -> Response:
    exp_group = get_exp_group(id)
    logger.info(exp_group)
    model = load_models(exp_group)
    if exp_group == 'control':
        return Response(exp_group=exp_group, recommendations=get_rec_feed_control(id, time,  model, limit))
    elif exp_group == "test":
        return Response(exp_group=exp_group, recommendations=get_rec_feed_test(id, time,  model, limit))
    else:
        raise ValueError("Unknown group")

