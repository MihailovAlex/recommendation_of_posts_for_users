from typing import List

from fastapi import FastAPI, Depends
from database import SessionLocal, engine

from sqlalchemy.orm import Session
from schema import UserGet, PostGet, FeedGet
from fastapi import HTTPException

from table_user import User
from table_post import Post
from table_feed import Feed

from sqlalchemy import func

import pickle

import pandas as pd
from datetime import datetime

app = FastAPI()


def get_db():
    with SessionLocal() as db:
        return db


@app.get("/user/{id}", response_model=UserGet)
def get_user(id: int, db: Session = Depends(get_db)):
    result = db.query(User).filter(User.id == id).first()
    if result is not None:
        return result
    else:
        raise HTTPException(status_code=404, detail="user not found")


@app.get("/post/{id}", response_model=PostGet)
def get_post(id: int, db: Session = Depends(get_db)):
    result = db.query(Post).filter(Post.id == id).first()
    if result is not None:
        return result
    else:
        raise HTTPException(status_code=404, detail="user not found")


@app.get("/user/{id}/feed", response_model=List[FeedGet])
def get_feed_user(id: int, db: Session = Depends(get_db), limit: int = 10):
    result = (db.query(Feed)
              .filter(Feed.user_id == id)
              .order_by(Feed.time.desc())
              .limit(limit)
              .all())
    if result is not None:
        return result
    else:
        raise HTTPException(status_code=200, detail="")


@app.get("/post/{id}/feed", response_model=List[FeedGet])
def get_feed_post(id: int, db: Session = Depends(get_db), limit: int = 10):
    result = (db.query(Feed)
              .filter(Feed.post_id == id)
              .order_by(Feed.time.desc())
              .limit(limit)
              .all())
    if result is not None:
        return result
    else:
        raise HTTPException(status_code=200, detail="")


@app.get("/post/recommendation/", response_model=List[PostGet])
def get_recommended_post(
        # id: int,
        limit: int = 10,
        db: Session = Depends(get_db)):
    result = (db.query(Post)
              .select_from(Feed)
              .join(Post)
              .filter(Feed.action == "like")
              .group_by(Post.id)
              .order_by(func.count(Post.id).desc())
              .limit(limit)
              .all())
    return result


""" Загрузка модели """


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("/my/super/path")
    model_load = pickle.load(open(model_path, 'rb'))
    return model_load


model = load_models()

""" Выгрузка признаков """


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        yield from chunk_dataframe.itertuples(index=False)
    conn.close()


def load_features() -> pd.DataFrame:
    query = 'SELECT * FROM "USER_NAME"'
    return pd.DataFrame(list(batch_load_sql(query)))


df = load_features()

post_data = pd.read_sql(
    """SELECT * FROM public.post_text_df;
    """, engine)

"""Отбор призников по запросу"""


def get_rec_feed(id: int, time: datetime, limit: int = 5):
    pred_df = df[(df['user_id'] == id)]
    df_dop = pred_df.copy()
    pred_df = pred_df.drop(['user_id', 'post_id', 'index'], axis=1)
    print(df.columns)
    pred_proba = model.predict_proba(pred_df)
    df_dop['pred_proba'] = pred_proba[:, 1]
    df_dop = df_dop.sort_values(by='pred_proba', ascending=False).head(limit)
    list_posts = df_dop['post_id'].to_list()

    return [
        PostGet(**{
            'id': i,
            'text': post_data[post_data.post_id == i].text.values[0],
            'topic': post_data[post_data.post_id == i].topic.values[0]
        }) for i in list_posts
    ]


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 5) -> List[PostGet]:
    return get_rec_feed(id, time, limit)

# 2021/10/29 20:56:58
# print(recommended_posts(204, datetime(2021, 10, 29, 20, 56, 58)))
