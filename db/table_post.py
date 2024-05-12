from database import Base, SessionLocal
from sqlalchemy import Column, Integer, String


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