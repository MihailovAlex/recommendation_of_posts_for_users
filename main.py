import pandas as pd

from database import engine

'''
df = pd.read_sql(
    """SELECT t1.* FROM public.feed_data AS t1 
    JOIN public.feed_data AS t2 ON t1.user_id=t2.user_id AND t2.timestamp <=t1.timestamp
    GROUP BY t1.post_id 
    HAVING COUNT(*) <=100;        
    """,
    engine
)
'''

df = pd.read_sql(
    """SELECT * FROM public.feed_data WHERE public.feed_data.timestamp<='2021-12-31 00:00:00' ORDER BY RANDOM() LIMIT 20000000;
;        
    """,
    engine
)

# df = df.to_csv('feeds_2.csv', index=False)
