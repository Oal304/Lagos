def create_sqlalchemy_engine():
    from sqlalchemy import create_engine
    from config import SQLALCHEMY_DATABASE_URI
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    return engine

def fetch_data_from_mysql(query, engine):
    import pandas as pd
    try:
        df = pd.read_sql(query, con=engine)
        return df
    except Exception as e:
        print(f"Error reading data from MySQL: {e}")
        return None
