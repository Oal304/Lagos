# config.py
SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://root@localhost/traffic_data'
POSTGRESQL_URI = 'postgresql+psycopg2://postgres:password@localhost:5432/postgres'
TIMEZONE = 'Africa/Lagos'
LOOK_BACK = 12
HORIZONS = [1, 2, 3, 4, 8, 12]
