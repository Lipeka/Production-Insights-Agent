
from sqlalchemy import create_engine

def get_engine():
    return create_engine("postgresql://postgres:root@localhost:5432/prod_insights")