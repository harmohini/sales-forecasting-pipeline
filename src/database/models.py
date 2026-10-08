"""
SQLAlchemy Database Models.

Defines schemas for sales records, predictions, and model metadata tables.
"""

from sqlalchemy import Column, Integer, String, Float, Date, DateTime, UniqueConstraint
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class SalesTrainRaw(Base):
    """
    Raw sales training data from Rossmann train.csv.
    Natural Business Key: (store, date)
    """
    __tablename__ = "sales_train_raw"

    store = Column(Integer, primary_key=True, nullable=False)
    date = Column(Date, primary_key=True, nullable=False)
    day_of_week = Column(Integer, nullable=False)
    sales = Column(Integer, nullable=False)
    customers = Column(Integer, nullable=False)
    open = Column(Integer, nullable=False)
    promo = Column(Integer, nullable=False)
    state_holiday = Column(String(10), nullable=False)
    school_holiday = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class StoresRaw(Base):
    """
    Raw store metadata from Rossmann store.csv.
    Primary Key: store
    """
    __tablename__ = "stores_raw"

    store = Column(Integer, primary_key=True, nullable=False)
    store_type = Column(String(10), nullable=False)
    assortment = Column(String(10), nullable=False)
    competition_distance = Column(Float, nullable=True)
    competition_open_since_month = Column(Integer, nullable=True)
    competition_open_since_year = Column(Integer, nullable=True)
    promo2 = Column(Integer, nullable=False)
    promo2_since_week = Column(Integer, nullable=True)
    promo2_since_year = Column(Integer, nullable=True)
    promo_interval = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SalesTestRaw(Base):
    """
    Raw sales test data from Rossmann test.csv.
    Primary Key: id
    Natural Key: (store, date)
    """
    __tablename__ = "sales_test_raw"
    __table_args__ = (
        UniqueConstraint("store", "date", name="uq_sales_test_store_date"),
    )

    id = Column(Integer, primary_key=True, nullable=False)
    store = Column(Integer, nullable=False)
    day_of_week = Column(Integer, nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Integer, nullable=True)
    promo = Column(Integer, nullable=False)
    state_holiday = Column(String(10), nullable=False)
    school_holiday = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def create_tables(engine):
    """
    Create database tables if they do not already exist.
    """
    Base.metadata.create_all(bind=engine)


def init_db(engine):
    """
    Initialize database tables defined in SQLAlchemy Base metadata.
    """
    create_tables(engine)

