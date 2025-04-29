import sqlite3
import pandas as pd

def query_climate_database(db_file, country, year_begin, year_end, month):
    """
    Retrieves temperature readings from the climate database for a specified country and date range.

    Parameters:
    - db_file (str): Path to the SQLite database file.
    - country (str): Country name to filter the temperature readings.
    - year_begin (int): Starting year for the range of readings.
    - year_end (int): Ending year for the range of readings.
    - month (int): Month of the year for which the readings are queried.

    Returns:
    - DataFrame: A pandas DataFrame containing temperature readings, with columns for station 
      name, latitude, longitude, country, year, month, and average temperature.
    """
    with sqlite3.connect(db_file) as conn:
        # conn is automatically closed when this block ends

        # NAME, LATITUDE, LONGITUDE, Country, Year, Month, Temp
        cmd = \
        f"""
        SELECT S.NAME, S.LATITUDE, S.LONGITUDE, C.Name as Country, T.Year, T.Month, T.Temp 
        FROM temperatures T
        LEFT JOIN stations S ON T.ID = S.ID
        LEFT JOIN countries C ON SUBSTR(T.ID, 1, 2) = C.`FIPS 10-4`
        WHERE T.Month = {month} 
            AND T.Year >= {year_begin} 
            AND T.Year <= {year_end}
            AND C.Name = '{country}'
        """
        df = pd.read_sql_query(cmd, conn)
        df = df.drop_duplicates()
    return df

def seasons_database(db_file, country, year_begin, year_end):
    """
    Retrieves temperature readings classified by hemisphere and seasons for a specified country and date range.

    Parameters:
    - db_file (str): Path to the SQLite database file.
    - country (str): Country name to filter the temperature readings.
    - year_begin (int): Starting year for the range of readings.
    - year_end (int): Ending year for the range of readings.

    Returns:
    - DataFrame: A pandas DataFrame containing the temperature readings along with additional columns 
      'Hemispheres' and 'Seasons' indicating the hemisphere (North or South) and the meteorological 
      season when the reading was taken.
    """

    with sqlite3.connect(db_file) as conn:
        cmd = \
        f"""
        SELECT S.NAME, S.LATITUDE, S.LONGITUDE, C.NAME as Country, T.Year, T.Month, T.Temp, 
        CASE 
            WHEN S.LATITUDE > 0 THEN 'North'
            ELSE 'South'
        END AS Hemispheres, 
        CASE 
            WHEN (S.LATITUDE > 0 AND (T.Month = 12 OR T.Month = 1 OR T.Month = 2)) THEN 'Winter'
            WHEN (S.LATITUDE <= 0 AND T.Month >= 6 AND T.Month <= 8) THEN 'Winter'
            WHEN (S.LATITUDE > 0 AND T.Month >= 3 AND T.Month <= 5) THEN 'Spring'
            WHEN (S.LATITUDE <= 0 AND T.Month >= 9 AND T.Month <= 11) THEN 'Spring'
            WHEN (S.LATITUDE > 0 AND T.Month >= 6 AND T.Month <= 8) THEN 'Summer'
            WHEN (S.LATITUDE <= 0 AND (T.Month = 12 OR T.Month = 1 OR T.Month = 2)) THEN 'Summer'
            ELSE 'Fall'
        END AS Seasons 
        FROM temperatures T 
        LEFT JOIN stations S ON T.ID = S.ID 
        LEFT JOIN countries C ON SUBSTR(T.ID, 1, 2) = C.`FIPS 10-4`
        WHERE T.Year >= {year_begin} 
            AND T.Year <= {year_end}
            AND C.NAME = '{country}';
        """
        df = pd.read_sql_query(cmd, conn)
        df = df.drop_duplicates()
    return df 
