import pandas as pd

#mode == 0: keep year only
#mode == 1: keep year and month only
#mode == 2: keep year and month and day
def reservation_information(filename: str, new_filename: str, mode: int) -> None:
    data = pd.read_csv(filename)
    day = [i[:2] for i in data['reservation_status_date']]
    month = [i[3:5] for i in data['reservation_status_date']]
    year = [i[-4:] for i in data['reservation_status_date']]
    if mode == 0:
        data['reservation_status_year'] = year
    elif mode == 1:
        data['reservation_status_year'] = year
        data['reservation_status_month'] = month
    elif mode ==2:
        data['reservation_status_year'] = year
        data['reservation_status_month'] = month
        data['reservation_status_day'] = day
    
    data = data.drop('reservation_status_date', axis=1)
    data.to_csv(new_filename, index = False) 
