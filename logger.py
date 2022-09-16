import pandas as pd
import os
from datetime import datetime

def error_logger(Class="N/A",Phase="N/A",Filename="N/A",ColumnIndex="N/A",Message="N/A"):
    if not os.path.exists("log"):
        os.mkdir("log")
    now=datetime.now()
    dt=now.strftime("%Y/%m/%d")
    tm=now.strftime("%H:%M:%S")
    row={'Date':[dt],
         'Time':[tm],
         'Class':[Class],
         'Phase':[Phase],
         'Filename':[Filename],
         'Column Index':[ColumnIndex],
         'Message':[Message]}
    df=pd.DataFrame(row)
    df.to_csv('log/ErrorLog_{}.csv'.format(now.strftime("%Y_%m_%d_%H_%M")), mode='a', index=False, header=False)
