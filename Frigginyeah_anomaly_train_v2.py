# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:20:10 2022

@author: perlzuser
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 04:52:26 2022

@author: perlzuser
"""
#Importing Libraries
import os
import ftplib
import pandas as pd
import numpy as np
#from IPython.display import display
import datetime
from datetime import date
import glob
from glob import glob
from time import time
import json
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
#Libs for error logging
import traceback
import email_sender as em
import logger as log
# Models
from pyod.models.copod import COPOD
import pickle
from datetime import datetime

#--------------------------------------------------------FTP Download-----------------------------------------------------------
Start_time = time()
def FTP_download():
    HOST=jsonData["HOST"]
    USER=jsonData["USERNAME"]
    PASSWORD=jsonData["PSWD"]
    dirName = jsonData["dirName_train"]
    ftp = ftplib.FTP(HOST,USER,PASSWORD)
    ftp.encoding = "utf-8"
    ftp.cwd(dirName)
    print(dirName)
    filematch = jsonData["filematch"]
    target_dir = jsonData["Train_file_path"]
    for filename in ftp.nlst(filematch):
        target_file_name = os.path.join(target_dir, os.path.basename(filename))
        print(filename,"Downloading....")
        with open(target_file_name ,'wb') as fhandle:
            try:
                ftp.retrbinary('RETR %s' % filename, fhandle.write) 
                
            except ftplib.all_errors as error:
                print("Error occured while Downloading in FTP")
                em.send_mail('CRITICAL',"FTP Download",'','','{} error occured while downloading the train input'.format(traceback.format_exc()))
                log.error_logger('CRITICAL',"FTP Download",'','','{} error occured while downloading the train input'.format(traceback.format_exc()))
            
    if ftp != None:
        ftp.close()
#----------------------------------------------------------genFilename--------------------------------------------------------- 
 
def genFilename(data):
    data=re.sub('\W','',data)
    return data

#----------------------------------------------------------Data Acceptence---------------------------------------------------------
def Data_Acceptance(data):
    for column in ['harvestDate', 'siteid','site_sku','combinedinput','product_online_price']:
        #listElem=[org_col.split('\t') for org_col in data.columns][0]
        #input(listElem)
        if not column in data.columns:
            em.send_mail('CRITICAL',"Data Acceptance",files,'','{} was missing or name mismatch'.format(column))
            log.error_logger('CRITICAL',"Data Acceptance",files,'','{} was missing or name mismatch'.format(column))
            return False
        if data[column].isnull().all():
            em.send_mail('CRITICAL',"Data Acceptance",files,'','{} has no data'.format(column))
            log.error_logger('CRITICAL',"Data Acceptance",files,'','{} has no data'.format(column))
            return False
        if column=='siteid' and data[column].isnull().any():
            em.send_mail('CRITICAL',"Data Acceptance",files,'','{} must not be null'.format(column))
            log.error_logger('CRITICAL',"Data Acceptance",files,'','{} must not be null'.format(column))
            return False
    return True

#---------------------------------------------------------- Data Preprocessing------------------------------------------------------
def Data_Preprocessing(t_data):
    global missing_values_df,special_char_df,zero_price_df,Anomaly_Preprocess,combinedinput_lessthan5
    print("Before Pre-processing Siteid value_counts"'\n')
    print("in pre-processing")
    t_data['product_online_price']= t_data['product_online_price'].replace("[\$,]",'', regex=True)
    t_data['combinedinput'] = t_data['combinedinput'].astype(str)
    t_data['combinedinput'] = t_data['combinedinput'].str.lower()   
    missing_values_df = t_data[(t_data['product_online_price'].isna()) | (t_data['combinedinput'].isna())]
    missing_values_df[['product_online_price','combinedinput']] = missing_values_df[['product_online_price','combinedinput']].fillna("Null")    
    t_data.dropna(subset = ['product_online_price','combinedinput'],inplace=True)    
    special_char_df = t_data[t_data['product_online_price'].str.contains('[-,@_!#$%^&*()<>?/|}{~:A-Za-z]',regex=True)]
    special_char_df['Anomaly_tag'] = "spec_char"    
    zero_price_df = t_data[t_data['product_online_price']==0]
    Anomaly_Preprocess = pd.concat([missing_values_df,zero_price_df],axis=0)
    Anomaly_Preprocess['Anomaly_tag'] =1
    if special_char_df.shape[0] |  zero_price_df.shape[0] >0:
        t_data = t_data.drop(t_data.index.isin(special_char_df.index,zero_price_df,index))
    t_data['product_online_price'] = t_data['product_online_price'].astype("float")
    
    combinedinput_lessthan5 = t_data.groupby("combinedinput").filter(lambda x: len(x) <5)
    combinedinput_lessthan5['Anomaly_tag'] = "<5"
    train_data = t_data.groupby("combinedinput").filter(lambda x: len(x) >=5)
    print("--------------------------------")
    print("Before Pre-processing Siteid value_counts"'\n',train_data.siteid.value_counts())
    print("")
    print("missing_values_df",missing_values_df.shape)
    print("")
    print("special_char_df",special_char_df.shape)
    print("")
    print("Anomaly_output_1",Anomaly_Preprocess.shape)
    print("")
    print("zero_price_df",zero_price_df.shape)
    print("")
    print("combinedinput_lessthan5",combinedinput_lessthan5.shape)
    print("")
    print("Unique combinedinput after pre-processing",train_data.combinedinput.nunique())
    print("")
    print("exit pre-processing")
    return train_data

#-------------------------------------------------------------------Find anomalies-------------------------------------------------------
def find_anomalies(value, lower_threshold, upper_threshold):
    
    if value < lower_threshold or value > upper_threshold:
        return 1
    else: return 0
    
#-------------------------------------------------------------------Fit Model (1)------------------------------------------------------
def Fit_Model(train_data):
    #start_time = time.
    anomaly_combinedinput = []
    anomaly_combinedinput = []
    Output_data = pd.DataFrame(columns=['harvestDate', 'siteid', 'inputid','site_sku','seller_name','combinedinput','product_online_price','Anomaly_tag'])
    outliers_count = []
    for count,combinedinput in enumerate(train_data['combinedinput'].unique()):  
        outliers_count = []
        Model_train_data = train_data[train_data['combinedinput']==combinedinput]
        min_val = Model_train_data['product_online_price'].quantile(.10)
        max_val = Model_train_data['product_online_price'].quantile(.90)
        for price in Model_train_data['product_online_price']:              
            if (price < min_val-(min_val*0.30)) or (price > max_val+(max_val*0.30)):
                outliers_count.append(price)
        
        outlier_fraction = len(outliers_count)/len(Model_train_data)
        
        if outlier_fraction >0:
                
                Sc = StandardScaler() 
                Model_train_data['product_online_price_1'] = Sc.fit_transform(Model_train_data[['product_online_price']])   
                anomaly_combinedinput.append(combinedinput)
                anomaly_combinedinput.append(len(outliers_count))
                COPOD_Model = COPOD(contamination=outlier_fraction).fit(Model_train_data[['product_online_price_1']])
                Model_train_data['Anomaly_tag'] =  COPOD_Model.predict(Model_train_data[['product_online_price_1']])
                pickle.dump(COPOD_Model, open('{}{}.pkl'.format(jsonData['Pickle_files'],genFilename(combinedinput)), 'wb'))
                Model_train_data = Model_train_data[Model_train_data['Anomaly_tag']==1]
                Output_data = Output_data.append(Model_train_data)
                print(count,combinedinput)
                
    Output_data.drop('product_online_price_1',axis=1,inplace=True) 
    Output_data['product_online_price'] = "$"+Output_data['product_online_price'].astype(str)            

    with open("trained_combinedinput_config.json","w",encoding="utf-8") as tcf:
        tcf.write(json.dumps(list(t_data['combinedinput'].unique()),indent=4))
        
    Train_data_not_included_observations = pd.concat([missing_values_df,special_char_df,zero_price_df,combinedinput_lessthan5],axis=0)
    Train_data_not_included_observations.to_csv('{}Train_data_not_included_observations_{}.txt'.format(jsonData['Train data summary'],datetime.now().date()),header=True, index=None, sep=' ', mode='a')                                          
                                              
    Train_data_Anomaly_Output = pd.concat([Output_data,Anomaly_Preprocess],axis=0)
    Site_id_name = {1:'walmart_1',4:'amazon_4',22:'homedepot_22',34:'officedepot_34',37:'overstock_37',46:'staples_46',52:'wayfair_52'}
    for site_id in Train_data_Anomaly_Output.keys():
        site_id_df = Train_data_Anomaly_Output[Train_data_Anomaly_Output['siteid']==site_id] 
        site_id_df['combinedinput']=site_id_df['combinedinput'].str.upper()
        out_filename="{}{}_{}.txt".format(jsonData['Train data Anomaly summary'],Site_id_name.get(site_id),datetime.now().date())            
        print(out_filename)            
        site_id_df.to_csv(out_filename,header=True, index=None,sep='\t')
    
    Train_data_summary_report = pd.DataFrame({'siteid':Site_id_name.get(site_id),'Null values':missing_values_df['combinedinput'].nunique(),'Special char':special_char_df['combinedinput'].nunique(),'zero in price':zero_price_df['combinedinput'].nunique(),'less_than5':combinedinput_lessthan5['combinedinput'].nunique(),'Anomalies':Train_data_Anomaly_Output['combinedinput'].nunique(),'Normal':(train_data['combinedinput'].nunique()- Train_data_Anomaly_Output['combinedinput'].nunique())},index=range(0,1))
    
    Train_data_summary_report_rows = pd.DataFrame({'siteid':Site_id_name.get(site_id),'Null values':len(missing_values_df),'Special char':len(special_char_df),'zero in price':len(zero_price_df),'less_than5':len(combinedinput_lessthan5),'Anomalies':len(Train_data_Anomaly_Output),'Normal':(len(train_data)- len(Train_data_Anomaly_Output))},index=range(0,1))
    
    Train_data_summary_report.to_csv('{}Train_data_summary_report_COPOD_{}.txt'.format(jsonData['Train data summary'],datetime.now().date()),header=True, index=None, sep=' ',mode='a')
    Train_data_summary_report_rows.to_csv('{}Train_data_summary_report_rows_COPOD_{}.txt'.format(jsonData['Train data summary'],datetime.now().date()),header=True, index=None, sep=' ',mode='a')
    print("exit fit model")  


#-------------------------------------------------------------------Fit Model (2)------------------------------------------------------    
def Fit_Model_Median(train_data):
    Output_data = pd.DataFrame(columns=['harvestDate', 'siteid', 'inputid','site_sku','seller_name','combinedinput','product_online_price','Anomaly_tag'])
    train_data_properties = pd.DataFrame()
    for count,combinedinput in enumerate(train_data['combinedinput'].unique()):          
        Model_train_data = train_data[train_data['combinedinput']==combinedinput]
        median = Model_train_data['product_online_price'].median()
        lower_threshold = median- (median*.50)
        upper_threshold = median+ (median*.50)        
        Model_train_data['Anomaly_tag'] = Model_train_data['product_online_price'].apply(find_anomalies, args=(lower_threshold, upper_threshold))
        Model_train_data = Model_train_data[Model_train_data['Anomaly_tag']==1]
        Output_data = Output_data.append(Model_train_data)
        temp_df= pd.DataFrame({'combinedinput':combinedinput,'lower':lower_threshold,'upper':upper_threshold,'median':median},index=range(0,1))
        train_data_properties = train_data_properties.append(temp_df,ignore_index=True)
        print(count,combinedinput)
     
    train_data_properties.to_csv('{}Train_data_lower_upper_threshold{}.txt'.format(jsonData['Train data Properties'],datetime.now().date()),header=True, index=None, sep=' ',mode='a')  
    Train_data_not_included_observations = pd.concat([missing_values_df,special_char_df,zero_price_df,combinedinput_lessthan5],axis=0)
    Train_data_not_included_observations.to_csv('{}Train_data_not_included_observations_{}.txt'.format(jsonData['Train data summary'],datetime.now().date()),header=True, index=None, sep=' ',mode='a')
    
    Output_data['product_online_price'] = "$"+Output_data['product_online_price'].astype(str)        
    Train_data_Anomaly_Output = pd.concat([Output_data,Anomaly_Preprocess],axis=0)
    
    Site_id_name = {1:'walmart_1',4:'amazon_4',22:'homedepot_22',34:'officedepot_34',37:'overstock_37',46:'staples_46',52:'wayfair_52'}
    for site_id in train_data['siteid'].unique():
        site_id_df = Train_data_Anomaly_Output[Train_data_Anomaly_Output['siteid']==site_id] 
        site_id_df['combinedinput']=site_id_df['combinedinput'].str.upper()
        out_filename="{}{}_{}.txt".format(jsonData['Train data Anomaly summary'],Site_id_name.get(site_id),datetime.now().date())            
        print(out_filename)            
        site_id_df.to_csv(out_filename,header=True, index=None,sep='\t')
                                          
    Train_data_summary_report = pd.DataFrame({'siteid':Site_id_name.get(site_id),'Null values':missing_values_df['combinedinput'].nunique(),'Special char':special_char_df['combinedinput'].nunique(),'zero in price':zero_price_df['combinedinput'].nunique(),'less_than5':combinedinput_lessthan5['combinedinput'].nunique(),'Anomalies':Train_data_Anomaly_Output['combinedinput'].nunique(),'Normal':(train_data['combinedinput'].nunique()- Train_data_Anomaly_Output['combinedinput'].nunique())},index=range(0,1))
    
    Train_data_summary_report_rows = pd.DataFrame({'siteid':Site_id_name.get(site_id),'Null values':len(missing_values_df),'Special char':len(special_char_df),'zero in price':len(zero_price_df),'less_than5':len(combinedinput_lessthan5),'Anomalies':len(Train_data_Anomaly_Output),'Normal':(len(train_data)- len(Train_data_Anomaly_Output))},index=range(0,1))
     
    Train_data_summary_report.to_csv('{}Train_data_summary_report_RuleBased_{}.txt'.format(jsonData['Train data summary'],datetime.now().date()),header=True, index=None, sep=' ',mode='a')
    Train_data_summary_report_rows.to_csv('{}Train_data_summary_report__rows_RuleBased_{}.txt'.format(jsonData['Train data summary'],datetime.now().date()),header=True, index=None, sep=' ',mode='a')
    print("exit fit model")
 
#-------------------------------------------------------------------Fit Model (3)------------------------------------------------------ 
def Fit_Model_IQR(train_data):

    Output_data = pd.DataFrame(columns=['harvestDate', 'siteid', 'inputid','site_sku','seller_name','combinedinput','product_online_price','Anomaly_tag'])
    train_data_properties = pd.DataFrame()
    outliers_count = []
    for count,combinedinput in enumerate(train_data['combinedinput'].unique()):  
        outliers_count = []
        Model_train_data = train_data[train_data['combinedinput']==combinedinput]
        
        quartiles = dict(Model_train_data['product_online_price'].quantile([.25,.75]))
        max_val, min_val = quartiles[0.75], quartiles[0.25]
        lower_threshold = min_val- (min_val*.30)
        upper_threshold = max_val+ (max_val*.30)
        temp_df= pd.DataFrame({'combinedinput':combinedinput,'lower':lower_threshold,'upper':upper_threshold},index=range(0,1))
        train_data_properties = train_data_properties.append(temp_df,ignore_index=True)
        Model_train_data['Anomaly_tag'] = Model_train_data['product_online_price'].apply(find_anomalies, args=(lower_threshold, upper_threshold))
        Model_train_data = Model_train_data[Model_train_data['Anomaly_tag']==1]
        Output_data = Output_data.append(Model_train_data)
        print(count,combinedinput)
        
    train_data_properties.to_csv('{}Train_data_lower_upper_threshold_IQR{}.txt'.format(jsonData['Train data Properties'],datetime.now().date()),header=True, index=None, sep=' ')    
                
                
    Train_data_not_included_observations = pd.concat([missing_values_df,special_char_df,zero_price_df,combinedinput_lessthan5],axis=0)
    #Train_data_not_included_observations.to_csv('{}Train_data_not_included_observations_{}.txt'.format(jsonData['Train data summary'],datetime.now().date()),header=True, index=None, sep=' ', mode='a')
    
    Output_data['product_online_price'] = "$"+Output_data['product_online_price'].astype(str)        
    Train_data_Anomaly_Output = pd.concat([Output_data,Anomaly_Preprocess],axis=0)
    Site_id_name = {1:'walmart_1',4:'amazon_4',22:'homedepot_22',34:'officedepot_34',37:'overstock_37',46:'staples_46',52:'wayfair_52'}
    for site_id in Site_id_name.keys():
        site_id_df = Train_data_Anomaly_Output[Train_data_Anomaly_Output['siteid']==site_id] 
        site_id_df['combinedinput']=site_id_df['combinedinput'].str.upper()
        out_filename="{}{}_{}.txt".format(jsonData['Train data Anomaly summary'],Site_id_name.get(site_id),datetime.now().date())            
        print(out_filename)            
        site_id_df.to_csv(out_filename,header=True, index=None,sep='\t')
                                          
    Train_data_summary_report = pd.DataFrame({'Null values':missing_values_df['combinedinput'].nunique(),'Special char':special_char_df['combinedinput'].nunique(),'zero in price':zero_price_df['combinedinput'].nunique(),'less_than5':combinedinput_lessthan5['combinedinput'].nunique(),'Anomalies':Train_data_Anomaly_Output['combinedinput'].nunique(),'Normal':(train_data['combinedinput'].nunique()- Train_data_Anomaly_Output['combinedinput'].nunique())},index=range(0,1))
    
    Train_data_summary_report_rows = pd.DataFrame({'Null values':len(missing_values_df),'Special char':len(special_char_df),'zero in price':len(zero_price_df),'less_than5':len(combinedinput_lessthan5),'Anomalies':len(Train_data_Anomaly_Output),'Normal':(len(train_data)- len(Train_data_Anomaly_Output))},index=range(0,1))
     
    Train_data_summary_report.to_csv('{}Train_data_summary_report_IQR_{}.txt'.format(jsonData['Train data summary'],datetime.now().date()),header=True, index=None, sep=' ')
    Train_data_summary_report_rows.to_csv('{}Train_data_summary_report_IQR_{}.txt'.format(jsonData['Train data summary'],datetime.now().date()),header=True, index=None, sep=' ')
    print("exit fit model")
#-------------------------------------------------------------------Main function------------------------------------------------------

if __name__ == "__main__":
    try:       
        # Importing the data       
        jsonData=json.load(open("Frigginyeah_model_config.json","r",encoding="utf-8"))
        #FTP_download()
        read_files = glob("{}*.txt".format(jsonData['Train_file_path']))
        df = pd.DataFrame()
        for files in read_files:   
            print(files)
            try:
                temp_data = pd.read_csv(files, delimiter="\t",) 
            except UnicodeDecodeError:
                input("Here")
                em.send_mail('CRITICAL',"Data Acceptence",files,'','Files not in UTF-8 Encoding')
                log.error_logger('CRITICAL',"Data Acceptence",files,'','Files not in UTF-8 Encoding')
                input("check")
                quit()
            df=df.append(temp_data)
        print("Unique siteid raw data",df.siteid.unique())
        print("Unique combinedinput raw data",df.combinedinput.nunique())
        print(df.shape)       
        print(df.isna().sum())   
        print(df.siteid.value_counts())
        #add loop for sideid wise run
        isAccept = Data_Acceptance(df)
        if isAccept:
            df = df.reset_index(drop=True)
            df.rename(columns={'index':'previous_index'},inplace=True) #check code!!!
            t_data = df.copy()
            train_data=Data_Preprocessing(t_data)
            #Fit_Model(train_data)
            #Fit_Model_Median(train_data)
            #Fit_Model_IQR(train_data)
 
    except:
        print(traceback.format_exc()) #To capture errors
        em.send_mail('CRITICAL',"",'','','{} Error while running the main function'.format(traceback.format_exc()))
        log.error_logger('CRITICAL',"",'','','{} Error while running the main function'.format(traceback.format_exc()))
      
print("Total time (in seconds)", round(time()-Start_time))
