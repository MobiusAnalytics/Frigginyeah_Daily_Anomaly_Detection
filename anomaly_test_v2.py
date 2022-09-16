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
import time
import json
import warnings
warnings.filterwarnings("ignore")
#Libs for error logging
import traceback
import email_sender as em
import logger as log
import shutil
import re
# Models
from sklearn.preprocessing import StandardScaler
from pyod.models.copod import COPOD
import pickle
from datetime import datetime

#--------------------------------------------------------FTP Download-----------------------------------------------------------
def FTP_download():
    HOST=jsonData["HOST"]
    USER=jsonData["USERNAME"]
    PASSWORD=jsonData["PSWD"]
    dirName = jsonData["dirName_test"]
    ftp = ftplib.FTP(HOST,USER,PASSWORD)
    ftp.cwd(dirName)
    filematch = jsonData["filematch"]
    target_dir = jsonData["target_dir_test"]
    for filename in ftp.nlst(filematch):
        target_file_name = os.path.join(target_dir, os.path.basename(filename))
        print(filename,"Downloading....")
        with open(target_file_name ,'wb') as fhandle:
            try:
                ftp.retrbinary('RETR %s' % filename, fhandle.write)               
            except ftplib.all_errors as error:
                print("Error occured while Downloading in FTP")
                em.send_mail('CRITICAL',"FTP Download",'','','{} error occured while downloading the test input'.format(traceback.format_exc()))
                log.error_logger('CRITICAL',"FTP Download",'','','{} error occured while downloading the test input'.format(traceback.format_exc()))
    if ftp != None:
        ftp.close()

#----------------------------------------------------------genFilename---------------------------------------------------------   
def genFilename(data):
    data=re.sub('\W','',data)
    return data
#----------------------------------------------------------Data Acceptence---------------------------------------------------------

def Data_Acceptance(data):
    for column in ['harvestDate', 'siteid','site_sku','combinedinput','product_online_price']:
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
    
def Data_Preprocessing(tt_data):
    global missing_values_df,special_char_df,zero_price_df,Anomaly_Preprocess
    print("in pre-processing")
    tt_data['product_online_price']= tt_data['product_online_price'].replace("[\$,]",'', regex=True)
    print(tt_data['combinedinput'].nunique())
    tt_data['combinedinput'] = tt_data['combinedinput'].astype(str)
    tt_data['combinedinput'] = tt_data['combinedinput'].str.lower()   
    missing_values_df = tt_data[(tt_data['product_online_price'].isna()) | (tt_data['combinedinput'].isna())]
    missing_values_df[['product_online_price','combinedinput']] = missing_values_df[['product_online_price','combinedinput']].fillna("Null")    
    tt_data.dropna(subset=['product_online_price','combinedinput'],inplace=True)       
    special_char_df = tt_data[tt_data['product_online_price'].str.contains('[-,@_!#$%^&*()<>?/|}{~:A-Za-z]',regex=True)]
    special_char_df['Anomaly_tag'] = "spec_char"    
    zero_price_df = tt_data[tt_data['product_online_price']==0]
    Anomaly_Preprocess = pd.concat([missing_values_df,zero_price_df],axis=0)
    Anomaly_Preprocess['Anomaly_tag'] =1
    if special_char_df.shape[0] |  zero_price_df.shape[0] >0:
        tt_data = tt_data.drop(special_char_df.index)
        tt_data = tt_data.drop(zero_price_df.index)
    tt_data['product_online_price'] = tt_data['product_online_price'].astype("float")   
    test_data = tt_data.groupby("combinedinput").filter(lambda x: len(x)>=1)
    print("exit pre-processing")
    return test_data

#-------------------------------------------------------------------Find anomalies-------------------------------------------------------
    
def find_anomalies(value, lower_threshold, upper_threshold):
    
    if value < lower_threshold or value > upper_threshold:
        return 1
    else: return 0
#-------------------------------------------------------------------Test Data Prediction (Model(1))---------------------------------------
     
def Test_Data_Prediction(test_data):
    global sitefilename
    Test_Output_data_a = pd.DataFrame(columns=['harvestDate', 'siteid', 'inputid','site_sku','seller_name','combinedinput','product_online_price','Anomaly_tag'])
    Test_Output_data_b = pd.DataFrame(columns=['harvestDate', 'siteid', 'inputid','site_sku','seller_name','combinedinput','product_online_price','Anomaly_tag'])   
    for combinedinput in test_data['combinedinput'].unique():
        filename="{}/{}.pkl".format(jsonData["Pickle_files"],genFilename(combinedinput))       
        if os.path.exists(filename):  
            print(combinedinput,"Testing with existing pickle files....")
            load_model = pickle.load(open(filename, 'rb'))
            Model_test_data_a = test_data[test_data['combinedinput']==combinedinput]
            Sc = StandardScaler()
            Model_test_data_a['product_online_price_1'] = Sc.fit_transform(Model_test_data_a[['product_online_price']])
            Model_test_data_a['Anomaly_tag']= load_model.predict(Model_test_data_a[['product_online_price_1']])
            Model_test_data_a=Model_test_data_a[Model_test_data_a['Anomaly_tag']==1]
            Test_Output_data_a=Test_Output_data_a.append(Model_test_data_a)
            Test_Output_data_a.drop('product_online_price_1',axis=1,inplace=True)            
        # else:
                # print("Retraining..")
                # Model_test_data_b = test_data[test_data['combinedinput']==combinedinput]
                # min_val = Model_test_data_b['product_online_price'].quantile(.10)
                # max_val = Model_test_data_b['product_online_price'].quantile(.90)
                # outliers_count = []
                # for price in Model_test_data_b['product_online_price']:              
                    # if (price < min_val-(min_val*0.30)) or (price > max_val+(max_val*0.30)):
                        # outliers_count.append(price)
                        
                # outlier_fraction = len(outliers_count)/len(Model_test_data_b)
                # if outlier_fraction >0:
                    # Model_test_data_b['product_online_price_1'] = Sc.fit_transform(Model_test_data_b[['product_online_price']])
                    # COPOD_Model = COPOD(contamination=outlier_fraction)
                    # Model_test_data_b['Anomaly_tag'] =  COPOD_Model.fit_predict(Model_test_data_b[['product_online_price_1']])  
                    # pickle.dump(COPOD_Model, open('{}{}.pkl'.format(jsonData['Pickle_files'],genFilename(combinedinput), 'wb'))           
                    # Model_test_data_b=Model_test_data_b[Model_test_data_b['Anomaly_tag']==1]
                    # Test_Output_data_b=Test_Output_data_b.append(Model_test_data_b)
                    # Test_Output_data_b.drop('product_online_price_1',axis=1,inplace=True)
                    #print(combinedinput)

    Test_Output_data = pd.concat([Test_Output_data_a,Test_Output_data_b],axis=0)
    Test_Output_data['product_online_price'] = "$"+Test_Output_data['product_online_price'].astype(str)
                           
    trained_sku_id=json.load(open("trained_combinedinput_config.json","r",encoding="utf-8"))
    # New_site_sku = pd.DataFrame(columns=['site_sku'])    
    # if len(set(test_data['site_sku']) - set(trained_sku_id)) > 0:
        # New_site_sku['site_sku'] = list(set(test_data['site_sku']) - set(trained_sku_id))               
        # New_site_sku = pd.merge(New_site_sku,test_data[['site_sku','siteid']].drop_duplicates(subset='site_sku'),on='site_sku',how='left') 
        # input(New_site_sku)
        # New_site_sku.to_csv('{}Site_sku_not_found_in_train_dump{}.txt'.format(jsonData['Anomaly_output'],datetime.now().date()),header=True, index=None, sep='\t')
        #FTP_Upload('{}Site_sku_not_found_in_train_dump{}.txt'.format(jsonData['Anomaly_output'],datetime.now().date()))
        
    Test_Output_data_all = pd.concat([Test_Output_data,Anomaly_Preprocess],axis=0)
    Site_id_name = {1:'walmart_1',4:'amazon_4',22:'homedepot_22',34:'officedepot_34',37:'overstock_37',46:'staples_46',52:'wayfair_52'}
    for site_id in Site_id_name.keys():
        site_id_df = Test_Output_data_all[Test_Output_data_all['siteid']==site_id] 
        site_id_df['combinedinput']=site_id_df['combinedinput'].str.upper()
        out_filename="{}{}_{}.txt".format(jsonData['Anomaly_output'],Site_id_name.get(site_id),datetime.now().date())            
        print(out_filename)            
        site_id_df.to_csv(out_filename,header=True, index=None,sep='\t') 
        FTP_Upload(out_filename)
        em.send_mail('SUCCESS',"OK","{} in {}".format(site_id,files),'',"Completed successfully")
        log.error_logger('SUCCESS',"OK","{} in {}".format(site_id,files),'',"Completed successfully")
                                                

    Test_data_summary_report = pd.DataFrame({'Null values':missing_values_df['combinedinput'].nunique(),'Special char':special_char_df['combinedinput'].nunique(),'zero in price':zero_price_df['combinedinput'].nunique(),'Anomalies':Test_Output_data_all['combinedinput'].nunique(),'Normal':(test_data['combinedinput'].nunique()- Test_Output_data_all['combinedinput'].nunique())},index=range(0,1))
    
    Test_data_summary_report_rows = pd.DataFrame({'Null values':len(missing_values_df),'Special char':len(special_char_df),'zero in price':len(zero_price_df),'Anomalies_sku':len(Test_Output_data_all),'Normal_sku':(len(test_data)- len(Test_Output_data_all))},index=range(0,1))
     
    Test_data_summary_report.to_csv('{}Test_data_summary_report_{}.txt'.format(jsonData['Test data summary'],datetime.now().date()),header=True, index=None, sep=' ', mode='a')
    Test_data_summary_report_rows.to_csv('{}Test_data_summary_report_{}.txt'.format(jsonData['Test data summary'],datetime.now().date()),header=True, index=None, sep=' ', mode='a')
#-------------------------------------------------------------------Test Data Prediction (Model(2))---------------------------------------

def Test_data_Median_Pred(test_data):
    Test_Output_data = pd.DataFrame()
    train_data_properties_add = pd.DataFrame()
    prop_files = glob("{}*.csv".format(jsonData['Train data Properties']))
    for prop_file in prop_files:
        train_data_properties = pd.read_csv(prop_file,sep=" ") 
    for column in train_data_properties.columns[1:]:
        train_data_properties[column] = train_data_properties[column].replace("[-,@_!#$%^&*()<>?/|}{~:A-Za-z]",0, regex=True)
        train_data_properties[column] = train_data_properties[column].astype("float")
    for id_count,combinedinput in enumerate(test_data['combinedinput'].unique()):
        Model_test_data = test_data[test_data['combinedinput']==combinedinput]
        
        lower_threshold = train_data_properties[train_data_properties['combinedinput']==combinedinput]['lower'].values     
        upper_threshold = train_data_properties[train_data_properties['combinedinput']==combinedinput]['upper'].values        
        Model_test_data['Anomaly_tag'] = Model_test_data['product_online_price'].apply(find_anomalies,args=(lower_threshold,upper_threshold))
        
        Model_test_data=Model_test_data[Model_test_data['Anomaly_tag']==1]        
        Test_Output_data = Test_Output_data.append(Model_test_data) 
        print(id_count)
    Test_Output_data['product_online_price'] = "$"+Test_Output_data['product_online_price'].astype(str)         
    Test_Output_data_all = pd.concat([Test_Output_data,Anomaly_Preprocess],axis=0)
    Test_Output_data_all['Anomaly_tag'] = Test_Output_data_all['Anomaly_tag'].astype('int')
    print(Test_Output_data_all)
    Site_id_name = {1:'walmart_1',4:'amazon_4',22:'homedepot_22',34:'officedepot_34',37:'overstock_37',46:'staples_46',52:'wayfair_52'}
    for site_id in test_data['siteid'].unique():
        site_id_df = Test_Output_data_all[Test_Output_data_all['siteid']==site_id] 
        site_id_df['combinedinput']=site_id_df['combinedinput'].str.upper()
        print(site_id_df.head())
        out_filename="{}{}_{}.txt".format(jsonData['Anomaly_output'],Site_id_name.get(site_id),datetime.now().date(),encoding="ANSI")            
        print(out_filename)            
        site_id_df.to_csv(out_filename,header=True, index=None,sep='\t') 
        FTP_Upload(out_filename)
        anomaly_count = len(Test_Output_data_all)
        em.send_mail('SUCCESS',"OK","{} in {}".format(site_id,files),'',"Completed successfully",anomaly_count)
        log.error_logger('SUCCESS',"OK","{} in {}".format(site_id,files),'',"Completed successfully")
       
        
        
    Test_data_summary_report = pd.DataFrame({'siteid':site_id,'Null values':missing_values_df['combinedinput'].nunique(),'Special char':special_char_df['combinedinput'].nunique(),'zero in price':zero_price_df['combinedinput'].nunique(),'Anomalies':Test_Output_data_all['combinedinput'].nunique(),'Normal':(test_data['combinedinput'].nunique()- Test_Output_data_all['combinedinput'].nunique())},index=range(0,1))
    
    Test_data_summary_report_rows = pd.DataFrame({'siteid':site_id,'Null values':len(missing_values_df),'Special char':len(special_char_df),'zero in price':len(zero_price_df),'Anomalies':len(Test_Output_data_all),'Normal':(len(test_data)- len(Test_Output_data_all))},index=range(0,1))
     
    Test_data_summary_report.to_csv('{}Test_data_summary_report_RuleBased_{}.txt'.format(jsonData['Test data summary'],datetime.now().date()),header=True, index=None, sep=' ', mode='a')
    Test_data_summary_report_rows.to_csv('{}Test_data_summary_report_rows_RuleBased_{}.txt'.format(jsonData['Test data summary'],datetime.now().date()),header=True, index=None, sep=' ', mode='a')
    print("exit fit model")
    
  
    
#-------------------------------------------------------------------FTP Upload---------------------------------------   
def FTP_Upload(FILENAME):
    HOST=jsonData["HOST"]
    USER=jsonData["USERNAME"]
    PASSWORD=jsonData["PSWD"]
    PATH=jsonData["Output_path"]
    ftp = ftplib.FTP(HOST,USER,PASSWORD)
    with open(FILENAME, "rb") as file:
        try:
            ftp.encoding = "utf-8" #forces the FTP server to use UTF-8 Encoding
            ftp.cwd(PATH) #Changes the Current working directory of FTP Server
            ftp.storbinary("STOR {}".format(os.path.basename(FILENAME)), file) # use FTP's STOR command to upload the file           
            print(os.path.basename(FILENAME))
            print("Uploaded Successfully in FTP")
        except ftplib.all_errors as error:
            print("Error occured while uploading in FTP")
            em.send_mail('CRITICAL',"FTP Upload",'','','{} error occured while uploading the output'.format(traceback.format_exc()))
            log.error_logger('CRITICAL',"FTP Upload",'','','{} error occured while uploading the output'.format(traceback.format_exc()))
        if ftp != None:
            ftp.close()

#-------------------------------------------------------------------FTP Mover---------------------------------------               
def FTP_Mover(filepathSource,filepathDestination):
    HOST=jsonData["HOST"]
    USER=jsonData["USERNAME"]
    PASSWORD=jsonData["PSWD"]
    PATH=jsonData["Output_path"]
    ftp = ftplib.FTP(HOST,USER,PASSWORD)
    print(filepathSource)
    
    try:
        ftp.rename(filepathSource, filepathDestination)
        print("Files moved successfully")
    except ftplib.all_errors as error:
        print("Error occured while moving the files in FTP")
        em.send_mail('CRITICAL',"FTP Upload",'','','{} error occured while moving the files in FTP'.format(traceback.format_exc()))
        log.error_logger('CRITICAL',"FTP Upload",'','','{} error occured while moving the files in FTP'.format(traceback.format_exc()))
    if ftp != None:
        ftp.close()		
#-------------------------------------------------------------------Main Function---------------------------------------    
if __name__ == "__main__":
    try: 
       sitefilename=""
       jsonData=json.load(open("Frigginyeah_model_config.json","r",encoding="utf-8"))
       FTP_download()
       df = pd.DataFrame()       
       read_files = glob("{}*.txt".format(jsonData['Input_file_path']))      
       if read_files:
           for files in read_files: 
                print(files)
                try:
                    df = pd.read_csv(files, delimiter="\t",encoding='ANSI')
                except UnicodeDecodeError:
                    em.send_mail('CRITICAL',"Data Acceptence",files,'','Files not in ANSI Encoding')
                    log.error_logger('CRITICAL',"Data Acceptence",files,'','Files not in ANSI Encoding')
                    quit()
                isAccept = Data_Acceptance(df)
                if isAccept:
                    df = df.reset_index(drop=True)
                    df.rename(columns={'index':'previous_index'},inplace=True)
                    tt_data = df.copy()
                    test_data=Data_Preprocessing(tt_data)
                    print(test_data)
                    #Test_Data_Prediction(test_data) 
                    Test_data_Median_Pred(test_data)                   
                    FTP_Mover(jsonData['dirName_test']+"/{}".format(os.path.basename(files)),jsonData['dirName_Archive']+"/{}".format(os.path.basename(files)))
                    shutil.move(jsonData["Input_file_path"]+"/{}".format(os.path.basename(files)),jsonData["Archive_path"]+"/{}".format(os.path.basename(files)))
                else:
                    print(files,"file not accepted")
    except:
        print(traceback.format_exc()) #To capture errors
        em.send_mail('CRITICAL',"",'','','{} Error while running the main function'.format(traceback.format_exc()))
        log.error_logger('CRITICAL',"",'','','{} Error while running the main function'.format(traceback.format_exc()))
      

