import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import psycopg2
import pandas as pd
import numpy as np
import functions as fn
from glob import glob
import model_code

def main():   
    fn.claim_log()
    fn.data_folder_gen()
    download_list=fn.download_serial_check()
    for serial_name in download_list:
        claim_df=pd.read_csv('claim_log_test.csv')
        df=claim_df[claim_df['serial_number']==serial_name]
        #data=fn.db_train_data_downlod(serial_name)
        data=pd.read_csv('../db_data/'+ serial_name + '.csv')
        cell_num=df['cell_num'].iloc[0]
        end_time=df['db_times'].iloc[0]
        casue_code=df['cause_code'].iloc[0]
        battery_model=df['battery_model'].iloc[0]
        if str(casue_code)=='11':
            fn.pre_processing_rul(data=data,serial=serial_name, 
                                cell_num=cell_num, end_time=end_time,
                                casue_code=casue_code,battery_model=battery_model)
        else:
            fn.pre_processing_abd(data=data,serial=serial_name, 
                                cell_num=cell_num, end_time=end_time,
                                casue_code=casue_code,battery_model=battery_model)
    
    train_battery_list=fn.train_availability_check(num_set=num_set)
    task=train_battery_list['task'].drop_duplicates()
    sample_length_list=train_battery_list['sample_length'].drop_duplicates()
    battery_model_list=train_battery_list['battery_model'].drop_duplicates()
    for battery_model_name in battery_model_list:
        for task_name in task:
            model_list=pd.read_csv(task_name + '_model_list.csv')
            use_count=0
            for sample_length_name in sample_length_list:
                tmp=train_battery_list[train_battery_list['battery_model']==battery_model_name]
                tmp2=tmp[tmp['task']==task_name]
                tmp3=tmp2[tmp2['sample_length']==str(sample_length_name)]
                if tmp3['availability'].iloc[0]==1:
                    use_count=use_count+1
            if use_count>=3:
                X_train_5000, Y_train_5000=fn.data_load(battery_model=battery_model_name,task=task_name,sample_length=5000)
                model_code.model_train(X_train_5000,Y_train_5000,model_list,battery_model=battery_model_name,task=task_name,sample_length=5000)
                X_train_25000, Y_train_25000=fn.data_load(battery_model=battery_model_name,task=task_name,sample_length=25000)
                model_code.model_train(X_train_25000,Y_train_25000,model_list,battery_model=battery_model_name,task=task_name,sample_length=25000)
                X_train_50000, Y_train_50000=fn.data_load(battery_model=battery_model_name,task=task_name,sample_length=50000)
                model_code.model_train(X_train_50000,Y_train_50000,model_list,battery_model=battery_model_name,task=task_name,sample_length=50000)
            else:
                print('Not enough data set : ' + battery_model_name + '/'+ task_name)

if __name__ == '__main__':
    num_set=50000