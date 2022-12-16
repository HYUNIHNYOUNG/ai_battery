import psycopg2
import pandas as pd
import numpy as np
import os
from glob import glob

def claim_log():
    try:
        connection = psycopg2.connect(
            host="repo.hass.wisoft.space",
            dbname="wisoft",
            user="robovolt",
            password="E7C2Mn0dkQexVJmnMX2zXoV8oEPY00RpN",
            port=10002
        )

        cursor = connection.cursor()
        print("connected PostgreSQL. [claim log fetch]")
        #print("PostgreSQL server information")
        #print(connection.get_dsn_parameters(), "\n")
        cursor.execute("""SELECT b.comm_serial_no, c.fault_symptom_code, c.fault_cause_code, bm.name, c.reg_datetime, c.fault_cell
                        FROM claim c, battery bt, battery_model bm, bms b
                        WHERE c.battery_id = bt.id
                        AND bt.battery_model_id = bm.id
                        AND c.battery_id = b.battery_id;""")
        # Fetch result
        record = cursor.fetchall()
        #print(F"You are connected to - {record}\n")
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
    df=pd.DataFrame(record,columns=['serial_number','symptom_code','cause_code','battery_model','datetimes','cell_num'])
    df.to_csv('claim_log.csv',index=False)

    df=pd.read_csv('claim_log.csv')
    df['db_times']=df['datetimes'].str.split('.',expand=True)[0]
    df['db_times']=pd.to_datetime(df['db_times'],format="%Y-%m-%d %H:%M:%S")
    df.to_csv('claim_log.csv',index=False)

def data_folder_gen():
    df=pd.read_csv('claim_log_test.csv')
    battery_models=list(df['battery_model'].drop_duplicates())
    cause_list=list(df['cause_code'].drop_duplicates())
    default_folder='../data/'
    sample_lengths=[5000,25000,50000]
    for battery_model in battery_models:
        if not os.path.exists(default_folder + battery_model):
            os.mkdir(default_folder + battery_model)
        for sample_length in sample_lengths:
            if not os.path.exists(default_folder + battery_model + '/' + str(sample_length)):
                os.mkdir(default_folder + battery_model + '/' + str(sample_length))
            if not os.path.exists(default_folder + battery_model + '/' + str(sample_length) + '/normal'):
                os.mkdir(default_folder + battery_model + '/' + str(sample_length) + '/normal')
            if not os.path.exists(default_folder + battery_model + '/' + str(sample_length) + '/abnormal'):
                os.mkdir(default_folder + battery_model + '/' + str(sample_length) + '/abnormal')

            for cause_code in cause_list:
                if not os.path.exists(default_folder + battery_model + '/' + str(sample_length) + '/abnormal/' + str(cause_code)):
                    os.mkdir(default_folder + battery_model + '/' + str(sample_length) + '/abnormal/' + str(cause_code))

            for cause_code in cause_list:
                if not os.path.exists(default_folder + battery_model + '/' + str(sample_length) + '/normal/' + str(cause_code)):
                    os.mkdir(default_folder + battery_model + '/' + str(sample_length) + '/normal/' + str(cause_code))

def download_serial_check():
    df=pd.read_csv('claim_log_test.csv')
    battery_list=list(df['serial_number'])
    npz_list=glob('../data/*/5000/abnormal/*/*.npz')
    serial_list=[]
    for npz in npz_list:
        tmp=npz.split('/')[-1].split('.')[0]
        serial_list.append(tmp)
    for name in serial_list:
        battery_list.remove(name)
    download_list=battery_list
    return download_list


def db_train_data_downlod(serial):
    print(str(serial) + " loading...")
    try:
        connection = psycopg2.connect(
            host="repo.hass.wisoft.space",
            dbname="wisoft",
            user="robovolt",
            password="E7C2Mn0dkQexVJmnMX2zXoV8oEPY00RpN",
            port=10002
        )
        print("DB connection completed!!")
        cursor = connection.cursor()
        #print("PostgreSQL server information")
        #print(connection.get_dsn_parameters(), "\n")
        cursor.execute("SELECT version();")
        # Fetch result
        record = cursor.fetchone()
        #print(F"You are connected to - {record}\n")
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
    cursor.execute(f"""
                            SELECT comm_serial_no                                                                AS "BMS No."
                        , datetime + '9 hour'::INTERVAL                                            AS "Date Time"
                        , created_at + '9 hour'::INTERVAL                                           AS "DB Time"
                        , (data ->> 'soc')::NUMERIC                                                 AS soc
                        , (data ->> 'soh')::NUMERIC                                                 AS soh
                        , (data ->> 'total_voltage')::NUMERIC                                       AS "Total Vol."
                        , (data ->> 'current')::NUMERIC                                             AS "Total Amp."
                        , (data ->> 'ac_state')::NUMERIC                                            AS "AC State"
                        , (data ->> 'dc_state')::NUMERIC                                            AS "DC State"
                        , (data ->> 'balancing_state')::NUMERIC                                     AS "Balancing State"
                        , (data ->> 'over_temp')::NUMERIC                                           AS "Over Temp."
                        , (data ->> 'low_temp')::NUMERIC                                            AS "Low Temp."
                        , (data ->> 'cell_over_vol')::NUMERIC                                       AS "Cell Overvol."
                        , (data ->> 'cell_low_vol')::NUMERIC                                        AS "Cell Low Vol."
                        , (data ->> 'process_code')::NUMERIC                                        AS "Process Code"
                        , (data ->> 'cell_vols_max')::NUMERIC                                       AS "Cell Max Vol."
                        , (data ->> 'cell_vols_min')::NUMERIC                                       AS "Cell Min Vol."
                        , (data ->> 'cell_vols_max')::NUMERIC - (data ->> 'cell_vols_min')::NUMERIC AS "Cell Vol Diff."
                        , (data ->> 'total_over_vol')::NUMERIC                                      AS "Total Over Vol."
                        , (data ->> 'total_low_vol')::NUMERIC                                       AS "Total Low Vol."
                        , (data ->> 'cell_deviation')::NUMERIC                                      AS "Cell Vol Deviation"
                        , (data ->> 'temp_deviation')::NUMERIC                                      AS "Temp Diff."
                        , (data ->> 'charging_over_cur')::NUMERIC                                   AS "Charging Overcurr."
                        , (data ->> 'discharging_over_cur')::NUMERIC                                AS "DisCharging Overcurr."
                        , (data -> 'cell_group1' ->> 'temp')::NUMERIC                               AS "CG1 Temp."
                        , (data -> 'cell_group1' -> 'cell_vols' ->> '1')::NUMERIC                   AS c01
                        , (data -> 'cell_group1' -> 'cell_vols' ->> '2')::NUMERIC                   AS c02
                        , (data -> 'cell_group1' -> 'cell_vols' ->> '3')::NUMERIC                   AS c03
                        , (data -> 'cell_group1' -> 'cell_vols' ->> '4')::NUMERIC                   AS c04
                        , (data -> 'cell_group1' -> 'cell_vols' ->> '5')::NUMERIC                   AS c05
                        , (data -> 'cell_group1' -> 'cell_vols' ->> '6')::NUMERIC                   AS c06
                        , (data -> 'cell_group1' -> 'cell_vols' ->> '7')::NUMERIC                   AS c07
                        , (data -> 'cell_group1' -> 'cell_vols' ->> '8')::NUMERIC                   AS c08
                        , (data -> 'cell_group1' -> 'cell_vols' ->> '9')::NUMERIC                   AS c09
                        , (data -> 'cell_group1' -> 'cell_vols' ->> '10')::NUMERIC                  AS c10
                        , (data -> 'cell_group1' -> 'cell_vols' ->> '11')::NUMERIC                  AS c11
                        , (data -> 'cell_group1' -> 'cell_vols' ->> '12')::NUMERIC                  AS c12
                        , (data -> 'cell_group2' ->> 'temp')::NUMERIC                               AS "CG2 Temp."
                        , (data -> 'cell_group2' -> 'cell_vols' ->> '1')::NUMERIC                   AS c13
                        , (data -> 'cell_group2' -> 'cell_vols' ->> '2')::NUMERIC                   AS c14
                        , (data -> 'cell_group2' -> 'cell_vols' ->> '3')::NUMERIC                   AS c15
                        , (data -> 'cell_group2' -> 'cell_vols' ->> '4')::NUMERIC                   AS c16
                        , (data -> 'cell_group2' -> 'cell_vols' ->> '5')::NUMERIC                   AS c17
                        , (data -> 'cell_group2' -> 'cell_vols' ->> '6')::NUMERIC                   AS c18
                        , (data -> 'cell_group2' -> 'cell_vols' ->> '7')::NUMERIC                   AS c19
                        , (data -> 'cell_group2' -> 'cell_vols' ->> '8')::NUMERIC                   AS c20
                        , (data -> 'cell_group2' -> 'cell_vols' ->> '9')::NUMERIC                   AS c21
                        , (data -> 'cell_group2' -> 'cell_vols' ->> '10')::NUMERIC                  AS c22
                        , (data -> 'cell_group2' -> 'cell_vols' ->> '11')::NUMERIC                  AS c23
                        , (data -> 'cell_group2' -> 'cell_vols' ->> '12')::NUMERIC                  AS c24
                    FROM sensing_data
                    WHERE comm_serial_no = '{serial}'
                    ORDER BY id DESC;""")

    result = cursor.fetchall()

    if (connection):
        cursor.close()
        connection.close()
        #print("PostgreSQL connection is closed")
    result=pd.DataFrame(result)
    print(">> " + str(len(result)) + " data from DB.")

    result.columns = ['serial','time','db_time','soc','soh','total_voltage','current','ac_state','dc_state','balancing_state','over_temp',
                    'low_temp','cell_over_vol','cell_low_vol','process_code','cell_vols_max','cell_vols_min','cell_vols_diff',
                    'total_over_vol','total_low_vol','cell_deviation','temp_deviation','charging_over_cur','discharging_over',
                    'cell_temp_1','cell_vols_1','cell_vols_2','cell_vols_3','cell_vols_4','cell_vols_5','cell_vols_6','cell_vols_7',
                    'cell_vols_8','cell_vols_9','cell_vols_10','cell_vols_11','cell_vols_12','cell_temp_2','cell_vols_13','cell_vols_14',
                    'cell_vols_15','cell_vols_16','cell_vols_17','cell_vols_18','cell_vols_19','cell_vols_20','cell_vols_21','cell_vols_22',
                    'cell_vols_23','cell_vols_24']

    return result


def pre_processing_rul(data,serial, cell_num=0, end_time='',casue_code='',battery_model=''):
    sample_lengths=[5000,25000,50000]
    print(serial + ' Pre-processing...')
    data['db_time']=data['db_time'].astype('string')
    data['db_time']=data['db_time'].str.split('+').str[0]
    data['db_time']=pd.to_datetime(data['db_time'])
    data=data.sort_values(by='db_time').reset_index(drop=True)
    data=data[~((data['current']<=0)&(data['current']>=-2))].reset_index(drop=True)
    data['cell_temp']=(data['cell_temp_1']+data['cell_temp_2'])/2
    data['rul']=(pd.to_datetime(end_time)-data['db_time']).dt.days/(4.5)
    cell_name='cell_vols_' + str(cell_num)
    abnormal_data=data[[cell_name,'cell_temp','current','rul']]
    for sample_length in sample_lengths:
        N_set=int(len(data)/sample_length)
        if N_set >= 1:
            V_data=np.zeros([N_set,int(sample_length)])
            I_data=np.zeros([N_set,int(sample_length)])
            T_data=np.zeros([N_set,int(sample_length)])
            label=np.zeros([N_set])
            for iter in range(N_set):
                V_data[iter,:] = abnormal_data[cell_name][sample_length*iter:sample_length*(iter+1)]
                I_data[iter,:] = abnormal_data['current'][sample_length*iter:sample_length*(iter+1)]
                T_data[iter,:] = abnormal_data['cell_temp'][sample_length*iter:sample_length*(iter+1)]
                label[iter]=abnormal_data['rul'][sample_length*(iter+1)-1]
            save_path='../data/' + battery_model + '/' + str(sample_length) + '/abnormal/' + str(casue_code) + '/'
            print(save_path + ' : processing...')
            np.savez_compressed(save_path + serial + '.npz',V=V_data,I=I_data,T=T_data,label=label)
            print(">> Data Pre-processing complete.")
        else:
            print(">> Get more data.")
            return False

def pre_processing_abd(data,serial, cell_num=0, end_time='',casue_code='',battery_model=''):
    sample_lengths=[5000,25000,50000]
    print(serial + ' Pre-processing...')
    data['db_time']=data['db_time'].astype('string')
    data['db_time']=data['db_time'].str.split('+').str[0]
    data['db_time']=pd.to_datetime(data['db_time'])
    data=data.sort_values(by='db_time').reset_index(drop=True)
    data=data[~((data['current']<=0)&(data['current']>=-2))].reset_index(drop=True)
    data['cell_temp']=(data['cell_temp_1']+data['cell_temp_2'])/2
    data['rul']=(pd.to_datetime(end_time)-data['db_time']).dt.days/(4.5)
    cell_name='cell_vols_' + str(cell_num)
    v_idx=[ 'cell_vols_1','cell_vols_2','cell_vols_3','cell_vols_4','cell_vols_5','cell_vols_6','cell_vols_7',
                    'cell_vols_8','cell_vols_9','cell_vols_10','cell_vols_11','cell_vols_12','cell_vols_13','cell_vols_14',
                    'cell_vols_15','cell_vols_16','cell_vols_17','cell_vols_18','cell_vols_19','cell_vols_20','cell_vols_21',
                    'cell_vols_22','cell_vols_23','cell_vols_24']
    use_idx=[]
    for v_num in v_idx:
        if data[v_num].mean() < 1:
            data=data.drop([v_num],axis=1)
        else:
            use_idx.append(v_num)
    use_idx.remove(cell_name)
    abnormal_data=data[[cell_name,'cell_temp','current','rul']]
    abnormal_data= abnormal_data[abnormal_data['rul'] <= 3].reset_index(drop=True)
    normal_data=data.drop([cell_name],axis=1)
    for sample_length in sample_lengths:
        N_set=int(len(abnormal_data)/sample_length)
        if N_set >= 1:
            V_data=np.zeros([N_set,int(sample_length)])
            I_data=np.zeros([N_set,int(sample_length)])
            T_data=np.zeros([N_set,int(sample_length)])
            label=np.zeros([N_set])
            for iter in range(N_set):
                V_data[iter,:] = abnormal_data[cell_name][sample_length*iter:sample_length*(iter+1)]
                I_data[iter,:] = abnormal_data['current'][sample_length*iter:sample_length*(iter+1)]
                T_data[iter,:] = abnormal_data['cell_temp'][sample_length*iter:sample_length*(iter+1)]
                label[iter]=1
            save_path='../data/' + battery_model + '/' + str(sample_length) + '/abnormal/' + str(casue_code) + '/'
            print(save_path + ' : processing...')
            np.savez_compressed(save_path + serial + '.npz',V=V_data,I=I_data,T=T_data,label=label)
            print(">> Data Pre-processing complete.")
        else:
            print(">> Get more data.")
            return False

    for sample_length in sample_lengths:
        N_set=int(len(normal_data)/sample_length)
        if N_set >= 1:
            V_data=np.zeros([N_set*len(use_idx),int(sample_length)])
            I_data=np.zeros([N_set*len(use_idx),int(sample_length)])
            T_data=np.zeros([N_set*len(use_idx),int(sample_length)])
            label=np.zeros([N_set*len(use_idx)])
            for iter in range(N_set):
                for n in range(len(use_idx)):
                    V_data[iter*len(use_idx) + n ,:] = normal_data[use_idx[n]][sample_length*iter:sample_length*(iter+1)]
                    I_data[iter*len(use_idx) + n ,:] = normal_data['current'][sample_length*iter:sample_length*(iter+1)]
                    T_data[iter*len(use_idx) + n ,:] = normal_data['cell_temp'][sample_length*iter:sample_length*(iter+1)]
                    label[iter*len(use_idx) + n]=0

            save_path='../data/' + battery_model + '/' + str(sample_length) + '/normal/' + str(casue_code) + '/'
            print(save_path + ' : processing...')
            np.savez_compressed(save_path + serial + '.npz',V=V_data,I=I_data,T=T_data,label=label)
            print(">> Data Pre-processing complete.")
        else:
            print(">> Get more data.")
            return False


def train_availability_check(num_set=50000):
    battery_df=pd.DataFrame(columns=['battery_model','task','sample_length','availability','count'])
    battery_model_list=glob('../data/*')
    for battery_model in battery_model_list:
        sample_length_list=glob(battery_model +'/*')
        for sample_length_name in sample_length_list:
            cause_list=glob(sample_length_name +'/abnormal/*')
            abd_count=0
            for cause_code in cause_list:
                if cause_code.split('/')[-1]=='11':
                    npz_list=glob(cause_code +'/*.npz')
                    count=0
                    for npz_name in npz_list:
                        data=np.load(npz_name)
                        count=count+len(data['label'])
                    if count >= num_set:
                        availability=1
                    else:
                        availability=0
                    battery_df=battery_df.append({'battery_model':battery_model.split('/')[-1],
                                                'task':'RUL',
                                                'sample_length':sample_length_name.split('/')[-1],
                                                'count':count,
                                                'availability':availability},ignore_index=True)
                else:
                    npz_list=glob(cause_code +'/*.npz')
                    abd_count=0
                    for npz_name in npz_list:
                        data=np.load(npz_name)
                        abd_count=abd_count+len(data['label'])
            if abd_count >= num_set:
                availability=1
            else:
                availability=0
            battery_df=battery_df.append({'battery_model':battery_model.split('/')[-1],
                                            'task':'ABD',
                                            'sample_length':sample_length_name.split('/')[-1],
                                            'count':abd_count,
                                            'availability':availability},ignore_index=True)
    return battery_df

def data_load(battery_model='',task='',sample_length=0):
    V_mean=np.load('./mean_std/V_mean.npy')
    I_mean=np.load('./mean_std/I_mean.npy')
    T_mean=np.load('./mean_std/T_mean.npy')
    V_std=np.load('./mean_std/V_std.npy')
    I_std=np.load('./mean_std/I_std.npy')
    T_std=np.load('./mean_std/T_std.npy')

    if task=='RUL':
        data_path='../data/' + battery_model +'/' + str(sample_length) + '/abnormal/11/'
        data_list=glob(data_path + '*.npz')
        for idx, npz_path in enumerate(data_list):
            if idx==0:
                data=np.load(npz_path)
                V=data['V']
                I=data['I']
                T=data['T']
                label=data['label']
            else:
                data=np.load(npz_path)
                V_tmp=data['V']
                I_tmp=data['I']
                T_tmp=data['T']
                label_tmp=data['label']
                V=np.concatenate((V, V_tmp), axis=0)
                I=np.concatenate((I, I_tmp), axis=0)
                T=np.concatenate((T, T_tmp), axis=0)
                label=np.concatenate((label, label_tmp), axis=0)
    else:
        data_path='../data/' + battery_model +'/' + str(sample_length) + '/*/'
        data_list=glob(data_path + '*/*.npz')
        data_list2=[]
        for data_name in data_list:
            if '11' in data_name.split('/'):
                continue
            else:
                data_list2.append(data_name)
        for idx, npz_path in enumerate(data_list2):
            if idx==0:
                data=np.load(npz_path)
                V=data['V']
                I=data['I']
                T=data['T']
                label=data['label']
            else:
                data=np.load(npz_path)
                V_tmp=data['V']
                I_tmp=data['I']
                T_tmp=data['T']
                label_tmp=data['label']
                V=np.concatenate((V, V_tmp), axis=0)
                I=np.concatenate((I, I_tmp), axis=0)
                T=np.concatenate((T, T_tmp), axis=0)
                label=np.concatenate((label, label_tmp), axis=0)
    V = (V - V_mean) / V_std
    I = (I - I_mean) / I_std
    T = (T - T_mean) / T_std
    X_train=np.stack((V,I,T),axis=2)
    Y_train=label
    X_train=np.stack((V,I,T),axis=2)
    Y_train=label

    return X_train, Y_train

def result_db_send(serial=None,rul=None,abd=None, usage=None, reliable = None):
    print(str(serial) + " sending..")
    try:
        connection = psycopg2.connect(
            host="repo.hass.wisoft.space",
            dbname="wisoft",
            user="robovolt",
            password="E7C2Mn0dkQexVJmnMX2zXoV8oEPY00RpN",
            port=10002
        )
        cursor = connection.cursor()
        #print("PostgreSQL server information")
        #print(connection.get_dsn_parameters(), "\n")
        cursor.execute("SELECT version();")
        # Fetch result
        record = cursor.fetchone()
        #print(F"You are connected to - {record}\n")
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

    sql = "INSERT INTO ai_predict (serial, remaining_useful_life, abnormal_battery_detection, battery_usage_period, model_availability) VALUES(%s, %s, %s, %s, %s);"
    cursor.execute(sql, (serial, rul, abd, usage, reliable))

    connection.commit()

    cursor.execute("select * from ai_predict;")
    result = cursor.fetchall()

    if (connection):
        cursor.close()
        connection.close()
        #print("PostgreSQL connection is closed")

def serial_load():
    print("serial loding...")
    try:
        connection = psycopg2.connect(
            host="repo.hass.wisoft.space",
            dbname="wisoft",
            user="robovolt",
            password="E7C2Mn0dkQexVJmnMX2zXoV8oEPY00RpN",
            port=10002
        )
        cursor = connection.cursor()
        #print("PostgreSQL server information")
        #print(connection.get_dsn_parameters(), "\n")
        cursor.execute("SELECT version();")
        # Fetch result
        record = cursor.fetchone()
        #print(F"You are connected to - {record}\n")
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

    cursor.execute(f"""
        select distinct comm_serial_no
    from sensing_data;""")

    result = cursor.fetchall()

    if (connection):
        cursor.close()
        connection.close()
        #print("PostgreSQL connection is closed")


    battery_model_list=[]
    for output in np.sort(np.squeeze(result)):
        name=output[0:2]+str(int(output[2:6])) + str(int(output[6:10]))
        battery_model_list.append(name)

    return np.sort(np.squeeze(result)), battery_model_list


def use_check(result,battery_model_list,data_save_info):
    df=pd.DataFrame(columns=['serial_number','battery_model','RUL_availability','ABD_availability'])
    df['serial_number']=result
    df['battery_model']=battery_model_list
    for idx,serial_name in enumerate(result):
        tmp_df=data_save_info[(data_save_info['battery_model']==battery_model_list[18])&(data_save_info['task']=='RUL')&(data_save_info['sample_length']=='50000')]
        if tmp_df['availability'].iloc[0]==1:
            df['RUL_availability'].iloc[idx]=1
        else:
            df['RUL_availability'].iloc[idx]=0

        tmp_df2=data_save_info[(data_save_info['battery_model']==battery_model_list[18])&(data_save_info['task']=='ABD')&(data_save_info['sample_length']=='50000')]
        if tmp_df2['availability'].iloc[0]==1:
            df['ABD_availability'].iloc[idx]=1
        else:
            df['ABD_availability'].iloc[idx]=0

    return df

def db_data_load(serial, record_count=0):
    print(str(serial) + " loading...")
    print("Amount of data : " + str(record_count))
    try:
        connection = psycopg2.connect(
            host="repo.hass.wisoft.space",
            dbname="wisoft",
            user="robovolt",
            password="E7C2Mn0dkQexVJmnMX2zXoV8oEPY00RpN",
            port=10002
        )
        print("DB connection completed!!")
        cursor = connection.cursor()
        #print("PostgreSQL server information")
        #print(connection.get_dsn_parameters(), "\n")
        cursor.execute("SELECT version();")
        # Fetch result
        record = cursor.fetchone()
        #print(F"You are connected to - {record}\n")
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
    cursor.execute(f"""
                SELECT comm_serial_no                                                                AS "BMS No."
            , datetime + '9 hour'::INTERVAL                                            AS "Date Time"
            , created_at + '9 hour'::INTERVAL                                           AS "DB Time"
            , (data ->> 'soc')::NUMERIC                                                 AS soc
            , (data ->> 'soh')::NUMERIC                                                 AS soh
            , (data ->> 'total_voltage')::NUMERIC                                       AS "Total Vol."
            , (data ->> 'current')::NUMERIC                                             AS "Total Amp."
            , (data ->> 'ac_state')::NUMERIC                                            AS "AC State"
            , (data ->> 'dc_state')::NUMERIC                                            AS "DC State"
            , (data ->> 'balancing_state')::NUMERIC                                     AS "Balancing State"
            , (data ->> 'over_temp')::NUMERIC                                           AS "Over Temp."
            , (data ->> 'low_temp')::NUMERIC                                            AS "Low Temp."
            , (data ->> 'cell_over_vol')::NUMERIC                                       AS "Cell Overvol."
            , (data ->> 'cell_low_vol')::NUMERIC                                        AS "Cell Low Vol."
            , (data ->> 'process_code')::NUMERIC                                        AS "Process Code"
            , (data ->> 'cell_vols_max')::NUMERIC                                       AS "Cell Max Vol."
            , (data ->> 'cell_vols_min')::NUMERIC                                       AS "Cell Min Vol."
            , (data ->> 'cell_vols_max')::NUMERIC - (data ->> 'cell_vols_min')::NUMERIC AS "Cell Vol Diff."
            , (data ->> 'total_over_vol')::NUMERIC                                      AS "Total Over Vol."
            , (data ->> 'total_low_vol')::NUMERIC                                       AS "Total Low Vol."
            , (data ->> 'cell_deviation')::NUMERIC                                      AS "Cell Vol Deviation"
            , (data ->> 'temp_deviation')::NUMERIC                                      AS "Temp Diff."
            , (data ->> 'charging_over_cur')::NUMERIC                                   AS "Charging Overcurr."
            , (data ->> 'discharging_over_cur')::NUMERIC                                AS "DisCharging Overcurr."
            , (data -> 'cell_group1' ->> 'temp')::NUMERIC                               AS "CG1 Temp."
            , (data -> 'cell_group1' -> 'cell_vols' ->> '1')::NUMERIC                   AS c01
            , (data -> 'cell_group1' -> 'cell_vols' ->> '2')::NUMERIC                   AS c02
            , (data -> 'cell_group1' -> 'cell_vols' ->> '3')::NUMERIC                   AS c03
            , (data -> 'cell_group1' -> 'cell_vols' ->> '4')::NUMERIC                   AS c04
            , (data -> 'cell_group1' -> 'cell_vols' ->> '5')::NUMERIC                   AS c05
            , (data -> 'cell_group1' -> 'cell_vols' ->> '6')::NUMERIC                   AS c06
            , (data -> 'cell_group1' -> 'cell_vols' ->> '7')::NUMERIC                   AS c07
            , (data -> 'cell_group1' -> 'cell_vols' ->> '8')::NUMERIC                   AS c08
            , (data -> 'cell_group1' -> 'cell_vols' ->> '9')::NUMERIC                   AS c09
            , (data -> 'cell_group1' -> 'cell_vols' ->> '10')::NUMERIC                  AS c10
            , (data -> 'cell_group1' -> 'cell_vols' ->> '11')::NUMERIC                  AS c11
            , (data -> 'cell_group1' -> 'cell_vols' ->> '12')::NUMERIC                  AS c12
            , (data -> 'cell_group2' ->> 'temp')::NUMERIC                               AS "CG2 Temp."
            , (data -> 'cell_group2' -> 'cell_vols' ->> '1')::NUMERIC                   AS c13
            , (data -> 'cell_group2' -> 'cell_vols' ->> '2')::NUMERIC                   AS c14
            , (data -> 'cell_group2' -> 'cell_vols' ->> '3')::NUMERIC                   AS c15
            , (data -> 'cell_group2' -> 'cell_vols' ->> '4')::NUMERIC                   AS c16
            , (data -> 'cell_group2' -> 'cell_vols' ->> '5')::NUMERIC                   AS c17
            , (data -> 'cell_group2' -> 'cell_vols' ->> '6')::NUMERIC                   AS c18
            , (data -> 'cell_group2' -> 'cell_vols' ->> '7')::NUMERIC                   AS c19
            , (data -> 'cell_group2' -> 'cell_vols' ->> '8')::NUMERIC                   AS c20
            , (data -> 'cell_group2' -> 'cell_vols' ->> '9')::NUMERIC                   AS c21
            , (data -> 'cell_group2' -> 'cell_vols' ->> '10')::NUMERIC                  AS c22
            , (data -> 'cell_group2' -> 'cell_vols' ->> '11')::NUMERIC                  AS c23
            , (data -> 'cell_group2' -> 'cell_vols' ->> '12')::NUMERIC                  AS c24
        FROM sensing_data
        WHERE comm_serial_no = '{serial}'
        ORDER BY id DESC
        limit {record_count};""")

    result = cursor.fetchall()

    if (connection):
        cursor.close()
        connection.close()
        #print("PostgreSQL connection is closed")
    result=pd.DataFrame(result)
    print(">> " + str(len(result)) + " data from DB.")
    return pd.DataFrame(result)


def pre_processing(data,serial,sample_length=25000):
    print(serial + ' Pre-processing...')
    data.columns = ['serial','time','db_time','soc','soh','total_voltage','current','ac_state','dc_state','balancing_state','over_temp',
                    'low_temp','cell_over_vol','cell_low_vol','process_code','cell_vols_max','cell_vols_min','cell_vols_diff',
                    'total_over_vol','total_low_vol','cell_deviation','temp_deviation','charging_over_cur','discharging_over',
                    'cell_temp_1','cell_vols_1','cell_vols_2','cell_vols_3','cell_vols_4','cell_vols_5','cell_vols_6','cell_vols_7',
                    'cell_vols_8','cell_vols_9','cell_vols_10','cell_vols_11','cell_vols_12','cell_temp_2','cell_vols_13','cell_vols_14',
                    'cell_vols_15','cell_vols_16','cell_vols_17','cell_vols_18','cell_vols_19','cell_vols_20','cell_vols_21','cell_vols_22',
                    'cell_vols_23','cell_vols_24']
    data['db_time']=data['db_time'].astype('string')
    data['db_time']=data['db_time'].str.split('+').str[0]
    data['db_time']=pd.to_datetime(data['db_time'])
    data=data.sort_values(by='db_time').reset_index(drop=True)
    data=data[~((data['current']<=0)&(data['current']>=-2))].reset_index(drop=True)
    data['cell_temp']=(data['cell_temp_1']+data['cell_temp_2'])/2

    if len(data) >= sample_length:
        V_data=np.zeros([24,int(sample_length)])
        I_data=np.zeros([24,int(sample_length)])
        T_data=np.zeros([24,int(sample_length)])
        v_idx=[ 'cell_vols_1','cell_vols_2','cell_vols_3','cell_vols_4','cell_vols_5','cell_vols_6','cell_vols_7',
                'cell_vols_8','cell_vols_9','cell_vols_10','cell_vols_11','cell_vols_12','cell_vols_13','cell_vols_14',
                'cell_vols_15','cell_vols_16','cell_vols_17','cell_vols_18','cell_vols_19','cell_vols_20','cell_vols_21',
                'cell_vols_22','cell_vols_23','cell_vols_24']

        for iter in range(len(v_idx)):
            V_data[iter,:] = data[v_idx[iter]][0:sample_length]
            I_data[iter,:] = data['current'][0:sample_length]
            T_data[iter,:] = data['cell_temp'][0:sample_length]
        V_mean = np.load('./mean_std/V_mean.npy')
        I_mean = np.load('./mean_std/I_mean.npy')
        T_mean = np.load('./mean_std/T_mean.npy')
        V_std = np.load('./mean_std/V_std.npy')
        I_std = np.load('./mean_std/I_std.npy')
        T_std = np.load('./mean_std/T_std.npy')
        
        V = (V_data - V_mean) / V_std
        I = (I_data - I_mean) / I_std
        T = (T_data - T_mean) / T_std

        X = np.stack([V,I,T],axis=2)
        print(">> Data Pre-processing complete.")
        return X

    else:
        print(">> Get more data.")
        return False
