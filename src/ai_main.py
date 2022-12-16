import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import functions as fn
import numpy as np
import datetime
import model_code

def main(alpha=15000,sample_length=25000):
    record_count=sample_length+alpha
    result, battery_model_list=fn.serial_load()
    data_save_info=fn.train_availability_check(num_set=50000)
    serial_list=fn.use_check(result,battery_model_list,data_save_info)
    for idx, serial_number in enumerate(list(serial_list['serial_number'])):
        if (serial_list['RUL_availability'].iloc[idx]==1)and(serial_list['ABD_availability'].iloc[idx]==1):
            print('===============================================')
            db_data = fn.db_data_load(serial_number, record_count)
            if len(db_data) < sample_length:
                fn.result_db_send(serial=serial_number,rul=None,abd=None, usage=None, reliable = False)
                print('AI process failed to complete. : lack of data')
                continue
            x_data = fn.pre_processing(db_data,serial_number,sample_length)
            if isinstance(x_data,bool):
                db_data = fn.db_data_load(serial_number, record_count+alpha)
                x_data = fn.pre_processing(db_data,serial_number,sample_length)
                if isinstance(x_data,bool):
                    fn.result_db_send(serial=serial_number,rul=None,abd=None, usage=None, reliable = False)
                    print('AI process failed to complete. : lack of data')                
                    continue
            x_data_5000 = fn.pre_processing(db_data,serial_number,5000)
            rul_5000,abd_5000=model_code.model_test(x_data_5000,battery_model=battery_model_list[idx],sample_length=5000,serial_num=serial_number)
            x_data_25000 = fn.pre_processing(db_data,serial_number,25000)
            rul_25000,abd_25000=model_code.model_test(x_data_25000,battery_model=battery_model_list[idx],sample_length=25000,serial_num=serial_number)
            x_data_50000 = fn.pre_processing(db_data,serial_number,50000)
            rul_50000,abd_50000=model_code.model_test(x_data_50000,battery_model=battery_model_list[idx],sample_length=50000,serial_num=serial_number)
            rul=0.5*rul_50000 + 0.3*rul_25000 + 0.2*rul_5000
            usage=(abd_5000+abd_25000+abd_50000)/3
            abd=bool(np.around((abd_5000+abd_25000+abd_50000)/3))
            fn.result_db_send(serial=serial_number,rul=rul,abd=abd, usage=usage, reliable = True)
            now = datetime.datetime.now()
            print("AI process Success! : " + str(now))
            print('===============================================')
        else:
            fn.result_db_send(serial=serial_number,rul=1800,abd=0, usage=0, reliable = False)
            print('AI process failed to complete. : No model was created.')  

if __name__ == '__main__':
    alpha = 25000
    sample_length = 50000
    main(alpha,sample_length)

