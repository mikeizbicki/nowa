import os.path
import reader

def loaddata(FLAGS):
    print('loading dataset ptb')
    datadir_dataset=os.path.join(FLAGS.input_data_dir,FLAGS.dataset)
    dataset_train,dataset_valid,dataset_test,_=reader.ptb_raw_data(datadir_dataset)
    return (dataset_train,dataset_valid,dataset_test)
