import pandas as pd
import os
import uuid
import ast
import argparse
import traceback
import sys
import configparser
#from fontastic import LOGGER
from sklearn.model_selection import train_test_split


def generate_test_train_data(data_path, test_size, stratify, experiments_path, experiment_uuid):
    '''[summary]
    
    Arguments:
        data_path {[type]} -- [description]
        test_size {[type]} -- [description]
        stratify {[type]} -- [description]
        experiments_path {[type]} -- [description]
        experiment_uuid {[type]} -- [description]
    '''
    print("Creating dataset for experiment {}".format(experiment_uuid))
    fonts_files = []
    if os.path.exists(data_path):
        print("Data path exists, processing font directories")
        for font_dir in os.listdir(data_path):
            print("Processing font directories {}".format(font_dir))
            if os.path.isdir(os.path.join(data_path, font_dir)):
                font_files = os.listdir(os.path.join(data_path, font_dir))
                print(os.path.join(data_path, font_dir))
                for files in font_files:
                    font_files_dict = {}
                    # skip other files
                    if not files.endswith(".jpg"):
                        continue

                    print(files)
                    font_files_dict['font_dir'] = font_dir
                    font_files_dict['filename'] = files
                    fonts_files.append(font_files_dict)
            else:
                # ignore regular files
                print('no dir')
                continue

        fonts_files_df = pd.DataFrame(fonts_files)

        print(fonts_files_df.head(20))
        print(fonts_files_df.head())
        print("Fonts files dataframe with columns {} and shape {}".format(
            fonts_files_df.columns, fonts_files_df.shape))
        print("Generating class name based on the font type")

        fonts_files_df['class'] = fonts_files_df.font_dir
        print(fonts_files_df.head())

        X = fonts_files_df['filename'].tolist()
        y = fonts_files_df['class'].tolist()
        print("Data Stratification required {}".format(stratify))

        if stratify:
            print("Shape of data {}".format(fonts_files_df.shape))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y)
            print("Size of training data {}".format(len(X_train)))
            print("Size of test data {}".format(len(X_test)))
        else:
            print("Shape of data {}".format(fonts_files_df.shape))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size)
            print("Size of training data {}".format(len(X_train)))
            print("Size of test data {}".format(len(X_test)))

        train_df = pd.DataFrame([X_train, y_train]).T
        train_df.columns = ['filename', 'class']
        test_df = pd.DataFrame([X_test, y_test]).T
        test_df.columns = ['filename', 'class']

        print("Train dataframe shape {}".format(train_df.shape))
        print("Train dataframe unique shape {}".format(train_df.filename.nunique()))
        print("Test dataframe shape {}".format(test_df.shape))
        print("Test dataframe unique shape {}".format(test_df.filename.nunique()))
        print(train_df.head())
        print(test_df.head())

        artifacts_path = os.path.join(experiments_path, experiment_uuid)
        if not os.path.exists(artifacts_path):
            print("Experiment folder does not exist, creating folder with path {}".format(artifacts_path))
            os.makedirs(artifacts_path)
        
        train_csv = os.path.join(artifacts_path, 'train.csv')
        test_csv = os.path.join(artifacts_path, 'test.csv')
        print("Writing train dataframe to {} and test dataframe to {}".format(train_csv, test_csv))
        train_df.to_csv(train_csv, index=None)
        test_df.to_csv(test_csv, index=None)

    else:
        print("Data path {} does not exist".format(data_path), exc_info=True)
        exit()

if __name__ == '__main__':
    try:
        arg_parser = argparse.ArgumentParser(
            description='Argument for generating random crops')
        arg_parser.add_argument('--config', type=str,
                                help='Path to configuration file')

        if not len(sys.argv) > 1:
            print(
                "Please pass the required command line arguments, use python module -h for help")
            exit()

        arguments = arg_parser.parse_args()
        config_file_path = arguments.config
        print("Configuration path is {}".format(config_file_path, ))

        config = configparser.ConfigParser()
        config.read(config_file_path)

        config_section = config['TEST_TRAIN_SPLIT']
        training_data_path = config_section['data_path']
        test_size = ast.literal_eval(config_section['test_size'])
        stratify = ast.literal_eval(config_section['stratify'])
        experiment_uuid = config_section['experiment_uuid']
        experiments_path = config_section['experiment_path']

        print(type(experiment_uuid))
        if experiment_uuid == '':
            experiment_uuid = str(uuid.uuid4())

        config_section['experiment_uuid'] = experiment_uuid
        with open(config_file_path, 'w') as config_write_file:
            config.write(config_write_file)
        generate_test_train_data(data_path=training_data_path,
                                 test_size=test_size,
                                 stratify=stratify,
                                 experiments_path=experiments_path,
                                 experiment_uuid=experiment_uuid)

    except Exception as e:
        print(traceback.format_exc())
