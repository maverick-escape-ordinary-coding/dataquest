#!/usr/bin/env python3.9.4

'''
__author__ = Sanjay Seetharam
__email__ = sanjayseetharam.hma@gmail.com
__version__ = 2.0
__status__ = complete
__purpose__ = 
Data Quality Validation,
1. To check for NaNs in the data
2. To check for duplication issues in the data 
3. To check the account types in the data
4. To check the age range (13 years and 120 years)
5. To check dateformat

__output__=
1. Validation <Passed><Failed>
2. Log file, 
Validation:<True><False>
Here if True, validation is successful

{'import':<True><False>,
'empty':<True><False>,
'duplicate':<True><False>,
'account_type':<True><False>,
'age_limit':<True><False>,
date_format':<True><False>}
Here if True, corresponding check failed

__improvements__=
1. If I had extra time, i could have used object oriented programming 
to take advantage of inheritence and encapsulation 
to maintain the code easily in the long run.
2. I could have pushed code generalisation further by understanding
theme of the clean data instead of inputing the columns 
explicity based on the situation.
3. In terms of quality checks, I noticed lot of additional issues
that could be checked like Incorrect Age based on the DOB, 
incorrect year, valid guid format.
'''

# Load Libraries
import os
import argparse
import logging
import pandas as pd

# Configure issues and results to log file
logging.basicConfig(            
            filename = str(os.path.basename(__file__))[:-3]+".log",
            filemode = 'a', format = '%(asctime)s - %(message)s',
            datefmt='%d-%b-%y %H:%M:%S',
            level = logging.INFO)


# %%
def import_data(file_name: str) -> pd.DataFrame:
    '''
    purpose: To import csv data and transform into dataframe

    input: path of the file in string format

    output: DataFrame
    '''
    try:
        global ERRORS

        # Read csv file as dataframe skipping the initial space in columns
        data = pd.read_csv(
            file_name,
            skipinitialspace = True)
        
        # If dataframe is successfully created
        ERRORS.update({'import':False})

        return data

    # Create Exception if file is not found
    except FileNotFoundError as file_exc:
        ERRORS.update({'import':True})
        logging.info("check_empty_data: %s",file_exc)

    # Create Exception if arguments have invalid values
    except ValueError as val_exc:
        ERRORS.update({'import':True})
        logging.info("check_empty_data: %s",val_exc)
    
    # Create Exception if file cannot be loaded
    except IOError as io_exc:
        ERRORS.update({'import':True})
        logging.info("check_empty_data: %s",io_exc)

# %%
def check_empty_data(data: pd.DataFrame):
    '''
    purpose: Check the condition of columns for NaN values
    and store the result under ERRORS

    input: DataFrame
    '''

    global ERRORS

    try:

        # checking any empty column value
        ERRORS.update({'empty':data.isna().sum().any()})

    # Create Exception if arguments have invalid values
    except ValueError as val_exc:
        ERRORS.update({'empty': True})
        logging.info("check_empty_data: %s",val_exc)

    # Create Exception if file is not found
    except FileNotFoundError as file_exc:
        ERRORS.update({'empty': True})
        logging.info("check_empty_data: %s",file_exc)

# %%
def check_duplicate_data(
    data: pd.DataFrame,
    identifier_column_name: str):
    '''
    purpose: To check for duplicates under column 'guid'

    input:
    DataFrame,
    column name of the identifier
    
    '''
    
    global ERRORS 

    try:
        
        # checking any duplicate identifier value
        ERRORS.update({
            'duplicate':data.duplicated([identifier_column_name],
            keep = False).any()})

    # Create Exception if arguments have invalid values
    except ValueError as val_exc:
        ERRORS.update({'duplicate': True})
        logging.debug("check_empty_data: %s",val_exc)

    # Create Exception if file is not found
    except FileNotFoundError as file_exc:
        ERRORS.update({'duplicate': True})
        logging.debug("check_empty_data: %s",file_exc)
    
# %%
def check_account_data(
    data: pd.DataFrame,
    account_column_name: str,
    account_type: list):
    '''
    purpose: To check if account type column has matching values

    input:
    Data Frame,
    column name of the account,
    type of accounts
    '''

    global ERRORS

    try:

        # converting column to lowercase and then matching with the reference list
        # checking if all the values do match the reference list
        ERRORS.update({'account_type':not data[account_column_name]
        .apply(lambda each_account: str(each_account).lower() in account_type)
        .all()})

    # Create Exception if arguments have invalid values
    except ValueError as val_exc:
        ERRORS.update({'account_type': True})
        logging.debug("check_empty_data: %s",val_exc)

    # Create Exception if file is not found
    except FileNotFoundError as file_exc:
        ERRORS.update({'account_type': True})
        logging.debug("check_empty_data: %s",file_exc)

# %%
def check_age_data(
    data: pd.DataFrame,
    age_column_name: str):
    '''
    purpose: Check if age is within the legal range 13 and 120

    input:
    Dataframe,
    column name of age

    '''

    global ERRORS

    try:

        # checking range of age between 13 years and 120 years
        ERRORS.update({'age_limit':not data[age_column_name]
        .between(13,120).all()})

    # Create Exception if arguments have invalid values
    except ValueError as val_exc:
        ERRORS.update({'age_limit': True})
        logging.debug("check_empty_data: %s",val_exc)

    # Create Exception if file is not found
    except FileNotFoundError as file_exc:
        ERRORS.update({'age_limit': True})
        logging.debug("check_empty_data: %s",file_exc)

# %%
def check_date_format(
    data: pd.DataFrame,
    dob_column_name: str,
    signup_column_name: str):
    '''
    purpose: Check if columns with dates are in specified format

    input:
    Dataframe,
    column name of date of birth,
    column name of signup date

    '''

    global ERRORS

    try:

        # checking date format month/day/year
        ERRORS.update({'date_format': data[[dob_column_name,signup_column_name]]
        .apply(pd.to_datetime,format='%m/%d/%Y', errors='raise')
        .isna().sum().any()})

    # Create Exception if arguments have invalid values
    except ValueError as val_exc:
        ERRORS.update({'date_format': True})
        logging.debug("check_empty_data: %s",val_exc)

    # Create Exception if file is not found
    except FileNotFoundError as file_exc:
        ERRORS.update({'date_format': True})
        logging.debug("check_empty_data: %s",file_exc)

# %%
def main():
    '''
    Function calling all the six quality checks
    '''

    # Creating argument parser to load file
    parser = argparse.ArgumentParser(description='Kindly load an input file')
    parser.add_argument('Path', metavar='path', type=str, nargs='+',
                        help='file path to CSV file')

    args = parser.parse_args()
    if len(args.Path) > 1:
        data = import_data(args.Path[1])
    else:
        data = import_data(args.Path[0])

    # Calling respective quality check functions
    if data is not None:
        check_empty_data(data)
        check_duplicate_data(data, 'guid')
        check_account_data(data, 'account_type', ['facebook','google','other'])
        check_age_data(data, 'age')
        check_date_format(data, 'birthday', 'signup_date')

    # Storing the final check if error exists
    result = all(not result for result in ERRORS.values())
    return result


# %%
if __name__ == "__main__":

    # Initialise error results for six quality checks
    ERRORS = {'import': True,
            'empty': True,
            'duplicate': True,
            'account_type': True,
            'age_limit': True,
            'date_format': True}

    validation = main()

    # Log validation checks
    logging.info("Validation:%s\n %s",validation,ERRORS)

    if validation:
        print("Validation Passed")
    else:
        print("Validation Failed")

# %%