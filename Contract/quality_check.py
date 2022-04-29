import itertools
import sys
import numpy as np
import pandas as pd


def import_data(file_name):
    data = pd.read_csv(file_name)
    


data_file = 'data-clean.csv'
df = pandas.read_csv(data_file)

check_cols_for_nan = ["age", "account_type ", "signup_date"]
check_cols_for_dupes = ["guid"]


errors = []

# Check for NAs in age
try:
    for row in df.iterrows():
        for column in check_cols_for_nan:
            print(row[column])
            if row[column] == np.nan:
                raise TypeError("Row found with NaN.")    
    errors.append(False)
except Exception as exec:    
    print(f"NA:{exec}")
    errors.append(True)

# Check account types of type ["google", "facebook", "other"]
try:
    actual = sorted(set(df.account_type.values))
    expected = sorted(set(["google", "facebook", "other"]))
    assert set(actual).issubset(expected)
    errors.append(False)
except Exception as exec:
    print("Account")
    errors.append(True)


try:
    for column in check_cols_for_dupes:
        # Check for duplicates
        ids = df[column]
        duplicated = df[ids.isin(ids[ids.duplicated()])].sort_values("guid")
        if len(duplicated):
            raise ValueError("Duplciated rows found")
    errors.append(False)
except Exception as exec:
    print("Duplicate")
    errors.append(True )



if errors:    
    num_errors = [x for x in errors if x==True]
    print("Validation failed - {} errors found".format(len(num_errors)))
else:
    print("Validation complete.")