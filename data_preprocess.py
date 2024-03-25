import pandas as pd
import os
import matplotlib.pyplot as plt
import logging
import time
import seaborn as sns
import sys
import json
import itertools
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_saved_df(load_dir):
    load_path = os.path.join(load_dir, 'filtered_androzoo_data.pkl')
    if os.path.exists(load_path):
        df = pd.read_pickle(load_path)
        logging.info(f'Loaded DataFrame from {load_path}.')
        return df
    else:
        logging.error(f'No saved DataFrame found at {load_path}.')
        return None

def add_months_to_date_and_get_families(df, current_date, number_of_months=4):
    mask = (df['year_month'] >= str(current_date)) & (df['year_month'] < str(current_date + np.timedelta64(number_of_months, 'M')))
    size_families = df.loc[mask].groupby("family").size()
    size_families_dict = dict(size_families)
    #size_families_sorted = sorted(list(size_families_dict.values()), reverse=True)
    sorted_size_families_dict = dict(sorted(size_families_dict.items(), key=lambda x:x[1], reverse=True))

    return sorted_size_families_dict

def filter_old_families(sorted_size_families_dict, families_all):
    for k in list(sorted_size_families_dict):
        if k in families_all:
            del sorted_size_families_dict[k]
    return sorted_size_families_dict

def get_split_dates(df, min_nbr_samples_for_new_family, min_nbr_fam_in_new_step, min_nbr_months_in_new_step):
    
    split_dates = []

    min_date = np.datetime64(df['year_month'].min(), 'M')
    max_date = np.datetime64(df['year_month'].max(), 'M')

    df['year_month'] = df['year_month'].dt.to_timestamp('s').dt.strftime('%Y-%m')

    new_families = []
    families_all = set()

    current_date = min_date
    split_dates.append(current_date)

    count_steps = 0

    while current_date <= max_date:

        #we first add 4 months for every new step. Then gradually add 1 month if nbr of samples is not enough
        extra_months = 0
        size_new_families = []

        #the 1st condition is to ensure that the loop is not infinite in the last date
        #the last condition is when we have min_nbr_fam_in_new_step but these families don't have min_nbr_samples_for_new_family
        while ((current_date+(min_nbr_months_in_new_step+extra_months-1) <= max_date) and 
               (len(size_new_families)<min_nbr_fam_in_new_step or 
                (len(size_new_families)>=min_nbr_fam_in_new_step and 
                 size_new_families[min_nbr_fam_in_new_step-1]<min_nbr_samples_for_new_family))):

            #we add one extra month each time until the above cconditions are met
            sorted_size_families_dict = add_months_to_date_and_get_families(df, current_date, min_nbr_months_in_new_step+extra_months)
            sorted_size_families_dict = filter_old_families(sorted_size_families_dict, list(families_all))
            size_new_families = list(sorted_size_families_dict.values())
            extra_months+=1

            print("extra_months added", extra_months-1, current_date+np.timedelta64(min_nbr_months_in_new_step+extra_months-1, 'M'))
            
        families_in_current_step = []

        if (current_date+(min_nbr_months_in_new_step+extra_months-1) <= max_date):#all the steps except the last one
            for k, v in sorted_size_families_dict.items():
                if v>=min_nbr_samples_for_new_family: #to add only new families and not the unknown
                    families_in_current_step.append(k)

            new_families.append(families_in_current_step)
            families_all.update(families_in_current_step)

        else: #in the last step, all the new families are added without checking if they have enough samples
            new_families.append(list(sorted_size_families_dict.keys()))
            families_all.update(list(sorted_size_families_dict.keys()))

        #update the current date with the new added months
        current_date = current_date + np.timedelta64(min_nbr_months_in_new_step+(extra_months-1), 'M') 

        #if the last step doesn't contain enough new samples, it is merged with the previous one
        if (len(size_new_families)<min_nbr_fam_in_new_step or 
            (len(size_new_families)>=min_nbr_fam_in_new_step and 
             size_new_families[min_nbr_fam_in_new_step-1]<min_nbr_samples_for_new_family)):
            
            split_dates[-1] = current_date

        else:  
            split_dates.append(current_date)

        print(f"----------------------- Step {count_steps}", current_date, size_new_families[:min_nbr_fam_in_new_step])
        count_steps+=1

    print(split_dates)
    return split_dates, new_families

def save_splits_to_file(steps_family_samples, data, keyword='train'):
    
    if not os.path.exists(f'../data_info/{data}'):
        os.makedirs(f'../data_info/{data}')
        
    with open(f'../data_info/{data}/steps_family_samples_{keyword}.json', "w") as file:
        json.dump(steps_family_samples, file, indent = 3)

    #with open(f'../data_info/whole/steps_samples_{train_or_test}.json', "w") as file:
    #    json.dump(steps_samples, file, indent = 3)

def construct_split_files(df, split_dates, test_samples_from_old_fam_in_current_step, 
                          min_nbr_samples_for_new_family, new_families, data, valid=False):
    steps_family_samples_train = {} #format={"step0": {"fam1": [img1, img2, ...], ...}, ...}
    steps_family_samples_valid = {}
    steps_family_samples_test = {}

    step_count=0
    min_date_for_step = split_dates[0]

    for current_date in split_dates[1:]:
        mask = (df['year_month'] >= str(min_date_for_step)) & (df['year_month'] < str(current_date))
        sorted_dict = df.loc[mask].sort_values(by='first_seen_year').groupby('family')['sha256'].apply(list).to_dict()

        steps_family_samples_train[f'step={step_count}'] = {}
        steps_family_samples_valid[f'step={step_count}'] = {}
        steps_family_samples_test[f'step={step_count}'] = {}

        for k, v in sorted_dict.items():
            if test_samples_from_old_fam_in_current_step:
                if len(v)>=min_nbr_samples_for_new_family: #If a family has enough samples, we split to train/test no matter if old or new
                    steps_family_samples_train[f'step={step_count}'][k] = v[:int(len(v)*0.8)] #80% in the train
                    if valid:
                        steps_family_samples_valid[f'step={step_count}'][k] = v[int(len(v)*0.8):int(len(v)*0.9)]
                        steps_family_samples_test[f'step={step_count}'][k] = v[int(len(v)*0.9):]
                    else:
                        steps_family_samples_test[f'step={step_count}'][k] = v[int(len(v)*0.8):]
                        
                else: #if a family is small (no matter if new or old) it samples go to the test
                    steps_family_samples_test[f'step={step_count}'][k] = v

            else:
                if k in new_families[step_count]: #if it is a new family in current step
                    steps_family_samples_train[f'step={step_count}'][k] = v[:int(len(v)*0.8)] #80% in the train
                    if valid:
                        steps_family_samples_valid[f'step={step_count}'][k] = v[int(len(v)*0.8):int(len(v)*0.9)]
                        steps_family_samples_test[f'step={step_count}'][k] = v[int(len(v)*0.9):]
                    else:
                        steps_family_samples_test[f'step={step_count}'][k] = v[int(len(v)*0.8):]

                else: #we want to test only on new families. All the samples from old fam will go to training
                    if k not in list(itertools.chain(*new_families[:step_count])): #The unknown will go to test
                        steps_family_samples_test[f'step={step_count}'][k] = v

                    if k in list(itertools.chain(*new_families[:step_count])): #if it is an old family in current step
                        steps_family_samples_train[f'step={step_count}'][k] = v

        step_count+=1
        min_date_for_step = current_date
        #print(k, len(v), steps_family_samples_train)
    save_splits_to_file(steps_family_samples_train, data, 'train')
    save_splits_to_file(steps_family_samples_test, data, 'test')
    if valid:
        save_splits_to_file(steps_family_samples_valid, data, 'valid')

def main():

    """Main function of the script."""
    data = 'without_type_whole'

    statistics_dir = f'../statistics_{data}_all_families'

    # Check if the processed DataFrame has already been saved to a CSV file
    if os.path.exists(os.path.join(statistics_dir, 'filtered_androzoo_data.pkl')):
        # If so, load the DataFrame from the saved file
        df = load_saved_df(statistics_dir)
    else:
        print('Please generate the dataframe first, using chronological_statistics.py script')
    
    # Start the clock
    start_time = time.time()

    min_nbr_samples_for_new_family = 20
    min_nbr_fam_in_new_step = 4
    min_nbr_months_in_new_step = 4
    valid=False
    
    #whether to test on samples from old fam that are present in current step
    test_samples_from_old_fam_in_current_step = True

    split_dates, new_families = get_split_dates(df, min_nbr_samples_for_new_family, min_nbr_fam_in_new_step, min_nbr_months_in_new_step)
    construct_split_files(df, split_dates, test_samples_from_old_fam_in_current_step, 
                          min_nbr_samples_for_new_family, new_families, data, valid)

    # Print the time elapsed
    print("Time elapsed: %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
