import pandas as pd
import os
import matplotlib.pyplot as plt
import logging
import time
import seaborn as sns
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_hash_and_family(samp_file_path):
    """Load hash and family information from the file 'train_file_paths.txt'."""
    hash_to_family = {}
    hash_to_type = {}

    with open(samp_file_path, 'r') as f:
        for line in f:
            split_line = line.strip().split('/')
            if len(split_line) == 4:  # New format
                _, type_, family, hash_ = split_line
                hash_ = hash_[:-4]
            else:  # Old format
                type_, family, hash_ = split_line
            if line.startswith('benign/'):
                continue
            #hash_to_family[hash_] = type_+'/'+family
            hash_to_family[hash_] = family
            hash_to_type[hash_] = type_
    logging.info('Loaded hash and family information.')

    return hash_to_family, hash_to_type


def process_df(filtered_df, hash_to_family):
    """Add a family column, convert 'first_seen_year' to datetime format and add a year-month column."""
    filtered_df['family'] = filtered_df['sha256'].map(hash_to_family)
    filtered_df['first_seen_year'] = pd.to_datetime(filtered_df['first_seen_year'])
    filtered_df['year_month'] = filtered_df['first_seen_year'].dt.to_period('M')
    return filtered_df

def save_df(df, save_dir):
    save_path = os.path.join(save_dir, 'filtered_androzoo_data.pkl')
    df.to_pickle(save_path)
    logging.info(f'Saved processed DataFrame to {save_path}.')

def load_saved_df(load_dir):
    load_path = os.path.join(load_dir, 'filtered_androzoo_data.pkl')
    if os.path.exists(load_path):
        df = pd.read_pickle(load_path)
        logging.info(f'Loaded DataFrame from {load_path}.')
        return df
    else:
        logging.error(f'No saved DataFrame found at {load_path}.')
        return None

def analyze_and_plot(filtered_df, statistics_dir, plot_switch):
    print(filtered_df)
    """Perform the data analysis and save the plots."""
    if plot_switch['new families per year and per month']:
        # Analysis 1: New families per year and per month
        new_families_per_year = filtered_df.groupby('family')['year_month'].min().dt.year.value_counts().sort_index()
        new_families_per_month = filtered_df.groupby('family')['year_month'].min().value_counts().sort_index()

        if new_families_per_year.empty or new_families_per_month.empty:
            logging.warning("No new families found for the given time period.")
            return

        new_families_per_year.plot(kind='bar')
        plt.title('New malware families per year')
        plt.savefig(os.path.join(statistics_dir, 'new_families_per_year.png'))
        plt.clf()

        new_families_per_month.plot()
        plt.title('New malware families per month')
        plt.savefig(os.path.join(statistics_dir, 'new_families_per_month.png'))
        plt.clf()
        logging.info('Saved plots for new families per year and per month.')

    if plot_switch['number of samples per family']:
        import matplotlib.ticker as ticker
        # Analysis 2: Number of samples in each family and their first appearance
        family_counts = filtered_df['family'].value_counts()
        first_appearance = filtered_df.groupby('family')['year_month'].min()

        family_stats = pd.DataFrame({'sample_count': family_counts, 'first_appearance': first_appearance}).sort_values('first_appearance')

        fig, ax = plt.subplots() 
        family_stats[['sample_count']].plot(kind='bar', ax=ax)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        plt.title('Number of samples per family')
        plt.savefig(os.path.join(statistics_dir, 'samples_per_family.png'))
        plt.clf()
        logging.info('Saved plot for number of samples per family.')

    if plot_switch['life length of each malware family']:
        # Analysis 3: Life length of each family
        family_life_length = filtered_df.groupby('family')['year_month'].apply(lambda x: (x.max() - x.min()).n)

        fig, ax = plt.subplots()
        family_life_length.plot(kind='bar', ax=ax)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        plt.title('Life length of each malware family')
        plt.savefig(os.path.join(statistics_dir, 'life_length_per_family.png'))
        plt.clf()
        logging.info('Saved plot for life length of each malware family.')

    if plot_switch['life_length_strip_of_each_malware_family']:
        # Analysis 4: Life length strip of each malware family
        family_life_start = filtered_df.groupby('family')['year_month'].min()
        family_life_end = filtered_df.groupby('family')['year_month'].max()

        # Convert Period to datetime
        family_life_start = family_life_start.dt.to_timestamp()
        family_life_end = family_life_end.dt.to_timestamp()

        # Get the top 20 families by sample size
        top_families = filtered_df['family'].value_counts().nlargest(20).index.tolist()

        # Sort families by their start dates but only include the top 20 families
        families = family_life_start[top_families].sort_values().index.tolist()

        # Create a figure and axes
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot a horizontal line for each family with varying colors
        for i, family in enumerate(families):
            ax.hlines(i, family_life_start[family], family_life_end[family], colors=sns.color_palette("husl", 20)[i], lw=4)
            
            # Annotate each line with start and end dates
            ax.annotate(pd.to_datetime(family_life_start[family]).strftime('%Y-%m'), (family_life_start[family], i), textcoords="offset points", xytext=(0,10), ha='center')
            ax.annotate(pd.to_datetime(family_life_end[family]).strftime('%Y-%m'), (family_life_end[family], i), textcoords="offset points", xytext=(0,-15), ha='center')

        # Set the y-axis ticks and labels
        ax.set_yticks(range(len(families)))
        ax.set_yticklabels(families, fontsize=12)

        # Set labels and title
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Family', fontsize=14)
        ax.set_title('Life Length of Top 20 Malware Families', fontsize=16)

        # Adjust layout to ensure everything fits
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(statistics_dir, 'life_length_strip_per_top_family.png'), format='png', dpi=300)
        plt.clf()
        logging.info('Saved improved plot for life length strip of top malware families.')


    if plot_switch['life length heatmap of each malware family']:
        # Analysis 5: Life length heatmap of each malware family
        # Create a sorted list of family names based on first appearance
        sorted_families = family_life_start.sort_values().index.tolist()

        # Pivot the DataFrame to create a grid with families on the y-axis and time on the x-axis
        heatmap_df = filtered_df.pivot_table(index='family', columns='year_month', aggfunc='size', fill_value=0)

        # Reindex the heatmap DataFrame using the sorted list of families
        heatmap_df = heatmap_df.reindex(sorted_families)

        # Plot the heatmap
        plt.figure(figsize=(12, 20))  # Adjust the size of the figure as needed
        sns.heatmap(heatmap_df, cmap='viridis', cbar=False)  # Remove the color bar
        
        # Set labels and title
        plt.xlabel('Year and Month')
        plt.ylabel('Family')
        plt.title('Life length of each malware family')

        # Save the plot
        plt.savefig(os.path.join(statistics_dir, 'life_length_heatmap_per_family.png'))
        plt.clf()
        logging.info('Saved plot for life length heatmap of each malware family.')

    if plot_switch['new_families_emerge_every_6_months']:
        # Analysis 6: New families emerge every 6 months
        new_families_per_half_year = (filtered_df.groupby('family')['year_month']
                                    .min()
                                    .dt.to_timestamp()  # Convert Period to datetime
                                    .reset_index()
                                    .set_index('year_month')
                                    .resample('6M')
                                    .count())

        # Convert datetime back to PeriodIndex with semiannual frequency
        new_families_per_half_year.index = new_families_per_half_year.index.to_period('6M')

        # Convert to DataFrame for better handling in Seaborn
        new_families_df = pd.DataFrame({'6-month period': new_families_per_half_year.index.astype(str),
                                        'Number of new families': new_families_per_half_year['family'].values})

        # Create a figure and axes using Seaborn
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))

        # Create bar plot with a darker color palette
        ax = sns.barplot(x='6-month period', y='Number of new families', data=new_families_df, palette="mako")

        # Rotate x-labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=12)

        # Set labels and title
        ax.set_xlabel('6-Month Period', fontsize=14)
        ax.set_ylabel('Number of New Families', fontsize=14)
        ax.set_title('Number of New Malware Families Emerging Every 6 Months', fontsize=16)

        # Annotate each bar for better information
        for p in ax.patches:
            ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)

        # Adjust layout to make sure everything fits
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(statistics_dir, 'new_families_per_half_year.png'), format='png', dpi=300)
        plt.clf()
        logging.info('Saved improved plot for number of new families per 6 months.')



def load_and_filter_df(hash_to_family, families_to_consider):
    """Load the DataFrame from the gzipped CSV, and filter it based on the hash list and family list."""
    df = pd.read_csv('DATASET/full_date.csv.gz')  # Download from: https://androzoo.uni.lu/static/lists/full_date.csv.gz
    logging.info('Loaded CSV data.')
    
    # First filter based on hash list
    filtered_df = df[df['sha256'].isin(hash_to_family.keys())]
    logging.info(f"Filtered DataFrame based on hash list. Remaining samples: {filtered_df.shape[0]}")
    
    # Add a temporary 'family' column for the second filtering
    filtered_df['family'] = filtered_df['sha256'].map(hash_to_family)
    
    # Second filter based on families to consider
    if len(families_to_consider):
        filtered_df = filtered_df[filtered_df['family'].isin(families_to_consider)]
    logging.info(f"Filtered DataFrame based on family list. Remaining samples: {filtered_df.shape[0]}")
    
    return filtered_df

def main():
    """Main function of the script."""
    
    data = 'whole'
    
    statistics_dir = f'statistics_without_type_{data}_all_families_0831'

    plot_switch = {"new families per year and per month": False,
                   "number of samples per family": False,
                   "life length of each malware family": False,
                   "life_length_strip_of_each_malware_family": True,
                   "life length heatmap of each malware family": False,
                   "new_families_emerge_every_6_months": True
                   }
    
    # List of families to consider
    families_to_consider = []
    

    # Start the clock
    start_time = time.time()

    # Ensure the statistics directory exists
    if not os.path.exists(statistics_dir):
        os.makedirs(statistics_dir)
        logging.info(f'Created directory: {statistics_dir}')
        
    if data == 'whole':
        samp_file_path = 'DATASET/MalNet/split_info/split_info/family/1.0/full.txt'
        hash_to_family, hash_to_type = load_hash_and_family(samp_file_path)
        
    elif data == 'tiny':
        # Load your list of hashes and families
        samp_file_train_path = 'DATASET/MalNet/malnet-images-tiny/train_file_paths.txt'
        samp_file_valid_path = 'DATASET/MalNet/malnet-images-tiny/val_file_paths.txt'
        samp_file_test_path = 'DATASET/MalNet/malnet-images-tiny/test_file_paths.txt'
    
        hash_to_family_train, hash_to_type_train = load_hash_and_family(samp_file_train_path)
        hash_to_family_valid, hash_to_type_valid = load_hash_and_family(samp_file_valid_path)
        hash_to_family_test, hash_to_type_test = load_hash_and_family(samp_file_test_path)
        hash_to_family = {**hash_to_family_train, **hash_to_family_valid, **hash_to_family_test}
        hash_to_type = {**hash_to_type_train, **hash_to_type_valid, **hash_to_type_test}
        hash_to_type = {**hash_to_type_train, **hash_to_type_valid, **hash_to_type_test}
        assert(len(hash_to_family_train) + len(hash_to_family_valid) + len(hash_to_family_test) == len(hash_to_family))

    
    
    # Check if the processed DataFrame has already been saved to a CSV file
    if os.path.exists(os.path.join(statistics_dir, 'filtered_androzoo_data.pkl')):
        # If so, load the DataFrame from the saved file
        filtered_df = load_saved_df(statistics_dir)
    else:
        # If not, load and filter the DataFrame from the gzipped CSV
        filtered_df = load_and_filter_df(hash_to_family, families_to_consider)
        filtered_df = process_df(filtered_df, hash_to_family)
        save_df(filtered_df, statistics_dir)
    
    # Perform the data analysis and save the plots
    analyze_and_plot(filtered_df, statistics_dir, plot_switch)
    
    with open(os.path.join(statistics_dir, 'hash_type_dict.pkl'), 'wb') as f:
        pickle.dump(hash_to_type, f)
        


    # Print the time elapsed
    print("Time elapsed: %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
