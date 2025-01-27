import os,sys
import xarray as xr
from dask.diagnostics import ProgressBar
import cdsapi
import time  # For handling retries
import multiprocessing

# Parameters
START_YEAR = 1980
END_YEAR = 2021
# Define the specific area
GLOBAL_AREA = [90, -180, -90, 180]  # [north, west, south, east]

# BASE_PATH specifies the base path where the data is downloaded and processed
# it defaults to the directory where the Python program is located
# NOTE:The data is actually currently stored at 
#    BASE_PATH/processed_data
# to avoid cluttering the base directory.
BASE_PATH = os.path.dirname(sys.argv[0])

# multitasking params
# NUMBER_OF_SLOTS: 
# SLOTS: the number of tasks to run in parallel
# MYSLOT: the task slot that THIS task is running in.
# The value for MYSLOT is passed in on the command line by the
# startruns.bat batch file.
# You must ensure that the number of slots assigned in the startruns.bat 
# matches the number assigned to SLOTS below
NUMBER_OF_SLOTS=8

# internal working code
# default is one slot (serial processing)
SLOTS=1
MYSLOT=0
# if a slot number was passed on the command line, then
# assign the NUMBER_OF_SLOTS value
if len(sys.argv) > 1:
    SLOTS=NUMBER_OF_SLOTS
    MYSLOT=sys.argv[1]

# special case. if a question mark (?) is specified on the
# command line, a special test mode is invoked.
if MYSLOT == "?":
    print(f"Number of slots = {SLOTS}")


'''
# Initialize the CDS API client
client = cdsapi.Client()
'''


def download_hourly_data(year, month, day, area, output_file, max_retries=3):
    """
    Downloads hourly temperature data for a single day using the CDS API.
    Implements a retry mechanism in case of internet failure.
    """
    dataset = "reanalysis-era5-pressure-levels"
    for attempt in range(max_retries):
        try:
            request = {
                "product_type": "reanalysis",
                "variable": ["temperature"],  # Temperature variable
                "pressure_level": [
                    "1", "2", "3", "5", "7", "10", "20", "30", "50",
                    "70", "100", "125", "150", "175", "200", "225",
                    "250", "300", "350", "400", "450", "500", "550",
                    "600", "650", "700", "750", "775", "800", "825",
                    "850", "875", "900", "925", "950", "975", "1000",
                ],
                "year": str(year),
                "month": f"{month:02d}",
                "day": f"{day:02d}",
                "time": [f"{hour:02d}:00" for hour in range(24)],
                "format": "netcdf",
                "grid": [0.5, 0.5],
                "area": area,  # Specific area
            }
            print(f"Submitting request for hourly data: {year}-{month:02d}-{day:02d} (Attempt {attempt + 1})")
            client = cdsapi.Client()
            client.retrieve(dataset, request, output_file)
            print(f"Downloaded hourly data to: {output_file}")
            return True  # Download successful
        except Exception as e:
            print(f"Failed to download data for {year}-{month:02d}-{day:02d} on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(5)  # Wait before retrying
            else:
                print("Max retries reached. Skipping this file.")
                return False  # Indicate failure

def process_daily_mean(hourly_file, daily_mean_file):
    """
    Processes hourly temperature data to compute the daily mean.
    Deletes the hourly file after processing to save space.
    """
    try:
        print(f"Processing {hourly_file}...")
        
        # Load the dataset
        ds = xr.open_dataset(hourly_file)

        # Check for the correct time dimension
        time_dim = None
        if 'valid_time' in ds.dims:
            time_dim = 'valid_time'
        elif 'time' in ds.dims:
            time_dim = 'time'
        else:
            raise ValueError(f"Time dimension not found in {hourly_file}. Available dimensions: {list(ds.dims)}")

        # Calculate the daily mean across the detected time dimension
        daily_mean = ds.mean(dim=time_dim)

        # Save the daily mean to a new NetCDF file
        with ProgressBar():
            daily_mean.to_netcdf(daily_mean_file+"_tmp", compute=True)
        os.rename(daily_mean_file+"_tmp", daily_mean_file)
        print(f"Saved daily mean to: {daily_mean_file}")
    except Exception as e:
        print(f"Error processing {hourly_file}: {e}")
    finally:
        # Ensure resources are freed
        if 'ds' in locals():
            ds.close()

        # Delete the hourly file to save space
        if os.path.exists(hourly_file):
            os.remove(hourly_file)
            print(f"Deleted hourly file: {hourly_file}")


def daily_mean_file_fn(task):
    year,month,day,area,output_dir,slot_no = task
    return os.path.join(output_dir, f"daily_mean_{year}-{month:02d}-{day:02d}.nc")
    

def download_and_process_day(task):

    year,month,day,area,output_dir,slot_no = task


    hourly_file = os.path.join(output_dir, f"hourly_{year}-{month:02d}-{day:02d}.nc")
    daily_mean_file = daily_mean_file_fn(task)

    if os.path.exists(daily_mean_file):
        print(f"Daily mean for {year}-{month:02d}-{day:02d} already exists. Skipping...")
        return

    download_success = download_hourly_data(year, month, day, area, hourly_file)
    if not download_success:
        print(f"Skipping daily mean calculation for {year}-{month:02d}-{day:02d} due to download failure.")
        return

    process_daily_mean(hourly_file, daily_mean_file)

def process_full_period(start_year, end_year, area, base_path):
    global GLOBAL_AREA
    """
    Processes data for the entire period (1980–2021).
    Downloads hourly data, calculates daily means, and deletes hourly files.
    """
    output_dir = os.path.join(base_path, "processed_data")
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    slot_no = 0
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):  # Loop through all months
            days_in_month = 31
            if month == 2:  # Handle February
                days_in_month = 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
            elif month in [4, 6, 9, 11]:  # April, June, September, November
                days_in_month = 30

            for day in range(1, days_in_month + 1):
                task = [year,month,day,area,output_dir,slot_no]
                # only add the day to the task list
                # if the day's daily mean file does not exist
                if not os.path.exists(daily_mean_file_fn(task)):
                    tasks = tasks + [task]
                    slot_no = (slot_no + 1) % SLOTS 
                    # if we are testing, quit after creating the first task
                    if MYSLOT=="?":
                        break
            # nested loop, so must check for testing mode at each loop level
            if MYSLOT=="?":
                break
        if MYSLOT=="?":
            break


    # initial stab at using the python multiprocessing module. It may work 
    # now that I moved the creation of the CDS API client such that
    # a new client object is created for each day separately
    # instead of reusing the same client object for all days.
    '''with multiprocessing.Pool(processes = SLOTS) as pool:
        results = pool.map(download_and_process_day,tasks)
    '''

    #  if we are in test mode, 
    if MYSLOT=="?":
        print("===== TESTING CONFIGURATION ON FIRST DAY ONLY IN LIMTED AREA (Milwaukee area) ======")
        GLOBAL_AREA = [43, -88, 42, -87]  # [north, west, south, east]
        download_and_process_day(tasks[0])
    else:
        for task in tasks:
            print(f"====== task: {','.join([str(i) for i in task])} MYSLOT = {MYSLOT} =======")
            if int(task[-1])  == int(MYSLOT):
                download_and_process_day(task)


# Run the processing for the specific area
process_full_period(START_YEAR, END_YEAR, GLOBAL_AREA, BASE_PATH)
