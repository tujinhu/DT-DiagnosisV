import os, sys
import shutil
import re
import numpy as np
import pandas as pd


def interpolate_data(smaller_data_np, larger_data_np):
    smaller_length = len(smaller_data_np)
    larger_length = len(larger_data_np)
    
    # Interpolation indices
    x_smaller = np.arange(smaller_length)
    x_larger = np.linspace(0, smaller_length - 1, larger_length)
    
    # Prepare an array to hold interpolated data
    interpolated_data_np = np.zeros_like(larger_data_np, dtype=float)
    
    # Perform interpolation for each column
    for col in range(smaller_data_np.shape[1]):
        y_smaller = smaller_data_np[:, col]
        
        # Use numpy's interpolation
        interpolated_column = np.interp(x_larger, x_smaller, y_smaller)
        interpolated_data_np[:, col] = interpolated_column
    
    return interpolated_data_np


if __name__ == "__main__":

    'D:\Tjh\Platform\DT\data\Dsim\2-forward\data'
    # 1-hover 2-forward 3-acc-x-axis-flag3-3000 5-motor03-flag4-085 6-motor03-flag3-200
    fault = False
    v2v = True
    r2vo = False
    
    fmode = '6-motor03-flag3-200'
    phase = ['2_3', '2_6','2_9']
    source = 'Draw'
    target = 'Dsim'

    if fault:
        sim_fmode = fmode + '-normal'

    if v2v:
        sim_fmode = fmode + '-normal-v2v'
        dest_name = sim_fmode 
    else:
        dest_name = sim_fmode

    if r2vo:
        sim_fmode = fmode + '-normal-r2vo'
        dest_name = sim_fmode

    raw_data_dir = os.path.join(sys.path[0], source, fmode, 'data')
    sim_data_dir = os.path.join(sys.path[0], target, sim_fmode, 'data')
    dest_data_dir = os.path.join(sys.path[0], 'Dpro', dest_name)

    files_to_extract = [f"sync_merge_data_mode{p}.csv" for p in phase]

    columns_to_read = [
        'gyro_rad[0]', 'gyro_rad[1]', 'gyro_rad[2]', 
        'accelerometer_m_s2[0]', 'accelerometer_m_s2[1]', 'accelerometer_m_s2[2]',
        'magnetometer_ga[0]', 'magnetometer_ga[1]', 'magnetometer_ga[2]'
    ]

    data_files = [f for f in os.listdir(sim_data_dir)]

    all_data = {}
    for mode in files_to_extract:
        match = re.search(r'data_(.*)\.csv', mode)
        mo = match.group(1)
        all_data[f"all_raw_data_{mo}"] = pd.DataFrame(columns=columns_to_read)
        all_data[f"all_sim_data_{mo}"] = pd.DataFrame(columns=columns_to_read)
        all_data[f"all_sycn_raw_data_{mo}"] = pd.DataFrame(columns=columns_to_read)
        all_data[f"all_sycn_sim_data_{mo}"] = pd.DataFrame(columns=columns_to_read)
        all_data[f"all_sycn_err_data_{mo}"] = pd.DataFrame(columns=columns_to_read)
    

    for index, data_file in enumerate(data_files, start=1):
        raw_data_path = os.path.join(raw_data_dir, data_file, 'mav1', 'ntrain data')
        sim_data_path = os.path.join(sim_data_dir, data_file, 'mav1', 'ntrain data')

        raw_matching_files = []
        sim_matching_files = []
        for mode in files_to_extract:
            raw_matching = [f for f in os.listdir(raw_data_path) if f.endswith(mode)]
            raw_matching_files.append(raw_matching[0])
            sim_matching = [f for f in os.listdir(sim_data_path) if f.endswith(mode)]
            sim_matching_files.append(sim_matching[0])
        
        for mode in raw_matching_files:
            match = re.search(r'data_(.*)\.csv', mode)
            mo = match.group(1)
            dest_data_mode_dir = os.path.join(dest_data_dir, mo, data_file)
            os.makedirs(dest_data_mode_dir, exist_ok=True)

            raw_data_mode_src = os.path.join(raw_data_path, mode)
            raw_data_mode_dest = os.path.join(dest_data_mode_dir, 'raw_' + mode)
            sim_data_mode_src = os.path.join(sim_data_path, mode)
            sim_data_mode_dest = os.path.join(dest_data_mode_dir, 'sim_' + mode)
            shutil.copy2(raw_data_mode_src, raw_data_mode_dest)
            shutil.copy2(sim_data_mode_src, sim_data_mode_dest)
            print(f'Copyed (raw_data_mode,sim_data_mode), saved to {dest_data_mode_dir}')

            raw_data = pd.read_csv(raw_data_mode_dest)
            sim_data = pd.read_csv(sim_data_mode_dest)
            raw_data_np = raw_data.to_numpy()
            sim_data_np = sim_data.to_numpy()
            
            if len(raw_data_np) < len(sim_data_np):
                raw_data_np = interpolate_data(raw_data_np, sim_data_np)
            else:
                sim_data_np = interpolate_data(sim_data_np, raw_data_np)
            
            error = sim_data_np - raw_data_np
            # error = wavelet_denoising(sim_data_np) - wavelet_denoising(raw_data_np)
            raw_sycn_df = pd.DataFrame(raw_data_np, columns=raw_data.columns)
            sim_sycn_df = pd.DataFrame(sim_data_np, columns=sim_data.columns)
            err_sycn_df = pd.DataFrame(error, columns=raw_data.columns)

            raw_sycn_df_path = os.path.join(dest_data_mode_dir, 'raw_sycn_dt.csv')
            sim_sycn_df_path = os.path.join(dest_data_mode_dir, 'sim_sycn_dt.csv')
            err_sycn_df_path = os.path.join(dest_data_mode_dir, 'err_sycn_dt.csv')
            raw_sycn_df.to_csv(raw_sycn_df_path, index=False)
            sim_sycn_df.to_csv(sim_sycn_df_path, index=False)
            err_sycn_df.to_csv(err_sycn_df_path, index=False)
            print(f'Processed (raw_sycn_df,sim_sycn_df,err_sycn_df), saved to {dest_data_mode_dir}')

            all_data[f"all_raw_data_{mo}"] = pd.concat([all_data[f"all_raw_data_{mo}"], raw_data], ignore_index=True)
            all_data[f"all_sim_data_{mo}"] = pd.concat([all_data[f"all_sim_data_{mo}"], sim_data], ignore_index=True)
            all_data[f"all_sycn_raw_data_{mo}"] = pd.concat([all_data[f"all_sycn_raw_data_{mo}"], raw_sycn_df], ignore_index=True)
            all_data[f"all_sycn_sim_data_{mo}"] = pd.concat([all_data[f"all_sycn_sim_data_{mo}"], sim_sycn_df], ignore_index=True)
            all_data[f"all_sycn_err_data_{mo}"] = pd.concat([all_data[f"all_sycn_err_data_{mo}"], err_sycn_df], ignore_index=True)
    

    output_base_path = dest_data_dir 
    for key, df in all_data.items():
        file_path = os.path.join(output_base_path, f"{key}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved {key} to {file_path}")
            

        

        
