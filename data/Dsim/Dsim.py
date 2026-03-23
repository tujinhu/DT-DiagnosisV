import os, sys
import shutil
import RflyDtrain


def get_cmd(path):
    with open(path, 'r') as file:
        content = file.read()
        cmd = content.split(';')
        return cmd


if __name__ == "__main__":
    
    # 1-hover 2-forward 3-acc-x-axis-flag3-3000 5-motor03-flag4-085 6-motor03-flag3-200 7-maneuverability
    fmode = '3-acc-x-axis-flag3-3000-normal-r2vo'

    ulg_dir = sys.path[0] + '/' + fmode + '/ulg'  
    data_dir = sys.path[0] + '/' + fmode + '/data'
    cmd_dir = sys.path[0] + '/' + fmode + '/cmd.txt'
    os.makedirs(data_dir, exist_ok=True)  

    ulg_files = [f for f in os.listdir(ulg_dir) if f.endswith('.ulg')]

    for index, ulg_file in enumerate(ulg_files, start=1):
        index_folder = os.path.join(data_dir, str(index), 'mav1')
        log_folder = os.path.join(index_folder, 'log')
        ntrain_folder = os.path.join(index_folder, 'ntrain data')
        
        os.makedirs(log_folder, exist_ok=True)
        os.makedirs(ntrain_folder, exist_ok=True)
        
        ulg_src = os.path.join(ulg_dir, ulg_file)
        ulg_dest = os.path.join(log_folder, ulg_file)
        shutil.copy2(ulg_src, ulg_dest)
        os.chdir(log_folder)
        cmd = f"ulog2csv {ulg_file}"
        os.system(cmd)
        print(f'Processed {ulg_file}, saved to {ulg_dest}')

        Dtrain = RflyDtrain.RflyDtrain()
        ctrlSeq = get_cmd(cmd_dir)
        Dtrain.get_normal_train_data(log_folder, ntrain_folder, ctrlSeq, fmode)


    print("All ULG files have been processed.")