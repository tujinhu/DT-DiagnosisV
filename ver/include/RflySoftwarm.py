
import os, sys, re
import time
import threading
import subprocess
import numpy as np
import psutil



class REG:
    bat_path = os.path.join(sys.path[0], 'SITLRun.bat')
    barrier = None
    softtool_wait_time = 45 

    
class RflySW:
    def __init__(self,num):
        self.BVNum(num)
        REG.barrier = threading.Barrier(num)
    
    def BVNum(self, num):
        newdata = ''
        with open(REG.bat_path, mode='r', encoding='UTF-8') as fline:
            for line in fline:
                if  line.find('SET /A VehicleNum=')!= -1:
                    line = line.replace(line,'SET /A VehicleNum={} \n'.format(num))
                newdata += line

        with open(REG.bat_path,mode='w',encoding='UTF-8') as f:
            f.write(newdata)

    def Start(self):  
        path = REG.bat_path
        self.child = subprocess.Popen(path,shell=True,stdout=subprocess.PIPE)
        print(f'Starting simulation software')
        time.sleep(REG.softtool_wait_time)

    def End(self):
        print(f'Exit simulation software')
        os.system('tasklist|find /i "CopterSim.exe" && taskkill /im "CopterSim.exe"')
        os.system('tasklist|find /i "QGroundControl.exe" && taskkill /f /im "QGroundControl.exe"')
        os.system('tasklist|find /i "RflySim3D.exe" && taskkill /f /im "RflySim3D.exe"')

        self.kill_process_and_children(self.child.pid)
        # self.child.terminate()
        # self.child.kill()
        print('All closed')
        time.sleep(5)
    
    def kill_process_and_children(self, proc_pid):
        try:
            process = psutil.Process(proc_pid)
            for proc in process.children(recursive=True):
                proc.kill()
            process.kill()
        except psutil.NoSuchProcess:
            pass
    
