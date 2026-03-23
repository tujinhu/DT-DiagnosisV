import os,sys
sys.path.append(os.getcwd())

import ver.include.PX4MavCtrlV4 as PX4MavCtrl
import ver.include.RflyDB as RflyDB
import ver.include.RflyCtrl as RflyCtrl
import ver.include.RflySoftwarm as RflySW
import ver.include.RflyDtrain as RflyDtrain
import re
import queue
import numpy as np
import collections
import time
import threading
import torch
import tkinter
import shutil
from sklearn.metrics import confusion_matrix
from tkinter.messagebox import *
from ver.include.NetFunc import *

class DataPool:
    def __init__(self, pool_size = 80):
        self.max_size = pool_size
        self.pool = collections.deque(maxlen=pool_size)

    def add_data(self, timestamp, data, flag=None):
        timestamp = time.time() 
        self.pool.append({'timestamp': timestamp, 'data': data, 'flag': flag})

    def is_pool_full(self):
        return len(self.pool) == self.max_size

    def __iter__(self):
        return iter(self.pool)
    

class FDMav:
    def __init__(self, mav, model, device, graph=True, Etype=False, sensor_num=6):
        self.mav = mav
        self.model = model
        self.device = device
        self.graph = graph
        self.Etype = Etype
        self.sensor_num = sensor_num

        self.lastTime = time.time()
        self.fd_hz = 20
        
        self.category = ['Normal', 'acc-x-axis-flag3-3000', 'motor03-flag3-200', 'motor03-flag4-085']
        self.label = self.get_label(mav)
        self.ori_lab = np.array([0,0,0,0])
        self.out_lab = np.array([0,0,0,0])
        print('self.label',self.label)

        self.num = 0
        self.right_cnt = 0
        self.fault_cnt = 0
        self.fault_info = []
        self.fault_log = ""
        self.fd_id = 0


    def record_accuracy(self, id, phase, max_index, info):
        if phase in self.label:
            self.num += 1
            if self.label[phase][max_index] == 1:
                self.right_cnt += 1
            else:
                self.fault_cnt += 1
                self.fault_log += f"LOG_ID [{id}], Phase [{phase}], Fault Type: {info}\n"
                finfo = f"LOG_ID [{id}], Phase [{phase}], Fault Type: {info}"
                self.fault_info.append(finfo)
    

    def get_label(self, mavs):
        ctrlSeq1 = mavs[0].ctrlSeq
        ctrlSeq2 = mavs[1].ctrlSeq
        final_result = None
        AllctrlSeq = [ctrlSeq1, ctrlSeq2]
        Allfinal_result = []
        for ctrlSeq in AllctrlSeq:
            for sub_seq in ctrlSeq:
                if sub_seq.startswith("2,9"):
                    result = sub_seq[4:].split(",")
                    seen = set()
                    filtered_result = []
                    
                    for item in result:
                        if item.isdigit() and int(item) >= 123450:
                            if item not in seen:
                                seen.add(item)
                                filtered_result.append(item)
                        else:
                            filtered_result.append(item)
                    
                    final_result = ",".join(filtered_result)
                    final_result = final_result.split(",")
                    Allfinal_result.append(final_result)
                    break
        
        if final_result == None:
            fault_label = [1,0,0,0]
        elif Allfinal_result[0] == Allfinal_result[1]:
            fault_label = [1,0,0,0]
        elif Allfinal_result[0][0] == '123450' and Allfinal_result[0][1] == '4':
            fault_label = [0,0,0,1]
        elif Allfinal_result[0][0] == '123450' and Allfinal_result[0][1] == '3':
            fault_label = [0,0,1,0]
        elif Allfinal_result[0][0] == '123544' and Allfinal_result[0][1] == '3':
            fault_label = [0,1,0,0]
            
        label = {
            '3':[1,0,0,0],
            '5':[1,0,0,0],
            '6':[1,0,0,0],
            '8':[1,0,0,0],
            '9':fault_label,
        }
        return label
    
    def ShowWin(self):
        window = tkinter.Tk()
        window.attributes("-topmost", 1) 
        window.geometry("600x600+100+100") 

        scrollbar = tkinter.Scrollbar(window)
        text_widget = tkinter.Text(window, wrap="word", yscrollcommand=scrollbar.set)
        scrollbar.config(command=text_widget.yview)

        scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        text_widget.pack(expand=True, fill='both')
        text_widget.config(state=tkinter.DISABLED)  

        self.update_window(window, text_widget)
        window.mainloop()
    
    def update_window(self, window, text_widget):
        global all_messages, FD_LOG, folder_name, log_cnt

        if not FD_LOG.empty():
            latest_result = FD_LOG.get()
            if isinstance(latest_result, str):
                all_messages += latest_result
            else:
                all_messages += f"LOG_ID [{latest_result[0]}], Phase [{latest_result[1]}], Fault Type: {latest_result[2]}\n"

            text_widget.config(state=tkinter.NORMAL)  
            text_widget.delete(1.0, tkinter.END)  
            text_widget.insert(tkinter.END, all_messages)  
            text_widget.config(state=tkinter.DISABLED)  

            text_widget.yview_moveto(1.0)

            if isinstance(latest_result, str):
                max_index = np.argmax(self.label['9'])
                file_path = os.path.join(sys.path[0], 'info')

                if not os.path.exists(file_path):
                    os.makedirs(file_path)

                path = os.path.join(file_path, f"LOG_{folder_name}_{self.category[max_index]}_{log_cnt}.txt")
                if os.path.exists(path):
                    os.remove(path)

                with open(path, "w") as file: 
                    file.write(all_messages)

                return 

        window.after(int((1.0/self.fd_hz) * 1000), self.update_window, window, text_widget) 


    def get_data_2_model(self, source_acc, source_gyro, source_mag, target_acc, target_gyro, target_mag):
        err_acc = target_acc - source_acc
        err_gyro = target_gyro - source_gyro
        err_mag = target_mag - source_mag

        if self.sensor_num == 6:
            raw_err = np.hstack((source_acc, err_acc, source_gyro, err_gyro, source_mag, err_mag))
        else:
            raw_err = np.hstack((source_acc, source_gyro, source_mag))
        return raw_err

    def get_data(self, mav_data):
        data = np.array([entry['data'] for entry in mav_data])
        return data

    def FD(self, data):
        orign = torch.tensor(data)
        self.model.eval()

        if self.graph:
            if self.sensor_num == 6:
                if self.Etype:
                    node_features, edge_index, edge_type = get_graph(orign)
                    node_features = node_features.to(self.device)
                    edge_index = edge_index.to(self.device)
                    edge_type = edge_type.to(self.device)

                    out = self.model(node_features, edge_index, edge_type).to('cuda')
                    out = out.cpu().detach().numpy()
                    self.out_lab = np.vstack((self.out_lab, out))

                    predicted_classes = np.argmax(out)

                
                else:
                    node_features, edge_index = get_graph_chain(orign)
                        
                    node_features = node_features.to(self.device)
                    edge_index = edge_index.to(self.device)

                    out = self.model(node_features, edge_index).to('cuda')
                    out = out.cpu().detach().numpy()
                    self.out_lab = np.vstack((self.out_lab, out))

                    predicted_classes = np.argmax(out)

            elif self.sensor_num == 3:
                node_features = orign
                node_features = node_features.view(-1, 3)

                num_nodes = node_features.size(0)
                edge_index = generate_chain_edge_index(num_nodes)

                node_features = node_features.to(self.device)
                edge_index = edge_index.to(self.device)   

                out = self.model(node_features, edge_index).to('cuda')
                out = out.cpu().detach().numpy()
                self.out_lab = np.vstack((self.out_lab, out))

                predicted_classes = np.argmax(out)
        
        else:
            datain = orign.to(self.device)
            out = self.model(datain).to('cuda')
            out = out.cpu().detach().numpy()
            self.out_lab = np.vstack((self.out_lab, out))

            predicted_classes = np.argmax(out)

        self.ori_lab = np.vstack((self.ori_lab, self.label[RflyCtrl.MAVREG.FD_LOG_PHASE]))        

        max_index = predicted_classes
        self.fd_id += 1
        info = [self.fd_id, RflyCtrl.MAVREG.FD_LOG_PHASE, self.category[max_index]]
        # print(f"Phase [{info[1]}], OUT [{out}],  Fault Type: {self.category[max_index]}")
        self.record_accuracy(self.fd_id, RflyCtrl.MAVREG.FD_LOG_PHASE, max_index, self.category[max_index])

        return out, info
    
    def caculate_cm(self, ori_lab, out_lab):
        train_cm = confusion_matrix(np.argmax(ori_lab, axis=1), np.argmax(out_lab, axis=1))

        TP = np.diag(train_cm)
        FP = np.sum(train_cm, axis=0) - TP
        FN = np.sum(train_cm, axis=1) - TP
        TN = np.sum(train_cm) - (TP + FP + FN)
        
        precision = np.mean(TP / (TP + FP))
        recall = np.mean(TP / (TP + FN))
        f1_score = 2 * (precision * recall) / (precision + recall)
        accuracy = np.sum(TP) / np.sum(train_cm)
        
        return TP, FP, FN, TN, precision, recall, f1_score, accuracy

    def FauluDiagnosis(self): 
        global mavs, breakflag, stop_flag, FD_LOG
        
        tShow = threading.Thread(target=self.ShowWin, args=())
        tShow.start()
        
        start_fd = True
        full = False
        while True:
            self.lastTime = self.lastTime + (1.0/self.fd_hz)
            sleepTime = self.lastTime - time.time()
            if sleepTime > 0:
                time.sleep(sleepTime)
            else:
                self.lastTime = time.time()

            if not full:
                if all(mav.mavMag.is_pool_full() for mav in mavs):
                    full = True
            
            if full:
                source_acc, source_gyro, source_mag = self.get_data(mavs[0].mavAccB), self.get_data(mavs[0].mavGyro), self.get_data(mavs[0].mavMag)
                target_acc, target_gyro, target_mag = self.get_data(mavs[1].mavAccB), self.get_data(mavs[1].mavGyro), self.get_data(mavs[1].mavMag)
                data_2_model = self.get_data_2_model(source_acc, source_gyro, source_mag, target_acc, target_gyro, target_mag)
                if RflyCtrl.MAVREG.FD_LOG_PHASE in self.label and start_fd:
                    _, info = self.FD(data_2_model)
                    # print(f"LOG_ID [{info[0]}], Phase [{info[1]}], Fault Type: {info[2]}")
                    FD_LOG.put(info)
                    
            if breakflag:
                acc_log = f"\nTrue/False Number of diagnosis results: [{self.right_cnt} / {self.fault_cnt}] \nAccuracy: {(self.right_cnt / self.num) * 100}%%\n"
                
                TP, FP, FN, TN, precision, recall, f1_score, accuracy = self.caculate_cm(self.ori_lab[1:], self.out_lab[1:])
                print("TP: {}".format(TP))
                print("FP: {}".format(FP))
                print("FN: {}".format(FN))
                print("TN: {}".format(TN))
                print("Precision: {:.4f}  Recall: {:.4f}  F1 Score: {:.4f}  Accuracy: {:.4f}".format(precision, recall, f1_score, accuracy))
                index_log = f"\nTP: {TP} \nFP: {FP} \nFN: {FN} \nTN: {TN} \nPrecision: {precision} \nRecall: {recall} \nF1 Score: {f1_score} \nAccuracy: {accuracy}"

                info = acc_log + self.fault_log + index_log
                FD_LOG.put(info)
                stop_flag = True

                break

class RflyMav:
    def __init__(self, port):
        self.ID = int((port-20100)/2+1)
        self.port = port
        self.mav = PX4MavCtrl.PX4MavCtrler(self.port)
        self.frame = 1

        self.hz = 500
        self.is_alive = False
        self.round_over = False
        self.stage = ['none','none']

        self.caseIndex = 0
        self.caseID, self.caseNum, self.caseInfo, self.ctrlSeq = self.get_cmd()
        print(f'mav{self.ID} case {self.caseID} ctrlSeq: ',self.ctrlSeq)
    
    def init_params(self):
        # Init test case
        self.lastTime = time.time()
        self.MavCmdInd = 0
        self.MavCmdNum = len(self.ctrlSeq)

        # Init control sequence class object
        self.CFID = RflyCtrl.CmdCtrl(self.mav,self.frame,self.ID) 
        self.CID1OBJ = self.CFID.CID1
        self.CID2OBJ = self.CFID.CID2

        # Init data pool which received from px4 
        self.mavPosNED = DataPool()      # Estimated Local Pos (related to takeoff position) from PX4 in NED frame
        self.mavVelNED = DataPool()      # Estimated local velocity from PX4 in NED frame
        self.mavAccB = DataPool()        # Estimated acc from PX4
        self.mavGyro = DataPool()        # Estimated Gyro from PX4
        self.mavMag = DataPool()         # Estimated Mag from PX4
        self.mavVibr = DataPool()        # Estimated vibration xyz from PX4
        self.mavAngEular = DataPool()    # Estimated Eular angles from PX4
        self.mavAngRate = DataPool()     # Estimated angular rate from PX4
        self.mavAngQuatern = DataPool()  # Estimated AngQuatern from PX4

        self.EXITFLAG = False
    
    def init_connection(self):
        self.mav.InitMavLoop(UDPMode=2)
        time.sleep(0.5)
        self.mav.InitTrueDataLoop()
        time.sleep(0.5)
        self.mav.initOffboard()
        time.sleep(0.5)
        self.mav.SendMavArm(1)
    
    def end_connection(self):
        self.mav.EndTrueDataLoop()
        time.sleep(0.5)
        self.mav.endMavLoop() 
        time.sleep(1)

    def get_cmd(self):
        db = RflyDB.RflyDB(json_path)
        caseNum = len(db.GET_CASEID()[self.ID - 1])
        caseID = db.GET_CASEID()[self.ID - 1][self.caseIndex]
        caseInfo = db.GET_CASEINFO(caseID)
        ctrlSeq = caseInfo.get('ControlSequence')
        case = re.split(';',ctrlSeq)
        cmd = np.array([])
        for i in range(len(case)):
            cmd = np.append(cmd,case[i])
        
        return caseID, caseNum, caseInfo, cmd
    
    def trigger(self,ctrlseq):
        global mission
        cmdseq = ctrlseq # '2,3,0,0,-20'
        cmdseq = re.findall(r'-?\d+\.?[0-9]*',cmdseq) # ['2', '3', '0', '0', '-20']
        cmdCID = cmdseq[0]
        if  cmdCID in self.CFID.CID:
            FID = self.CFID.FIDPro(cmdCID)
            # if has param
            if len(cmdseq) > 2:
                # get param
                param = cmdseq[2:len(cmdseq)]
                param = [float(val) for val in param]
                FID[cmdseq[1]](param)

            else:
                '''mission mode'''
                if cmdseq[1] == '8':
                    FID[cmdseq[1]](mission)
                else:
                    FID[cmdseq[1]]()
        else:
            print(f'mav{self.ID} Command input error, please re-enter')

    
    def run(self):
        # Start time and end time (unlock after startup to prevent the ground station from not starting timing)
        print(f'mav{self.ID} Sim start')
        while True: 
            if self.caseIndex >= self.caseNum:
                print(f'mav{self.ID} all case test finish!')
                break

            # step1: init_connection
            self.init_params()
            self.init_connection()
            self.is_alive = True

            # step2: run
            while True:
                # 250HZ receiving data
                self.lastTime = self.lastTime + (1.0/self.hz)
                sleepTime = self.lastTime - time.time()
                if sleepTime > 0:
                    time.sleep(sleepTime)
                else:
                    self.lastTime = time.time()
                
                # Starting receiving data at 250HZ using fault diagnoise
                mavTimestamp = self.mav.uavTimeStmp
                mavPosNED = self.mav.uavPosNED
                mavVelNED = self.mav.uavVelNED  
                mavAccB = self.mav.uavAccB   
                mavGyro = self.mav.uavGyro        
                mavMag = self.mav.uavMag   
                mavVibr = self.mav.uavVibr   
                mavAngEular = self.mav.uavAngEular   
                mavAngRate = self.mav.uavAngRate   
                mavAngQuatern = self.mav.uavAngQuatern

                self.mavPosNED.add_data(timestamp=mavTimestamp,data=mavPosNED)
                self.mavVelNED.add_data(timestamp=mavTimestamp,data=mavVelNED)
                self.mavAccB.add_data(timestamp=mavTimestamp,data=mavAccB)
                self.mavGyro.add_data(timestamp=mavTimestamp,data=mavGyro)
                self.mavMag.add_data(timestamp=mavTimestamp,data=mavMag)
                self.mavVibr.add_data(timestamp=mavTimestamp,data=mavVibr)
                self.mavAngEular.add_data(timestamp=mavTimestamp,data=mavAngEular)
                self.mavAngRate.add_data(timestamp=mavTimestamp,data=mavAngRate)
                self.mavAngQuatern.add_data(timestamp=mavTimestamp,data=mavAngQuatern)     
                # print('mavPosNED',len(self.mavPosNED.pool))         

                # Processing instruction sequence
                self.stage = self.ctrlSeq[self.MavCmdInd]
                self.trigger(self.ctrlSeq[self.MavCmdInd])
                if re.findall(r'-?\d+\.?[0-9]*',self.ctrlSeq[self.MavCmdInd])[0] == '1' and self.CID1OBJ.isDone == 1 or re.findall(r'-?\d+\.?[0-9]*',self.ctrlSeq[self.MavCmdInd])[0] == '2' and self.CID2OBJ.isDone == 1:
                    self.MavCmdInd = self.MavCmdInd + 1
                    # print(f'mav{self.ID}: Process: [{self.MavCmdInd} / {self.MavCmdNum}]')
                
                if self.MavCmdInd >= self.MavCmdNum and self.EXITFLAG == False:
                    self.EXITFLAG = True
                    print(f'mav{self.ID}: CaseID {self.caseID} test completed')
                    break
                
            self.caseIndex += 1
            self.end_connection()
            self.round_over = True

def get_data(mav, cnt='1'):
    PlatFormpath = 'C:/PX4PSP'
    # 1、List the directories under the file
    log_path = PlatFormpath + f'/Firmware/build/px4_sitl_default/instance_{mav.ID}/log'

    PlatForm_log_dirs = os.listdir(log_path) 
    log_data = PlatForm_log_dirs[len(PlatForm_log_dirs)-1]
    path = os.path.join(log_path,log_data) 
    dirs = os.listdir(path) 

    # 2、Get the latest ulg file
    ulg = dirs[len(dirs)-1]
    ulgPath = os.path.join(path,ulg) 

    # 3、Copy the ulg file to the log folder
    LogPath = os.path.join(sys.path[0], 'data', cnt, f'mav{mav.ID}', 'log')
    TargetPath_log = LogPath + '/{}'.format(ulg)
    if os.path.exists(LogPath):
        shutil.rmtree(LogPath)
        os.makedirs(LogPath)
    else:
        os.makedirs(LogPath)

    shutil.copyfile(ulgPath, TargetPath_log) 
    # print(f'Successfully download mav{self.ID} log file to {TargetPath_log} path')

    for root, dirs, files in os.walk(LogPath):
        for file in files:
            if file.endswith('.ulg'):
                current_dir = os.path.abspath(root)
                os.chdir(current_dir)
                cmd = f"ulog2csv {file}"
                os.system(cmd)

    ntrain_path = os.path.join(sys.path[0], 'data', cnt, f'mav{mav.ID}', 'ntrain data')
    if os.path.exists(ntrain_path):
        shutil.rmtree(ntrain_path)
        os.makedirs(ntrain_path)
    else:
        os.makedirs(ntrain_path)
    

    Dtrain = RflyDtrain.RflyDtrain()
    Dtrain.get_normal_train_data(LogPath, ntrain_path, mav.ctrlSeq)

def generate_faulty_mission(oscillate_axis='x', amplitude=1, frequency=1, total_time=10, time_step=0.1):

    time_points = np.arange(0, total_time, time_step)
    mission = []
    vx, vy, vz, yaw = 0, 0, 0, 0
    oscillation = amplitude * np.sign(np.sin(2 * np.pi * frequency * time_points))
    
    for osc in oscillation:
        if oscillate_axis == 'x':
            vx = osc
        elif oscillate_axis == 'y':
            vy = osc
        elif oscillate_axis == 'z':
            vz = osc
        
        mission.append([vx, vy, vz, yaw])
    
    return mission

if __name__ == "__main__":

    json_path = os.path.join(sys.path[0], 'db_FD.json')

    drones = [20100]
    mav_num = len(drones)
    sw = RflySW.RflySW(mav_num)
    sw.Start()

    mavs = []
    threads = []
    for drone in drones:
        mav = RflyMav(drone)
        thread = threading.Thread(target=mav.run)
        mavs.append(mav)
        threads.append(thread)
    
    for thread in threads:
        thread.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'FD_MGAT_GRU.pth'
    folder_name = model_name.split('.')[0]
    path = "..."
    graph = True
    Etype = True
    sensor_num = 6
    from utilits.FD_MGAT_GRU import *
    model = torch.load(path)
    model.to(device)

    all_messages = ""
    FD_LOG = queue.Queue()
    log_cnt = 1
    breakflag = False
    stop_flag = False
    
    mission = generate_faulty_mission(oscillate_axis='x', amplitude=1, frequency=1.095, total_time=5, time_step=0.01)

    fdt = threading.Thread(target=FDMav(mavs, model, device, graph, Etype, sensor_num).FauluDiagnosis)
    fdt.start()

    lastTime = time.time()
    hz = 500
    while True:
        lastTime = lastTime + (1.0/hz)
        sleepTime = lastTime - time.time()
        if sleepTime > 0:
            time.sleep(sleepTime)
        else:
            lastTime = time.time()

        if all(mav.round_over for mav in mavs):
            breakflag = True
            break
    
    for mav in mavs:
        get_data(mav, cnt='1')
    
    


    