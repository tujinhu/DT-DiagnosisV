import time
import numpy as np
import math
import threading
from ver.include.RflySoftwarm import REG

class MAVREG:
    FD_LOG_PHASE = None

class Sleep:
    def __init__(self,mav, id):
        self.CID = 1
        self.mav = mav
        self.ID = id
        self.isDone = 0 
        self.WaitFlag = 0
        self.WaitResetFlag = 0
        self.start_time = 0
    
    def Wait(self,times): 
        self.isDone = 0
        if self.WaitFlag == 0:
            print(f'Mav{self.ID} wait {times[0]}s')
            self.start_time = time.time() + times[0]
            self.WaitFlag = 1
    
        if self.start_time - time.time() < 0: 
            self.isDone = 1
            self.WaitFlag = 0
    
    def WaitReset(self,targetPos): 
        self.isDone = 0
        curPos=self.mav.uavPosNED
        if self.WaitResetFlag == 0:
            print('wait reset')
            self.WaitResetFlag = 1

        dis = math.sqrt((curPos[0]-targetPos[0])**2+(curPos[1]-targetPos[1])**2)
        if dis < 5:
            print('Arrive at the destination')
            self.isDone = 1
            self.WaitResetFlag = 0

class Command:
    def __init__(self,mav,id,connect):
        self.CID = 2
        self.mav = mav
        self.connect = connect
        self.ID = id
        self.ARMFLAG = False 
        self.isDone = 0 
        self.RECORDFLAG = False
        self.LANDFLAG = False
        self.LANDFLAGTAG = False
        self.silInt = np.zeros(8).astype(int).tolist()
        self.silFloats = np.zeros(20).astype(float).tolist()
        self.INJECTFLAG = False
        self.TAKEOFFFLAG = False
        self.MISSIONFLAG = False
        self.FAULTID = 0
        self.isInitOff = 0

    def Arm(self):  # ID = 1
        self.isDone = 0
        REG.barrier.wait()
        self.mav.SendMavArm(1)
        print(f'Mav{self.ID} armed!')
        MAVREG.FD_LOG_PHASE = '1'
        self.mav.SendMavCmdLong(183,2,1,0,0,0,0,666)
        self.ARMFLAG = True
        self.isDone = 1
        self.RECORDFLAG = True

    def DisArm(self): # ID = 2
        self.isDone = 0
        self.mav.SendMavArm(0) 
        print(f'Mav{self.ID} disArmed!') 
        self.isDone = 1
    
    def QuadTakeoff(self,pos): # ID = 3
        self.isDone = 0
        REG.barrier.wait()
        curPos=self.mav.uavPosNED
        if pos[0] == 0 and pos[1] == 0:
            pos[0] = curPos[0]
            pos[1] = curPos[1]
        self.mav.SendPosNED(pos[0],pos[1],pos[2])
        self.mav.SendMavCmdLong(183,2,3,pos[0],pos[1],pos[2],0,666)
        print(f'Mav{self.ID} send target Pos {pos}')
        MAVREG.FD_LOG_PHASE = '3'
        self.isDone = 1

    def QuadPos(self,pos):  # ID = 4
        self.isDone = 0
        REG.barrier.wait()
        self.mav.SendPosNED(pos[0],pos[1],pos[2])
        self.mav.SendMavCmdLong(183,2,4,pos[0],pos[1],pos[2],0,666)
        print(f'Mav{self.ID} send target Pos {pos}')
        MAVREG.FD_LOG_PHASE = '4'
        self.isDone = 1

    def QuadVel(self,vel):  # ID = 5
        self.isDone = 0
        REG.barrier.wait()
        self.mav.SendVelNED(vel[0],vel[1],vel[2])
        self.mav.SendMavCmdLong(183,2,5,vel[0],vel[1],vel[2],0,666)
        print(f'Mav{self.ID} send target Vel {vel}')
        MAVREG.FD_LOG_PHASE = '5'
        self.isDone = 1

    def hover(self):  # ID = 6
        self.isDone = 0
        REG.barrier.wait()
        curPos=self.mav.uavPosNED
        self.mav.SendPosNED(curPos[0],curPos[1],curPos[2])
        print(f'Mav{self.ID} start hovering')
        self.mav.SendMavCmdLong(183,2,6,0,0,0,0,666)
        MAVREG.FD_LOG_PHASE = '6'
        self.isDone = 1

    def UAVLand(self):  # ID = 7
        self.isDone = 0
        curPos=self.mav.uavPosNED
        REG.barrier.wait()
        self.LANDFLAG = True
        if self.LANDFLAGTAG == False:
            self.mav.sendMavLand(curPos[0],curPos[1],0)
            self.mav.SendMavCmdLong(183,2,7,0,0,0,0,666)
            print(f'Mav{self.ID} start landing')
            self.LANDFLAGTAG = True
            MAVREG.FD_LOG_PHASE = '7'
            self.isDone = 1
            self.LANDFLAGTAG = False
    
    def MissionRun(self, setpoint):
        for setp in setpoint:
            self.mav.SendVelNED(setp[0],setp[1],setp[2])
            time.sleep(0.1)
        
        # self.mav.SendVelNED(0,0,0,0)

    def UAVMission(self, setpoint):  # ID = 8
        self.isDone = 0
        REG.barrier.wait()
        Tm = threading.Thread(target=self.MissionRun, args=(setpoint, ))
        Tm.start()
        self.mav.SendMavCmdLong(183,2,8,0,0,0,0,666)
        print(f'Mav{self.ID} start mission')
        MAVREG.FD_LOG_PHASE = '8'
        self.isDone = 1
    
    def FaultInject(self,param):  # ID = 9
        self.isDone = 0
        self.inInts = np.array([])
        self.inFloats = np.array([])
        for i in range(len(param)):
            if param[i] >= 123450:
                self.inInts = np.append(self.inInts,param[i])
            else:
                self.inFloats = np.append(self.inFloats,param[i])

        # REG.barrier.wait()
        for i in range(len(self.inInts)):
            self.silInt[i] = self.inInts[i].astype(int)
        for i in range(len(self.inFloats)):
            self.silFloats[i] = self.inFloats[i].astype(np.double)

        print('Start Inject Fault')

        if self.connect == 'dt-fault' or self.connect == 'real':
            # self.mav.sendSILIntFloat(self.silInt,self.silFloats)
            # self.mav.SendMavCmdLong(183,2,9,self.silInt[0],self.inFloats[0],self.inFloats[1],0,666)
            mode = int(param[0])
            flag = int(param[1])
            ctrls = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            len_pc = len(param) - 2
            for i in range(len_pc):
                ctrls[i] = param[i + 2]
            
                # if self.connect == 'real' and i == len_pc-1:
                #     ctrls[i] = param[i + 2] * 3

            # if MAVREG.REG_WAIT == True:
            #     REG.sycn_cmd_barrier.wait()
            print('Start Inject Fault')

            self.mav.SendHILCtrlMsg(mode, flag, ctrls)
            self.mav.SendMavCmdLong(183,2,9,mode,flag,0,0,666)
        
        self.FAULTID = self.silInt[0]
        MAVREG.FD_LOG_PHASE = '9'
        self.isDone = 1
        self.INJECTFLAG = True
    
    def cmdStopTag(self,param): 
        self.isDone = 0
        self.mav.SendVelNED(0,0,0)
        print(f'Mav{self.ID} stop cmd')
        self.mav.SendMavCmdLong(183,param[0],param[1],0,0,0,0,777)
        self.isDone = 1
        

class CmdCtrl:
    def __init__(self,mav,frame,id,connect):
        self.mav = mav
        self.frame = frame
        self.ID = id
        self.connect = connect
        self.CID = {
        '1':Sleep(mav, id),
        '2':Command(mav, id, connect)
        }
        self.CID1 = self.CID['1']
        self.CID2 = self.CID['2']
        self.FID = 0
    
    def GetWaitseq(self):
        Waitseq = {
                '1':self.CID1.Wait,
                '2':self.CID1.WaitReset
        }
        return Waitseq

    def GetCmdseq(self):
        if self.frame == 1: 
            Cmdseq = {
                '1':self.CID2.Arm,
                '2':self.CID2.DisArm,
                '3':self.CID2.QuadTakeoff,
                '4':self.CID2.QuadPos,
                '5':self.CID2.QuadVel,
                '6':self.CID2.hover,
                '7':self.CID2.UAVLand,
                '8':self.CID2.UAVMission,
                '9':self.CID2.FaultInject,
                '10':self.CID2.cmdStopTag
            }
        return Cmdseq

    def FIDPro(self,cmdCID):
        if cmdCID == '1':
            self.FID = CmdCtrl.GetWaitseq(self)
        elif cmdCID == '2':
            self.FID = CmdCtrl.GetCmdseq(self)
        return self.FID



