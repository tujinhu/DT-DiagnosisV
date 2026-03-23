import json
import re


class RflyDB:
    def __init__(self, path): 
        self.jsonpath = path

    def GET_CASEINFO(self, case_id):
        path = self.jsonpath

        with open(path, "r",encoding='utf-8') as f:
            json_data = json.load(f)
        
        caseInfo = json_data['FAULT_CASE']

        for fault_case in caseInfo:
            if fault_case['CaseID'] == case_id:
                return fault_case

        return None

    
    def GET_CASEID(self): 
        path = self.jsonpath

        with open(path, "r",encoding='utf-8') as f:
            json_data = json.load(f)

        case = json_data.get('TEST_CASE')
        '''
        case_eg:
        "1,2,5;2,1,1;4,5,6;"
        '''
        len_seg = len(re.findall(r';',case))
        if len_seg:
            mcase = re.findall(r'-?\d+', case.split(";")[0]) # ['1,2', '2,6']
            '''['1', '2']'''
            len_case = len(mcase)
            if len_case > 1:
                ''' Multi-machine mode multiple test cases, eg: "1,2;3,5" '''
                segments = case.split(";")
                '''
                ['1,2,5', '2,1,1', '4,5,6', '']
                '''
                segment_lists = [list(filter(None, segment.split(','))) for segment in segments]
                '''
                [['1', '2', '5'], ['2', '1', '1'], ['4', '5', '6'], []]
                '''
                max_length = max(len(segment) for segment in segment_lists)

                result = []
                for i in range(max_length):
                    result.append([int(segment[i]) for segment in segment_lists if i < len(segment)])
                '''
                [[1, 2, 4], [2, 1, 5], [5, 1, 6]]
                '''
                return result
            else:
                ''' Multi-machine mode single test case, eg:"1;2" ''' 
                segments = case.split(";")
                ''' ['1', '2'] '''
                result = []
                
                for segment in segments:
                    elements = list(map(int, segment.split(',')))
                    result.append(elements)

                return result
        else:
            result = [[int(ca) for ca in re.findall(r'-?\d+', case)]]
            return result

            warn = 'Sorry, your configuration is wrong, please check as follows:\n1) Whether your UAV num is greater than 1; \n2) Did you forget to add a semicolon(;) in the "TEST_CASE"  of your json file to switch to multi-machine mode?\n'
            print(warn)


    def MAV_JSONPro(self,case_id): 
        '''
        Change the test status information of json file
        '''
         
        path = self.jsonpath
        with open(path, "r",encoding='utf-8') as f:
            db_data = json.load(f)
        json_case = db_data.get('FAULT_CASE')
        '''
        [
        {'CaseID': 1, 'Subsystem': 'Power', 'Component': 'Motor', 'FaultID': '123450', 'FaultType': 'Motor Fault', 'FaultMode': 'Decreased efficiency of motor execution', 'FaultParams': '0', 'ControlSequence': '2,1;1,1,5;2,3,150,0,-30;1,1,10;2,4,1000,0,-30;1,1,10;2,6,123450,0;1,1,10', 'TestStatus': 'Finished'}
        ]
        '''
        if len(json_case) >= 1:
            path = self.jsonpath

            with open(path, "r",encoding='utf-8') as f:
                db_data = json.load(f)
                db_data['FAULT_CASE'][case_id-1]['TestStatus'] = 'Finished'
                data = db_data
            f.close()

            with open(path, "w",encoding='utf-8') as w:
                json.dump(data,w,indent=4) 
            w.close()