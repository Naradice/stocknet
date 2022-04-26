import threading
import pandas as pd
import os

class pt_logs:
    def __init__(self,env, root_path = './', folder='logs', save_obs = False):
        self.logs = []
        self.invalid_max = 0
        self.invalid_min = 0
        self.valid_max = -100
        self.valid_min = 100
        self.env = env
        self.__so = save_obs
        self.columns = env.columns.copy()
        self.columns.append("budgets")
        self.columns.append("coins")
        
        try:
            self.base_path = os.path.join(root_path, folder)
        except:
            print(f'specified path {root_path} {folder} is invalid')
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            
        if self.__so:
            self.obs_path = os.path.join(self.base_path, 'obs')
            if not os.path.exists(self.obs_path):
                os.makedirs(self.obs_path)
            self.obs = pd.DataFrame([])

    def __check_logs(self, logs_df):
        ## check if reward is in range
        #TOTO change -1, 1 to min, max of env
        if logs_df.reward.max() > 1 or logs_df.reward.min() < -1:
            print(f"invalid reward at {self.env.get_data_index()-1}: {logs_df.reward.max()}")
        ## check is number of kind of actions is less than action_scpace
    
    def __check_obs(self, obs):
        #TOTO change -1, 1 to min, max of env
        max_value = obs.max()
        if max_value > 1:
            if self.invalid_max != max_value:
                self.invalid_max = max_value
                print(f"invalid valu in observations at {self.env.get_data_index()-1}: {self.invalid_max}")
        else:
            if max_value > self.valid_max:
                self.valid_max = max_value
            
        min_value = obs.min()
        if min_value < -1:
            if self.invalid_min != min_value:
                self.invalid_min = min_value
                print(f"invalid valu in observations at {self.env.get_data_index()-1}: {self.invalid_min}")
            else:
                if min_value < self.valid_min:
                    self.value_min = min_value
        
        ## check is obs is same as data obtained by index

    def store(self, obs, action, reward):
        log = {}
        log["index"] = self.env.get_data_index()-1
        log["ask"] = self.env.ask
        log["bid"] = self.env.bid
        log["act"] = action
        log["reward"] = reward
        self.logs.append(log)
        self.__check_obs(obs)
        if self.__so:
            self.obs = self.obs.append(pd.DataFrame(obs, columns=self.columns))
            
        
    def __save(self,logs, ep_num):
        content = pd.DataFrame(logs)
        try:
            path = os.path.join(self.base_path, f'episode_{ep_num}.csv')
            content.to_csv(path)
        except Exception as e:
            print(e)
        self.logs = []
        
        if self.__so:
            file = os.path.join(self.obs_path, f'episode_{ep_num}.csv')
            self.obs.to_csv(file)
            self.obs = pd.DataFrame([])
        
    def save(self, ep_num):
        logs = self.logs.copy()
        threading.Thread(target=self.__save(logs, ep_num))
        self.logs = []