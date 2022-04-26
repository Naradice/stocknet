import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import torch
import os
from torchinfo import summary
from stocknet import logger as lg
from stocknet.envs.render.graph import Rendere
import stocknet.envs.render.graph as graph
import json
import threading
import copy
##DEBUG
#from envs.render.graph import Rendere

def check_directory(model_name:str) -> None:
    if '/' in model_name:
        names = model_name.split('/')
        path_ = os.path.join('models', *names)
        if os.path.exists(path_) is False:
            os.makedirs(path_)
    elif os.path.exists('models') is False:
        os.makedirs('models')

def load_model(model, model_name):
    model_path = f'models/{model_name}.torch'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("model name doesn't exist. new model will be created.")

def save_model(model, model_name):
    check_directory(model_name)
    torch.save(model.state_dict(), f'models/{model_name}.torch')
    
def save_model_architecture(model, input, batch_size, model_name):
    check_directory(model_name)
    sum = summary(
        model,
        input_size = (batch_size, *input.shape),
        col_names=["input_size", "output_size", "num_params"]
    )
    sum_str = str(sum)
    with open(f'models/{model_name}.arc', 'w', encoding='utf-8') as f:
        f.write(sum_str)
        
#def copy_model(model):
#    copied_model = copy.deepcopy(model)
#    return copied_model

def save_tarining_params(params:dict, model_name):
    check_directory(model_name)
    params_str = json.dumps(params)
    with open(f'models/{model_name}.param', 'w', encoding='utf-8') as f:
        f.write(params_str)

class Validater():
    def __init__(self) -> None:
        pass
    
    def val_plot(self, val_loader):
        for input, out_ex in val_loader:
            pass

class Trainer():
    
    def __init__(self, model_name, loss_fn, train_loader, val_loader, device) -> None:
        self.validation_losses = []
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.name = model_name
        
    def __save_model(self, model):
        save_model(model, self.name)
    
    def __check(self):
        mean_loss = 0.0
        with torch.no_grad():
            for values, ans in self.val_loader:
                outputs = self.model_cp(values).to(self.device)
                ans = ans.to(self.device)
                loss = self.loss_fn(outputs, ans)
                mean_loss += loss.item()
        print("current validation loss:", mean_loss/len(self.val_loader))
        if len(self.validation_losses) > 0 :
            if self.validation_losses[-1]  > mean_loss:
                self.__save_model(self.model_cp)
                self.val_decreased = False
            else:
                self.val_decreased = True
        self.validation_losses.append(mean_loss)
        
    def check(self, model):
        self.model_copy(model)
        threading.Thread(target=self.__check())
        
    def plot_validation_results(self):
        graph.line_plot(self.validation_losses, 10, True, f'{self.name}.png')
        
    def model_copy(self, model):
        self.model_cp.load_state_dict(model.state_dict())
        
    def training_loop(self,model, optimizer, n_epochs=-1,mode="human"):
        self.model_cp = copy.deepcopy(model)
        losses = []
        start_time = datetime.datetime.now()
        ep_consumed_total_time = datetime.timedelta(0)
        load_model(model, self.name)
        model = model.to(self.device)
        #save_tarining_params(ds.get_params(), model_name)
        
        epoch = 1
        auto = False
        if n_epochs == -1:
            auto = True
            n_epochs = 10
            
        self.val_decreased = False
        print(start_time,'start epocs')
        while epoch <= n_epochs:
            loss_train  = 0.0
            ep_start_time = datetime.datetime.now()
            for inputValues, ansValue in self.train_loader:
                outputs = model(inputValues.to(self.device))
                loss = self.loss_fn(outputs.to(self.device), ansValue.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            ep_end_time = datetime.datetime.now()
            ep_consumed_time = ep_end_time - ep_start_time
            #total = ep_consumed_time.total_seconds()
            #print(f'{time_records["render"]}/{total}, {time_records["action"]}/{total}, {time_records["step"]}/{total}, {time_records["observe"]}/{total}, {time_records["log"]}/{total}')
            ep_consumed_total_time += ep_consumed_time
            
            if mode == "graph":
                losses.append(loss_train/len(self.train_loader))
            else:
                if epoch == 1 or epoch % 10 == 0:
                    print(f'{datetime.datetime.now()} Epoch {epoch}, Training loss {loss_train/len(self.train_loader)}')
                    print(f"consumed time: {ep_consumed_time}, may end on :{ep_end_time + (n_epochs - epoch) *  ep_consumed_total_time/epoch}")
            self.check(model)
            if auto:
                if self.val_decreased:
                    break
                else:
                    n_epochs+=1
            epoch+=1
        #self.plot_validation_results()
        print(f'training finished on {datetime.datetime.now()}, Training loss {loss_train/len(self.train_loader)}')
                    
class RlTrainer():
    
    def __init__(self) -> None:
        self.end_time = None
    
    # TODO: create base class
    def set_end_time(self, date:datetime):
        self.end_time = date
        
    def add_end_time(self, day:int = 0, hours:int = 0):
        self.end_time = datetime.datetime.now() + datetime.timedelta(days=day,hours=hours)

    def training_loop(self, env, agent, model_name:str, n_episodes:int = 20, max_step_len:int = 1000, device=None, render=True):
        mr = 0
        pl = 0
        max_mean_reward = env.reward_range[0]*max_step_len
        ep_consumed_total_time = datetime.timedelta(0)
        load_model(agent.model, model_name)
        
        start_time = datetime.datetime.now()
        start_time = start_time.isoformat().split('.')[0].replace(':', '-')
        logger = lg.pt_logs(env, folder=f'logs/{model_name}/{start_time}')
        viewer = Rendere()
        viewer.add_subplot()
        obs = env.reset()
        save_model_architecture(agent.model, obs, agent.minibatch_size, model_name)
        save_tarining_params(env.get_params(), model_name)
        
        print(start_time,'start episodes')
        ## show details
        ##
        for i in range(1, n_episodes + 1):
            #obs = obs.to('cpu').detach().numpy().copy()
            R = 0  # return (sum of rewards)
            t = 0  # time step
            ep_start_time = datetime.datetime.now()
            #time_records = { 'render':0, 'action':0, 'step':0, 'observe':0, 'log':0 }
            while True:
                # Uncomment to watch the behavior in a GUI window
                #render_start = datetime.datetime.now()
                if render:
                    env.render()
                #render_end = datetime.datetime.now()
                #time_records['render'] += (render_end - render_start).total_seconds()
                
                action = agent.act(obs)
                #act_end = datetime.datetime.now()
                #time_records['action'] += (act_end - render_end).total_seconds()
                
                obs, reward, done, ops = env.step(action)
                #step_end = datetime.datetime.now()
                #time_records['step'] += (step_end - act_end).total_seconds()
                
                R += reward
                t += 1
                reset = t == max_step_len
                agent.observe(obs, reward, done, reset)
                #observe_end = datetime.datetime.now()
                #time_records['observe'] += (observe_end - step_end).total_seconds()
                
                logger.store(obs, action, reward)
                #log_end = datetime.datetime.now()
                #time_records['log'] += (log_end - observe_end).total_seconds()
                
                if reset or done:
                    break
            ep_end_time = datetime.datetime.now()
            ep_consumed_time = ep_end_time - ep_start_time
            #total = ep_consumed_time.total_seconds()
            #print(f'{time_records["render"]}/{total}, {time_records["action"]}/{total}, {time_records["step"]}/{total}, {time_records["observe"]}/{total}, {time_records["log"]}/{total}')
            ep_consumed_total_time += ep_consumed_time
            logger.save(i)
            pl += env.pl
            mean_reward = R/t
            mr += mean_reward
            viewer.append_x(mean_reward,0)
            viewer.append_x(env.pl, 1)
            viewer.plot()
            obs = env.reset()
            if i % 10 == 0:
                sma = mr/10#TODO: changeg it to simple moving average or ema
                if sma > max_mean_reward:
                    save_model(agent.model, f'{model_name}_mr')
                    max_mean_reward = sma
                print('statistics:', agent.get_statistics(), 'R:', R/t, 'Mean R:', sma, 'PL:', env.pl, 'Mean PL:', pl/10)
                print(f"consumed time: {ep_consumed_time}, may end on :{ep_end_time + (n_episodes -i) *  ep_consumed_total_time/i}")
                mr = 0
                pl = 0
            if self.end_time != None:
                if self.end_time < datetime.datetime.now():
                    break
        print(f'Finished on {datetime.datetime.now()}')
        save_model(agent.model, model_name)
        viewer.write_image(f'models/{model_name}.png')