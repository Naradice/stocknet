import copy
import datetime
import threading

import numpy as np
import torch
from tqdm import tqdm

from .. import utils
from ..envs.render import graph


def __sqe2seq_main(index, model, ds, criterion, batch_size, split_tgt_func):
    temp = ds[index : index + batch_size]
    src, tgt = temp[:2]
    option_args = temp[2:]

    input_tgt, output_tgt = split_tgt_func(tgt)
    logits = model(src, input_tgt, *option_args)
    loss = criterion(logits, output_tgt)
    return loss


def seq2seq_eval(model, ds, criterion, batch_size, batch_first=True):
    model.eval()
    ds.eval()
    losses = []

    if batch_first is True:
        split_tgt = lambda tensor: (tensor[:, :-1, :], tensor[:, 1:, :])
    else:
        split_tgt = lambda tensor: (tensor[:-1], tensor[1:])
    end_index = len(ds)
    for index in tqdm(range(0, end_index - batch_size, batch_size)):
        loss = __sqe2seq_main(index, model, ds, criterion, batch_size, split_tgt)
        losses.append(loss.item())

    return np.mean(losses)


def seq2seq_train(model, ds, optimizer, criterion, batch_size, batch_first=True):
    model.train()
    ds.train()
    losses = []

    if batch_first is True:
        split_tgt = lambda tensor: (tensor[:, :-1, :], tensor[:, 1:, :])
    else:
        split_tgt = lambda tensor: (tensor[:-1], tensor[1:])
    end_index = len(ds)
    for index in tqdm(range(0, end_index - batch_size, batch_size)):
        optimizer.zero_grad()
        loss = __sqe2seq_main(index, model, ds, criterion, batch_size, split_tgt)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


def seq2seq_simulation_train(obs_length, model, generator, optimizer, criterion, device, SMA=100, batch_first=True):
    model.train()
    losses = np.array([])
    sma_loss = np.inf

    if batch_first is True:
        get_mask_size = lambda tensor: tensor.size(1)
    else:
        get_mask_size = lambda tensor: tensor.size(0)

    for observations in generator:
        # assume batch_first=True
        src = observations[:, :obs_length]
        tgt = observations[:, obs_length:]

        input_tgt = tgt[:, :-1]

        mask_tgt = torch.nn.Transformer.generate_square_subsequent_mask(get_mask_size(input_tgt), device=device)
        logits = model(src=src, tgt=input_tgt, mask_tgt=mask_tgt)

        optimizer.zero_grad()

        output_tgt = tgt[:, 1:]
        loss = criterion(logits, output_tgt)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        losses = np.append(losses, loss_value)
        if len(losses) >= SMA:
            if len(losses) % 10 == 0:
                mean_loss = losses[-SMA:].mean()
                if sma_loss >= mean_loss:
                    sma_loss = mean_loss
                else:
                    break
    return losses.mean()


class Trainer:
    def __init__(self, model_name, loss_fn, train_loader, val_loader, device) -> None:
        self.validation_losses = []
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.name = model_name

    def __save_result(self, txt: str):
        utils.save_result_as_txt(self.name, txt)

    def __save_model(self, model):
        utils.save_model(model, self.name)

    def __check(self):
        mean_loss = 0.0
        with torch.no_grad():
            for values, ans in self.val_loader:
                outputs = self.model_cp(values.to(self.device)).to(self.device)
                ans = ans.to(self.device)
                loss = self.loss_fn(outputs, ans)
                mean_loss += loss.item()
        print("current validation loss:", mean_loss / len(self.val_loader))
        if len(self.validation_losses) > 0:
            if self.validation_losses[-1] > mean_loss:
                self.__save_model(self.model_cp)
                self.val_decreased = False
            else:
                self.val_decreased = True
        self.validation_losses.append(mean_loss)

    def check(self, model):
        self.model_copy(model)
        threading.Thread(target=self.__check())

    def plot_validation_results(self):
        graph.line_plot(self.validation_losses, 10, True, f"{self.name}.png")

    def model_copy(self, model):
        self.model_cp.load_state_dict(model.state_dict())
        self.model_cp = self.model_cp.to(self.device)

    def save_client(self, client):
        utils.save_client_params(self.name, client)

    def training_loop(self, model, optimizer, n_epochs=-1, mode="human", validate=False):
        self.model_cp = copy.deepcopy(model)
        mean_loss = 0
        start_time = datetime.datetime.now()
        ep_consumed_total_time = datetime.timedelta(0)
        utils.load_model(model, self.name)
        model = model.to(self.device)
        # save_tarining_params(ds.get_params(), model_name)

        epoch = 1
        auto = False
        if n_epochs == -1:
            auto = True
            n_epochs = 10
            max_cont_false = 2
            false_count = 0

        self.val_decreased = False
        print(start_time, "start epocs")
        while epoch <= n_epochs:
            loss_train = np.array([])
            ep_start_time = datetime.datetime.now()
            for inputValues, ansValue in self.train_loader:
                outputs = model(inputValues.to(self.device))
                loss = self.loss_fn(outputs.to(self.device), ansValue.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # loss_train += loss.item()
                loss_train = np.append(loss_train, loss.item())
            ep_end_time = datetime.datetime.now()
            ep_consumed_time = ep_end_time - ep_start_time
            # total = ep_consumed_time.total_seconds()
            # print(f'{time_records["render"]}/{total}, {time_records["action"]}/{total}, {time_records["step"]}/{total}, {time_records["observe"]}/{total}, {time_records["log"]}/{total}')
            ep_consumed_total_time += ep_consumed_time

            temp_loss = mean_loss
            mean_loss = loss_train.mean()
            diff_loss = mean_loss - temp_loss

            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"{datetime.datetime.now()} Epoch {epoch}, Training loss:: Mean: {mean_loss} : Std: {loss_train.std()}, Range: {loss_train.min()} to {loss_train.max()}, Diff: {diff_loss}"
                )
                print(f"consumed time: {ep_consumed_time}, may end on :{ep_end_time + (n_epochs - epoch) *  ep_consumed_total_time/epoch}")
            if validate:
                self.check(model)
                if auto:
                    if self.val_decreased:
                        break
                    else:
                        n_epochs += 1
            elif auto:
                if epoch > 1:
                    if min_loss > mean_loss:
                        min_loss = mean_loss
                        self.__save_model(model)
                        false_count = 0
                    else:
                        false_count += 1
                        if false_count == max_cont_false:
                            break
                else:
                    min_loss = mean_loss
                n_epochs += 1
            epoch += 1
        self.__save_model(model)
        # self.plot_validation_results()
        result_txt = f"training finished on {datetime.datetime.now()}, {datetime.datetime.now()} Epoch {epoch}, Training loss:: Mean: {mean_loss} : Std: {loss_train.std()}, Range: {loss_train.min()} to {loss_train.max()}, Diff: {diff_loss}"
        print(result_txt)
        self.__save_result(result_txt)

    def validate(self, model, val_loader):
        mean_loss = 0
        out_ = {}
        ans_ = {}
        output_shape = val_loader.dataset[0][1].shape  # tuple(batch, input, output]
        output_size = output_shape.numel()
        for index in range(0, output_size):
            out_[index] = np.array([])
            ans_[index] = np.array([])

        viewer = graph.Rendere()

        with torch.no_grad():
            count = 0
            for values, ans in val_loader:
                outputs = model(values).to(self.device)
                ans = ans.to(self.device)
                loss = self.loss_fn(outputs, ans)
                mean_loss += loss.item()
                # output: [batchDim, outputDim]
                for index in range(0, output_size):
                    out_[index] = np.append(out_[index], outputs.to("cpu").detach().numpy().copy())
                    ans_[index] = np.append(ans_[index], ans.to("cpu").detach().numpy().copy())
                count += 1

        print("--------------------------------------------------")
        print(f"mean loss: {mean_loss/count}")
        print(
            f"mean dif ({index}): {[(out_[index] - ans_[index]).mean() for index in range(0, output_size)]}, var: {[(out_[index] - ans_[index]).var() for index in range(0, output_size)]}"
        )
        print("--------------------------------------------------")

        for index in range(0, output_size):
            viewer.register_xy(ans_[index], out_[index], index=-1)
        file_name = utils.get_validate_filename(self.name, "png")
        viewer.plot()
        viewer.write_image(file_name)
