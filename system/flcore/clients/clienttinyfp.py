import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict


class clientePBFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # torch.manual_seed(0)
        self.args = args

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        self.num_data = sum(self.args.data_dist[self.id])
        self.data_dist_norm = [x / sum(self.args.data_dist[self.id]) for x in self.args.data_dist[self.id]]

        self.mask_idx_dict = defaultdict(list)

    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs

        protos = defaultdict(list)
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                # prototype distillation loss
                if global_protos is not None:
                    if self.args.add_cps:
                        proto_new = torch.zeros_like(rep.detach())

                        mask_indices = torch.tensor(
                            [self.mask_idx_dict[yy.item()][0] for yy in y],
                            device=self.device
                        )
                        global_proto_values = torch.stack(
                            [global_protos[yy.item()].data for yy in y],
                            dim=0
                        )
                        proto_new.scatter_(1, mask_indices, global_proto_values)
                    else:
                        proto_new = copy.deepcopy(rep.detach())

                        for i, yy in enumerate(y):
                            y_c = yy.item()

                            if type(global_protos[y_c]) != type([]):
                                proto_new[i, :] = global_protos[y_c].data

                        loss += self.lamda * self.loss_mse(proto_new, rep)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        save_item(
            agg_func(self.args, protos, self.args.data_dist[self.id], self.mask_idx_dict, self.data_dist_norm),
            self.role,
            'protos',
            self.save_folder_name
        )
        save_item(model, self.role, 'model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_mask(self, g_mask_idx_dict):
        self.mask_idx_dict = g_mask_idx_dict

    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        model.eval()

        test_acc = 0
        test_num = 0

        if global_protos is not None:
            if self.args.add_cps:
                global_protos_extended = {}
                # old_version
                for [label, proto_list] in global_protos.items():
                    mask_500d = np.zeros(self.args.feature_dim)
                    mask_500d = torch.tensor(mask_500d.astype((np.float32))).to(self.args.device)
                    mask_500d[self.mask_idx_dict[label][0]] = proto_list.data
                    global_protos_extended[label] = mask_500d
                global_protos = global_protos_extended

            #### tensor batch version
            # shape: (num_classes, feature_dim)
            global_protos = (
                torch.stack([global_protos[i] for i in range(self.args.num_classes)]).to(self.device))

            with torch.no_grad():
                for x, y in testloader:
                    if isinstance(x, list):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = model.base(x)

                    rep_expanded = rep.unsqueeze(1)  # shape: (batch_size, 1, feature_dim)
                    mse = torch.mean((rep_expanded - global_protos) ** 2, dim=2)  # shape: (batch_size, num_classes)

                    output = mse

                    test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    if self.args.add_cps:
                        proto_new = torch.zeros_like(rep.detach())

                        mask_indices = torch.tensor(
                            [self.mask_idx_dict[yy.item()][0] for yy in y],
                            device=self.device
                        )
                        global_proto_values = torch.stack(
                            [global_protos[yy.item()].data for yy in y],
                            dim=0
                        )
                        proto_new.scatter_(1, mask_indices, global_proto_values)

                    else:
                        proto_new = copy.deepcopy(rep.detach())

                        for i, yy in enumerate(y):
                            y_c = yy.item()

                            if type(global_protos[y_c]) != type([]):
                                proto_new[i, :] = global_protos[y_c].data

                    loss += self.lamda * self.loss_mse(proto_new, rep)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(conf, protos, class_num, mask_dict, proto_weights):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data

            protos[label] = proto / len(proto_list)  # original code of FedProto

            if conf.simple_scale:
                protos[label] = protos[label] * class_num[label]
        else:
            if conf.simple_scale:
                protos[label] = proto_list[0] * class_num[label]
            else:
                protos[label] = proto_list[0]

    if conf.add_cps:
        for key in protos.keys():
            indices = mask_dict[key][0]
            protos[key] = protos[key][indices]

    return protos


