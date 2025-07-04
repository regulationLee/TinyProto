import time
import os
import csv
import numpy as np
import itertools
from itertools import combinations
import torch
from flcore.clients.clienttinyfp import clientTinyFP
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

class TinyFP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientTinyFP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes
        self.i_i = 0

        # set proto mask
        if args.add_cps:
            self.mask_dict = {}
            self.mask_idx_dict = {}
            length = args.feature_dim // args.num_classes

            vector_name = f'mask_idx_dict_fd_{args.feature_dim}_nc_{args.num_classes}_csr_{args.csr_ratio}.pth'
            vector_path = os.path.join('./mask_indices', vector_name)

            if os.path.exists(vector_path):
                print("Load Mask Index.")
                self.mask_idx_dict = torch.load(vector_path)
            else:
                print("Create Mask Index.")
                per_mask = int(args.num_classes * (args.csr_ratio / 100))
                vectors, _ = find_optimal_vectors_greedy(args.num_classes, per_mask)
                for key in range(args.num_classes):
                    masked_proto = np.zeros(args.feature_dim)
                    for i, val in enumerate(vectors[key]):
                        start_idx = i * length
                        end_idx = start_idx + length
                        masked_proto[start_idx:end_idx] = val
                    self.mask_dict[key] = torch.tensor(masked_proto.astype((np.float32)))
                    self.mask_idx_dict[key] = list(np.nonzero(masked_proto))
                torch.save(self.mask_idx_dict, vector_path)
                print("Finished creating Mask Index.")

    def train(self):
        print(f"\n-------------Local Models-------------")
        for c in self.clients:
            print(f'Client {c.id}: {c.model_name}')
            if self.args.add_csr:
                c.set_mask(self.mask_idx_dict)

        for i in range(self.global_rounds + 1):
            self.i_i = i
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()
                if self.diverge:
                    break

            print("\nLocal Training.")
            for client in self.selected_clients:
                client.train()

            print("\nPrototype Aggregation.")
            self.receive_protos()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        if self.diverge:
            print("\nDiverge")
            data_to_add = {
                "exp_name": self.args.goal,
                "accuracy": "nan",
                "min_dist": "nan",
                "max_dist": "nan",
                "avg_dist": "nan",
                "avg_time": "nan",
            }

        else:
            argmax_idx = np.argmax(self.rs_test_acc)
            data_to_add = {
                "exp_name": self.args.goal,
                "accuracy": max(self.rs_test_acc),
                "min_dist": self.rs_min_dist[argmax_idx],
                "max_dist": self.rs_max_dist[argmax_idx],
                "avg_dist": self.rs_avg_dist[argmax_idx],
                "avg_time": sum(self.Budget[1:]) / len(self.Budget[1:]),
            }

            print("\nBest accuracy.")
            print(data_to_add['accuracy'])
            print(f"Minimum distance = {data_to_add['min_dist']:.4f} / angle = {data_to_add['min_ang']:.4f}")
            print(f"Maximum distance = {data_to_add['max_dist']:.4f} / angle = {data_to_add['max_ang']:.4f}")
            print(f"Average distance = {data_to_add['avg_dist']:.4f} / angle = {data_to_add['avg_ang']:.4f}")

            print("\nAverage Process time.")
            print(data_to_add['avg_time'])

        self.save_results()

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_data_num = []
        uploaded_protos = []

        uploaded_n_ij = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_data_num.append(client.num_data)
            uploaded_client_n_ij = defaultdict(list)

            protos = load_item(client.role, 'protos', client.save_folder_name)
            for label in protos.keys():
                uploaded_client_n_ij[label].append(self.args.data_dist[client.id][label])

            uploaded_protos.append(protos)
            uploaded_n_ij.append(uploaded_client_n_ij)

        global_protos = proto_aggregation(
            self.args,
            uploaded_protos,
            uploaded_n_ij,
            self.uploaded_data_num,
            self.uploaded_ids
        )

        if self.args.add_cps:
            self.calc_proto_distance(global_protos, device=self.args.device, mask_idx=self.mask_idx_dict)
        else:
            self.calc_proto_distance(global_protos, device=self.args.device)

        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(conf, local_protos_list, local_proto_n_ij, data_num_list, uploaded_id):
    ### edited version of aggregation
    agg_protos_label = defaultdict(list)
    agg_protos_label_true_dist = defaultdict(list)
    agg_protos_n_ij = defaultdict(list)
    for local_protos, proto_n_ij in zip(local_protos_list, local_proto_n_ij):
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])
            agg_protos_label_true_dist[label].append(local_protos[label])
            agg_protos_n_ij[label].append(proto_n_ij[label])

    # using true class distribution FedProto - Eq(6)
    for k in agg_protos_label.keys():
        protos = torch.stack(agg_protos_label[k])
        if conf.simple_scale:
            agg_protos_label[k] = len(protos) * conf.constant_scale_factor * torch.mean(protos, dim=0).detach()
        else:
            agg_protos_label[k] = torch.mean(protos, dim=0).detach()

        return agg_protos_label

def masking_global_proto(conf, global_protos, proto_mask):
    for [label, proto_list] in global_protos.items():
        tmp_mask = proto_mask[label]
        tmp_proto = tmp_mask.to(conf.device) * proto_list
        global_protos[label] = tmp_proto
    return global_protos


def proto_visualization(conf, g_proto, iter):
    g_proto_np = np.transpose(np.stack([g_proto[key].cpu().numpy() for key in sorted(g_proto.keys())]))

    plt.figure(figsize=(10, 6))
    sns.heatmap(g_proto_np, cmap='viridis')
    plt.title("Global_Proto_" + conf.goal + '_iter_' + str(iter))
    plt.ylabel("Feature Dimension")
    plt.xlabel("Class")
    result_name = conf.goal + '_iter_' + str(iter)
    plt.savefig(result_name + '.png')


def hamming_distance(v1, v2):
    return np.count_nonzero(v1 != v2)


def generate_random_vector(n, ones_count):
    vector = np.zeros(n, dtype=int)
    vector[:ones_count] = 1
    np.random.shuffle(vector)
    return vector


def create_start_vector(n, ones_count):
    vector = np.zeros(n, dtype=int)
    vector[-ones_count:] = 1  # Place 1s at the end
    return vector


def find_optimal_vectors_greedy(n, ones_count):
    # Start with the generalized vector
    start_vector = create_start_vector(n, ones_count)
    vectors = [start_vector]

    for _ in range(n - 1):  # We already have one vector, so generate n-1 more
        best_vector = None
        max_total_distance = -1
        attempts = 0
        max_attempts = 10000

        while attempts < max_attempts:
            candidate = generate_random_vector(n, ones_count)

            # Check if the candidate is unique
            if any(np.array_equal(candidate, v) for v in vectors):
                attempts += 1
                continue

            total_distance = sum(hamming_distance(candidate, v) for v in vectors)
            if total_distance > max_total_distance:
                max_total_distance = total_distance
                best_vector = candidate

            attempts += 1

        if best_vector is None:
            raise ValueError(f"Failed to find a unique vector after {max_attempts} attempts")

        vectors.append(best_vector)

    vectors = np.array(vectors)
    total_distance = sum(hamming_distance(v1, v2) for v1, v2 in combinations(vectors, 2))
    average_distance = total_distance / (n * (n - 1) / 2)

    return vectors, average_distance