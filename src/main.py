import argparse

import dill
import numpy as np
import torch
from modules.causal_construction import CausaltyGraph4Visit
from src.modules.RaVSNet import RaVSNet
from training import Test, Train, pre_training, Case
from util import  relevance_mining, adj_matrix

def set_seed():
    torch.manual_seed(1203)
    np.random.seed(2048)


def parse_args():
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument("--debug", default=False,
                        help="debug mode, the number of samples, "
                             "the number of generations run are very small, "
                             "designed to run on cpu, the development of the use of")
    parser.add_argument("--Test", default=False, help="test mode")
    parser.add_argument("--pretrain", default=False, help="re-pretrain this time")

    # environment
    parser.add_argument('--dataset', default='mimic3', help='mimic3/mimic4')
    parser.add_argument('--resume_path', default="../saved/mimic3/trained_model_0.5763", type=str,
                        help='path of well trained model, only for evaluating the model, needs to be replaced manually')
    parser.add_argument("--resume_path_pretrained", default="../saved/mimic3/pretrained/mask_single_record_dim=256.pt",
                        help="pretrained model")
    parser.add_argument('--device', type=int, default=0, help='gpu id to run on, negative for cpu')
    parser.add_argument('--table_path', default="../saved/mimic3/best_table_0.5763", type=str,
                        help='path of well trained embedding table, only for evaluating the model, needs to be replaced manually')

    # parameters
    parser.add_argument('--dim', default=64, type=int, help='model dimension')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--dp', default=0.7, type=float, help='dropout ratio')
    parser.add_argument('--epochs', default=10, type=int, help='the epochs for training')

    args = parser.parse_args()
    if args.Test and args.resume_path is None:
        raise FileNotFoundError('Can\'t Load Model Weight From Empty Dir')

    return args


if __name__ == '__main__':
    set_seed()
    args = parse_args()
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device("cpu")
        if not args.Test:
            print("GPU unavailable, switch to debug mode")
            # args.debug = True
    else:
        device = torch.device(f'cuda:{args.device}')

    data_path = f'../data/{args.dataset}/outputs/records_final.pkl'
    voc_path = f'../data/{args.dataset}/outputs/voc_final.pkl'
    ddi_adj_path = f'../data/{args.dataset}/outputs/ddi_A_final.pkl'
    relevance_diag_med_path = f'../data/{args.dataset}/graphs/Diag_Med_causal_effect.pkl'
    relevance_proc_med_path = f'../data/{args.dataset}/graphs/Proc_Med_causal_effect.pkl'
    relevance_sym_med_path = f'../data/{args.dataset}/graphs/Sym_Med_causal_effect.pkl'
    pre_train_data_path = f"../data/{args.dataset}/outputs/mask_single_records_final.pkl"

    with open(ddi_adj_path, 'rb') as Fin:
        ddi_adj = torch.from_numpy(dill.load(Fin)).to(device)
    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
        adm_id = 0
        for patient in data:
            for adm in patient:
                adm.append(adm_id)
                adm_id += 1
        if args.debug:
            data = data[:5]
    # data = data[:50]

    if args.dataset == 'mimic3':
        for patient in data:
            for visit in patient:
                del visit[4]  # 删除索引为4的元素

    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)
    with open(pre_train_data_path, 'rb') as Fin:
        pre_train_data = dill.load(Fin)
    with open(relevance_proc_med_path, 'rb') as Fin:
        relevance_proc_med = dill.load(Fin)
    with open(relevance_diag_med_path, 'rb') as Fin:
        relevance_diag_med = dill.load(Fin)
    with open(relevance_sym_med_path, 'rb') as Fin:
        relevance_sym_med = dill.load(Fin)

    #load data
    pre_train_split_point = int(len(pre_train_data) * 4 / 5)
    pre_train_data_train = pre_train_data[:pre_train_split_point]
    pre_train_data_eval = pre_train_data[pre_train_split_point:]
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    diag_voc, pro_voc, sym_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['sym_voc'], voc['med_voc']
    voc_size = [
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(sym_voc.idx2word),
        len(med_voc.idx2word),
    ]
    print(voc_size)

    ehr_adj_med_diag, ehr_adj_med_proc, ehr_adj_med_med, ehr_adj_med_sym = relevance_mining(data, voc_size)
    ehr_adj_med_diag = adj_matrix(ehr_adj_med_diag)
    ehr_adj_med_proc = adj_matrix(ehr_adj_med_proc)
    ehr_adj_med_sym = adj_matrix(ehr_adj_med_sym)


    causal_graph = CausaltyGraph4Visit(data, data_train, voc_size[0], voc_size[1], voc_size[2],  voc_size[3], args.dataset)
    print("*********Pretrain*********")
    pretrained_model = pre_training(pre_train_data_train, pre_train_data_eval, voc_size, ddi_adj, device, args)
    pretrained_model.to(device=device)

    pretrained_embedding = []
    medication_list = []

    for step, patient in enumerate(data):
        for idx, adm in enumerate(patient):
            medication_list.append(adm[3])
    print(len(medication_list))

    print("*********Start*********")
    model = RaVSNet(
        causal_graph,
        voc_size,
        ddi_adj,
        ehr_adj_med_diag,
        ehr_adj_med_proc,
        ehr_adj_med_med,
        ehr_adj_med_sym,
        medication_list,
        pretrained_model.embeddings,
        emb_dim=args.dim,
        device=device,
    ).to(device)

    print("1.Training Phase")
    if args.Test:
        print("Test mode, skip training phase")
        with open(args.resume_path, 'rb') as Fin:
            model.load_state_dict(torch.load(Fin, map_location=device))
        visit_embedding_table = torch.load(args.table_path)
    else:
        for step, patient in enumerate(data_train):
            embedding_pretrained = [torch.zeros([1, args.dim], dtype=torch.float).to(device) for _ in range(4)]
            for idx, adm in enumerate(patient):
                for diag in adm[0]:
                    embedding_pretrained[0] += pretrained_model.embeddings[0].weight.data[diag]
                for pro in adm[1]:
                    embedding_pretrained[1] += pretrained_model.embeddings[1].weight.data[pro]
                for sym in adm[2]:
                    embedding_pretrained[2] += pretrained_model.embeddings[2].weight.data[sym]
                if len(patient[: idx + 1]) <= 1:
                    embedding_pretrained[3] += torch.zeros((1, args.dim)).to(device)
                else:
                    for med in patient[: idx + 1][-2][3]:
                        embedding_pretrained[3] += pretrained_model.embeddings[3].weight.data[med]
                pretrained_embedding.append(torch.cat(
                    [embedding_pretrained[0], embedding_pretrained[1], embedding_pretrained[2],
                     embedding_pretrained[3]], dim=-1).squeeze(dim=0))

        visit_embedding_table = torch.stack(pretrained_embedding)
        model, visit_embedding_table = Train(model, device, data_train, data_eval, voc_size, visit_embedding_table, args)

    print("2.Testing Phase")
    Case(model,device,data_test, voc_size,visit_embedding_table,med_voc)
    Test(model, device, data_test, voc_size, visit_embedding_table)