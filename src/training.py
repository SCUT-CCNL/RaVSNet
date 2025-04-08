import copy
import math
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import multilabel_margin_loss
from torch.optim import Adam
import torch.nn.functional as F

from modules.RaVSNet import BasicModel
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, parameter_report, Regularization, \
    graph_report, adjust_learning_rate


def pre_training(data_train, data_eval, voc_size, ddi_adj, device, args):
    pretrained_model = BasicModel(
        voc_size,
        ddi_adj,
        emb_dim=args.dim,
        device=device,
    )
    pretrained_model.to(device=device)

    if not args.pretrain or args.Test:
        pretrained_model.load_state_dict(torch.load(args.resume_path_pretrained, map_location=device))
        return pretrained_model

    else:
        optimizer = Adam(list(pretrained_model.parameters()), lr=args.lr)

        # start iterations
        best_epoch, best_ja = 0, 0
        best_model = None

        EPOCH = 20
        for epoch in range(EPOCH):
            tic = time.time()
            print("\nepoch {} --------------------------".format(epoch))

            pretrained_model.train()
            for step, patient in enumerate(data_train):

                loss = 0
                for idx, adm in enumerate(patient):

                    seq_input = patient[: idx + 1]
                    loss_bce_target = np.zeros((1, voc_size[3]))
                    loss_bce_target[:, adm[3]] = 1

                    loss_multi_target = np.full((1, voc_size[3]), -1)
                    for idx, item in enumerate(adm[3]):
                        loss_multi_target[0][idx] = item

                    result, loss_ddi = pretrained_model(seq_input)

                    loss_bce = F.binary_cross_entropy_with_logits(
                        result, torch.FloatTensor(loss_bce_target).to(device)
                    )
                    loss_multi = F.multilabel_margin_loss(
                        F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device)
                    )

                    loss = 0.97 * loss_bce + 0.03 * loss_multi


                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                llprint("\rtraining step: {} / {}".format(step, len(data_train)))
            print()
            tic2 = time.time()
            print("\n验证集结果：")
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_loss, avg_med = pretrain_eval(
                pretrained_model, data_eval, voc_size, device
            )
            print(
                "training time: {}, test time: {}".format(
                    time.time() - tic, time.time() - tic2
                )
            )

            if epoch != 0:
                if best_ja < ja:
                    best_epoch = epoch
                    best_ja = ja
                    best_model = pretrained_model
                print(
                    "best_epoch: {}, best_ja: {:.4}".format(best_epoch, best_ja))
        #
        torch.save(best_model.state_dict(), args.resume_path_pretrained)
        return best_model

def pretrain_eval(model, data_eval, voc_size, device):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]

    """自己加入一些loss来看"""
    loss_bce, loss_multi, loss = [[] for _ in range(3)]

    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input):
            target_output, _ = model(input[: adm_idx + 1])

            """自己加的loss，在输出时候用来看loss的改变，不训练"""
            loss_bce_target = np.zeros((1, voc_size[3]))
            loss_bce_target[:, adm[3]] = 1

            loss_multi_target = np.full((1, voc_size[3]), -1)
            for idx, item in enumerate(adm[3]):
                loss_multi_target[0][idx] = item

            with torch.no_grad():
                loss_bce1 = F.binary_cross_entropy_with_logits(
                    target_output, torch.FloatTensor(loss_bce_target).to(device)
                ).cpu()
                loss_multi1 = F.multilabel_margin_loss(
                    F.sigmoid(target_output), torch.LongTensor(loss_multi_target).to(device)
                ).cpu()
                loss1 = 0.95 * loss_bce1.item() + 0.05 * loss_multi1.item()

            loss_bce.append(loss_bce1)
            loss_multi.append(loss_multi1)
            loss.append(loss1)
            """"""

            y_gt_tmp = np.zeros(voc_size[3])
            y_gt_tmp[adm[3]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record)

    # """列表转np"""
    # loss_multi = np.array(loss_multi)
    # loss_bce = np.array(loss_bce)
    # loss = np.array(loss)

    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4},"
        "AVG_Loss: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            np.mean(loss),
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        np.mean(loss),
        med_cnt / visit_cnt,
    )

def eval_one_epoch(model, data_eval, voc_size, visit_embedding_table):
    model = model.eval()
    smm_record, ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(6)]
    med_cnt, visit_cnt = 0, 0
    # visit_i = 9332
    ddi_sum = 0
    for step, input_seq in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input_seq):
            output, _, _ = model(input_seq[:adm_idx + 1], visit_embedding_table)

            y_gt_tmp = np.zeros(voc_size[3])
            y_gt_tmp[adm[3]] = 1
            y_gt.append(y_gt_tmp)

            output = torch.sigmoid(output).detach().cpu().numpy()[0]
            y_pred_prob.append(output)

            y_pred_tmp = output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)
            # print(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            ddi_sum += abs(ddi_rate_score([[adm[3]]])-ddi_rate_score([y_pred_label]))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)
        # smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(step + 1, len(data_eval)))

    ddi_rate = (ddi_sum/visit_cnt)
    output_str = '\nDDI Rate: {:.4f}, Jaccard: {:.4f}, PRAUC: {:.4f}, ' + \
                 'AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n'
    llprint(output_str.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p),
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), \
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt

def Case(model, device, data_test, voc_size, visit_embedding_table,med_voc):
    model = model.to(device).eval()
    print('--------------------Begin Case--------------------')
    input_seq = data_test[0]
    print(input_seq)
    y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
    for adm_idx, adm in enumerate(input_seq):
        output, _, _ = model(input_seq[:adm_idx + 1], visit_embedding_table)
        output = torch.sigmoid(output).detach().cpu().numpy()[0]
        y_pred_prob.append(output)
        y_pred_tmp = output.copy()
        y_pred_tmp[y_pred_tmp >= 0.5] = 1
        y_pred_tmp[y_pred_tmp < 0.5] = 0
        y_pred.append(y_pred_tmp)

        y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
        y_pred_label.append(sorted(y_pred_label_tmp))
        gt = []
        predict = []
        print('--------------------ground truth--------------------')
        for i in adm[3]:
            gt.append(med_voc.idx2word[i])
        print(f"ground truth len: {len(gt)}")
        print(gt)
        print('--------------------predict--------------------')
        # print(y_pred_label)
        for i in y_pred_label[adm_idx]:
            predict.append(med_voc.idx2word[i])
        print(predict)
        print('--------------------ddi--------------------')
        print(ddi_rate_score([y_pred_label]))
        ground_truth_set = set(gt)
        predict_set = set(predict)

        # 计算correct, miss, unseen
        correct = ground_truth_set & predict_set
        miss = ground_truth_set - predict_set
        unseen = predict_set - ground_truth_set
        print(f"correct: {len(correct)}, {correct}")
        print(f"miss: {len(miss)}, {miss}")
        print(f"unseen: {len(unseen)}, {unseen}")

def Test(model, device, data_test, voc_size, visit_embedding_table):
    model = model.to(device).eval()
    print('--------------------Begin Testing--------------------')
    ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
    tic, result, sample_size = time.time(), [], round(len(data_test) * 0.8)
    np.random.seed(0)
    for _ in range(10):
        selected_indices = np.random.choice(len(data_test), size=round(len(data_test) * 0.8), replace=True)
        selected_indices_list = selected_indices.tolist()
        test_sample = [data_test[i] for i in selected_indices_list]
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval_one_epoch(model, test_sample, voc_size, visit_embedding_table)
        result.append([ja, ddi_rate, avg_f1, prauc, avg_med])
    result = np.array(result)
    mean, std = result.mean(axis=0), result.std(axis=0)
    metric_list = ['ja', 'ddi_rate', 'avg_f1', 'prauc', 'med']
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean, std))
    ])

    print(outstring)
    print('average test time: {}'.format((time.time() - tic) / 10))
    print('parameters', get_n_params(model))


def Train(model, device, data_train, data_eval, voc_size, visit_embedding_table, args):
    optimizer = Adam(model.parameters(), lr=args.lr)
    history = defaultdict(list)
    best = {"epoch": 0, "ja": 0, "ddi": 0, "prauc": 0, "f1": 0, "med": 0, 'model': model, 'table': visit_embedding_table}
    total_train_time, ddi_losses, ddi_values = 0, [], []

    EPOCH = args.epochs
    visit_embedding_table = torch.nn.Parameter(visit_embedding_table, requires_grad=False)
    if args.debug:
        EPOCH = 3
    for epoch in range(EPOCH):
        print(f'----------------Epoch {epoch + 1}------------------')
        model = model.train()
        visit_emb = []
        cu_iter = 0
        tic, ddi_losses_epoch = time.time(), []
        for step, input_seq in enumerate(data_train):
            for adm_idx, adm in enumerate(input_seq):
                cu_iter += 1
                seq_input = input_seq[: adm_idx + 1]
                adjust_learning_rate(optimizer, cu_iter, args, len(seq_input))
                bce_target = np.zeros((1, voc_size[3]))
                bce_target[:, adm[3]] = 1
                multi_target = np.full((1, voc_size[3]), -1)
                for idx, item in enumerate(adm[3]):
                    multi_target[0][idx] = item
                result, loss_ddi, p_emb = model(seq_input, visit_embedding_table)
                visit_emb.append(p_emb.detach().clone())
                sigmoid_res = torch.sigmoid(result)
                loss_bce = binary_cross_entropy_with_logits(result, torch.FloatTensor(bce_target).to(device))
                loss_multi = multilabel_margin_loss(sigmoid_res, torch.LongTensor(multi_target).to(device))
                result = sigmoid_res.detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = ddi_rate_score(
                    [[y_label]]
                )
                ddi_loss = F.mse_loss(torch.FloatTensor([current_ddi_rate]),torch.FloatTensor([ddi_rate_score([[adm[3]]])]))
                loss = 0.97 * loss_bce + 0.03 * loss_multi + ddi_loss
                ddi_losses_epoch.append(loss_ddi.detach().cpu().item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            llprint('\rtraining step: {} / {}'.format(step, len(data_train)))
        visit_embedding_table = torch.cat(visit_emb, dim=0)
        ddi_losses.append(sum(ddi_losses_epoch) / len(ddi_losses_epoch))
        print(f'\nddi_loss : {ddi_losses[-1]}\n')
        train_time, tic = time.time() - tic, time.time()
        total_train_time += train_time
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval_one_epoch(model, data_eval, voc_size, visit_embedding_table)
        print(f'training time: {train_time}, testing time: {time.time() - tic}')
        ddi_values.append(ddi_rate)
        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 5:
            print('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
            ))

        if epoch != 0:
            if best['ja'] < ja:
                best['epoch'] = epoch
                best['ja'] = ja
                best['model'] = copy.deepcopy(model)
                best['table'] = visit_embedding_table
                best['ddi'] = ddi_rate
                best['prauc'] = prauc
                best['f1'] = avg_f1
                best['med'] = avg_med
            print("best_epoch: {}, best_ja: {:.4f}".format(best['epoch'], best['ja']))

    print('avg training time/epoch: {:.4f}'.format(total_train_time / EPOCH))

    torch.save(best['model'].state_dict(), "/home/heyichen/RK-VSNet/saved/{}/trained_model_{:.4f}".format(args.dataset, best['ja']))
    torch.save(best['table'], "/home/heyichen/RK-VSNet/saved/{}/best_table_{:.4f}".format(args.dataset, best['ja']))
    return best['model'], best['table']

