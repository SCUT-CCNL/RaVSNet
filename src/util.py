import sys
import warnings

import dill
import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings('ignore')

dataset = 'mimic3'
# dataset = 'mimic4'


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


# use the same metric from DMNC


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def sequence_output_process(output_logits, filter_token):
    pind = np.argsort(output_logits, axis=-1)[:, ::-1]

    out_list = []
    break_flag = False
    for i in range(len(pind)):
        if break_flag:
            break
        for j in range(pind.shape[1]):
            label = pind[i][j]
            if label in filter_token:
                break_flag = True
                break
            if label not in out_list:
                out_list.append(label)
                break
    y_pred_prob_tmp = []
    for idx, item in enumerate(out_list):
        y_pred_prob_tmp.append(output_logits[idx, item])
    sorted_predict = [
        x for _, x in sorted(
            zip(y_pred_prob_tmp, out_list),
            reverse=True
        )
    ]
    return out_list, sorted_predict


def sequence_metric(y_gt, y_pred, y_prob, y_label):
    def average_prc(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if (average_prc[idx] + average_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(
                    2 * average_prc[idx] * average_recall[idx] /
                    (average_prc[idx] + average_recall[idx])
                )
        return score

    def jaccard(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_pred_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(
                y_gt[b], y_pred_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(
                y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob_label, k):
        precision = 0
        for i in range(len(y_gt)):
            TP = 0
            for j in y_prob_label[i][:k]:
                if y_gt[i, j] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_gt)

    try:
        auc = roc_auc(y_gt, y_prob)
    except ValueError:
        auc = 0
    p_1 = precision_at_k(y_gt, y_label, k=1)
    p_3 = precision_at_k(y_gt, y_label, k=3)
    p_5 = precision_at_k(y_gt, y_label, k=5)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def multi_label_metric(y_gt, y_pred, y_prob):
    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(
                    2 * average_prc[idx] * average_recall[idx] /
                    (average_prc[idx] + average_recall[idx])
                )
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(
                y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(
                y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    # roc_auc
    try:
        auc = roc_auc(y_gt, y_prob)
    except:
        auc = 0
    # precision
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    # macro f1
    f1 = f1(y_gt, y_pred)
    # precision
    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    # pre, recall, f1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    # jaccard
    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def ddi_rate_score(record, path="/home/heyichen/RK-VSNet/data/mimic3/outputs/ddi_A_final.pkl"):
    # ddi rate
    ddi_A = dill.load(open(path, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt


def buildPrjSmiles(molecule, med_voc, device="cpu:0"):
    average_index, smiles_all = [], []

    print(len(med_voc.items()))  # 131
    for index, ndc in med_voc.items():

        smilesList = list(molecule[ndc])

        """Create each data with the above defined functions."""
        counter = 0  # counter how many drugs are under that ATC-3
        for smiles in smilesList:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                smiles_all.append(smiles)
                counter += 1
            else:
                continue
                # print('[SMILES]', smiles)
                # print('[Error] Invalid smiles')
        average_index.append(counter)

        """Transform the above each data of numpy
        to pytorch tensor on a device (i.e., CPU or GPU).
        """
    # transform into projection matrix
    n_col = sum(average_index)
    n_row = len(average_index)

    average_projection = np.zeros((n_row, n_col))
    col_counter = 0
    for i, item in enumerate(average_index):
        average_projection[i, col_counter: col_counter + item] = 1 / item
        col_counter += item

    # print("Smiles Num:{}".format(len(smiles_all)))
    # print("n_col:{}".format(n_col))
    # print("n_row:{}".format(n_row))

    binary_projection = np.where(average_projection != 0, 1, 0)

    # return binary_projection, torch.FloatTensor(average_projection), smiles_all
    return binary_projection, average_projection, smiles_all


def parameter_report(best, regular):
    """输出一个参数的文件"""
    with open(f"/home/heycihen/RK-VSNet/saved/{dataset}/parameter_report.txt", 'w+') as f:
        f.write(
            "best eval:\n epoch:{},jaccard:{:.4f},ddi:{:.4f},prauc:{:.4f},f1:{:.4f},med:{:.4f}\n".
            format(best['epoch'], best['ja'], best['ddi'], best['prauc'], best['f1'], best['med']))
        for name, w in regular.get_weight(best['model']):
            f.write(name + str(w) + "\n")


def overfit_report(history, history_on_train):
    """输出对过拟合的发现"""
    # 画ja的图
    max_eval_ja, max_eval_ja_index = max(history["ja"]), history["ja"].index(max(history["ja"]))
    max_train_ja, max_train_ja_index = max(history_on_train["ja"]), history_on_train["ja"].index(
        max(history_on_train["ja"]))

    # 创建一个新的图形
    plt.figure()

    # 绘制每个列表的线
    plt.plot(history["ja"], label="eval")
    plt.plot(history_on_train["ja"], label="train")

    # 在每个最大值的节点旁边标注其对应的值
    plt.text(max_eval_ja_index, max_eval_ja, str(max_eval_ja), color='red', ha='center', va='bottom')
    plt.text(max_train_ja_index, max_train_ja, str(max_train_ja), color='red', ha='center', va='bottom')

    # 添加标题和标签
    plt.title("OverFitting-ja")
    plt.xlabel('epoch')
    plt.ylabel('ja_value')
    # 显示图例
    plt.legend()
    plt.savefig('results/ja.png')
    # 做的loss系列
    # 创建一个新的图形
    plt.figure()

    # 绘制每个列表的线
    plt.plot(history["loss"], label="eval")
    plt.plot(history_on_train["loss"], label="train")

    # 添加标题和标签
    plt.title('OverFitting-loss')
    plt.xlabel('epoch')
    plt.ylabel('results/loss_value')

    # 显示图例
    plt.legend()
    plt.savefig('results/loss.png')
    # 显示图形
    plt.show()


def graph_report(history):
    """图像方式输出结果"""
    # 为了合并两个图形，我们将使用1行2列的子图布局
    plt.figure(figsize=(12, 6))  # 设置图形的大小，可以根据需要进行调整

    max_eval_ja, max_eval_ja_index = max(history["ja"]), history["ja"].index(max(history["ja"]))
    plt.plot(history["ja"], label="eval")
    plt.text(max_eval_ja_index, max_eval_ja, "{:.4f}".format(max_eval_ja), color='red', ha='center', va='bottom')
    plt.title("jaccard")
    plt.xlabel('epoch')
    plt.ylabel('jaccard_value')
    plt.legend()
    plt.savefig(f'/data1/home/heyichen/RK-VSNet/saved/{dataset}/jaccard.png')


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        # self.weight_info(self.weight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            weight = (name, param)
            weight_list.append(weight)
            # if 'weight' in name:
            #     weight = (name, param)
            #     weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")

def relevance_mining(records, voc_size):
    # 药物-所有的ehr
    ehr_adj_med_diag = np.zeros((voc_size[3], voc_size[0]))
    ehr_adj_med_proc = np.zeros((voc_size[3], voc_size[1]))
    ehr_adj_med_sym = np.zeros((voc_size[3], voc_size[2]))
    ehr_adj_med_med = np.zeros((voc_size[3], voc_size[3]))


    for patient in records:
        for adm in patient:
            for i in range(len(adm[3])):
                for j in range(len(adm[0])):
                    node1 = adm[3][i]
                    node2 = adm[0][j]
                    ehr_adj_med_diag[node1, node2] += 1        #med-diag共现矩阵
                for k in range(len(adm[1])):
                    node1 = adm[3][i]
                    node2 = adm[1][k]
                    ehr_adj_med_proc[node1, node2] += 1        #med-proc共现矩阵
                for m in range(len(adm[2])):
                    node1 = adm[3][i]
                    node2 = adm[2][m]
                    ehr_adj_med_sym[node1, node2] += 1        #med-proc共现矩阵
                for l in range(i + 1, len(adm[3])):
                    node1 = adm[3][i]
                    node2 = adm[3][l]
                    ehr_adj_med_med[node1, node2] += 1
                    ehr_adj_med_med[node2, node1] += 1         #med-med共现矩阵

    return ehr_adj_med_diag, ehr_adj_med_proc, ehr_adj_med_med, ehr_adj_med_sym

def adj_matrix(matrix):
    original_matrix = torch.tensor(matrix)
    transposed_matrix = original_matrix.t()

    m, n = original_matrix.shape
    adj_matrix = torch.zeros((m + n, m + n), dtype=original_matrix.dtype)

    adj_matrix[:m, :n] = original_matrix
    adj_matrix[m:, n:] = transposed_matrix

    return adj_matrix

def lr_poly(base_lr, iter, max_iter, power, current_length):
    # ratio_length = 1 - (float(current_length) / 30)
    # iter = iter + ratio_length
    iter = iter + current_length
    if iter > max_iter:
        iter = iter % max_iter
    return base_lr * ((1 - float(iter) / max_iter) ** (power))#+ (float(current_length) / 30) ** (power))
    # return base_lr * (((1 - float(iter) / max_iter) ** (power))+ 0.1*((1 - (float(current_length) / 30) ** (power))))

def adjust_learning_rate(optimizer, i_iter, args, current_length):
    lr = lr_poly(args.lr, i_iter, 50000, 0.9, current_length)
    optimizer.param_groups[0]['lr'] = np.min(np.around(lr,8))
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def f1(y_true_hot, y_pred, metrics='weighted'):
    result = np.zeros_like(y_true_hot)
    for i in range(len(result)):
        true_number = np.sum(y_true_hot[i] == 1)
        result[i][y_pred[i][:true_number]] = 1
    return f1_score(y_true=y_true_hot, y_pred=result, average=metrics, zero_division=0)

def top_k_prec_recall(y_true_hot, y_pred, ks):
    a = np.zeros((len(ks),))
    r = np.zeros((len(ks),))
    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        for i, k in enumerate(ks):
            p = set(pred[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            # r[i] += len(it) / min(k, len(t))
            r[i] += len(it) / len(t)
    return a / len(y_true_hot), r / len(y_true_hot)

def evaluate_fn(model, data_eval, voc_size, visit_embedding_table):
    model.eval()
    total_loss = 0.0
    labels = []
    preds = []
    for step, patient in enumerate(data_eval):
        # y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        output, _, _ = model(patient[:],visit_embedding_table,4000)
        output = torch.sigmoid(output)
        pred = torch.argsort(output, dim=-1, descending=True)
        # print(pred)
        # print(pred.shape)
        preds.append(pred)
        # print(preds)
        bce_target = np.zeros((1, voc_size))
        bce_target[:, patient[-1][0]] = 1
        labels.append(bce_target)
    labels = np.vstack(labels)
    # print(labels)
    # print(labels.shape)
    # for i, array in enumerate(labels):
    #     print(f"Array {i + 1} shape: {array.shape}")
    preds = torch.vstack(preds).detach().cpu().numpy()
    # print(preds.shape)
    f1_score = f1(labels, preds)
    prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])
    print('\r    Evaluation: --- f1_score: %.4f --- top_k_recall: %.4f, %.4f, %.4f, %.4f'
            % (f1_score, recall[0], recall[1], recall[2], recall[3]))
    return f1_score, prec, recall
