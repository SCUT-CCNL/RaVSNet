import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .SetTransformer import ScaledDotProductAttention, SAB, MAB
from .homo_relation_graph import homo_relation_graph

import math

from torch.nn.parameter import Parameter


class Aggregation(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(Aggregation, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU()
        )

        self.gate_layer = nn.Linear(64, 1)

    def forward(self, seqs):
        gates = self.gate_layer(self.h1(seqs))
        output = F.sigmoid(gates)

        return output

class CausaltyReview(nn.Module):
    def __init__(self, casual_graph, num_med):
        super(CausaltyReview, self).__init__()

        self.num_med = num_med
        self.c1 = casual_graph
        diag_med_high = casual_graph.get_threshold_effect(0.97, "Diag", "Med")
        diag_med_low = casual_graph.get_threshold_effect(0.90, "Diag", "Med")
        proc_med_high = casual_graph.get_threshold_effect(0.97, "Proc", "Med")
        proc_med_low = casual_graph.get_threshold_effect(0.90, "Proc", "Med")
        sym_med_high = casual_graph.get_threshold_effect(0.97, "Sym", "Med")
        sym_med_low = casual_graph.get_threshold_effect(0.90, "Sym", "Med")
        self.c1_high_limit = nn.Parameter(torch.tensor([diag_med_high, proc_med_high, sym_med_high]))  # 选用的97%
        self.c1_low_limit = nn.Parameter(torch.tensor([diag_med_low, proc_med_low, sym_med_low]))  # 选用的90%
        self.c1_minus_weight = nn.Parameter(torch.tensor(0.01))
        self.c1_plus_weight = nn.Parameter(torch.tensor(0.01))

    def forward(self, pre_prob, diags, procs, syms):
        reviewed_prob = pre_prob.clone()

        for m in range(self.num_med):
            max_cdm = 0.0
            max_cpm = 0.0
            max_csm = 0.0
            for d in diags:
                cdm = self.c1.get_effect(d, m, "Diag", "Med")
                max_cdm = max(max_cdm, cdm)
            for p in procs:
                cpm = self.c1.get_effect(p, m, "Proc", "Med")
                max_cpm = max(max_cpm, cpm)
            for s in syms:
                csm = self.c1.get_effect(s, m, "Sym", "Med")
                max_csm = max(max_csm, csm)
            if max_cdm < self.c1_low_limit[0] and max_cpm < self.c1_low_limit[1] and max_csm < self.c1_low_limit[2]:

                reviewed_prob[0, m] -= self.c1_minus_weight
            elif max_cdm > self.c1_high_limit[0] or max_cpm > self.c1_high_limit[1] or max_csm > self.c1_high_limit[2]:
                reviewed_prob[0, m] += self.c1_plus_weight

        return reviewed_prob

class BasicModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            ddi_adj,
            emb_dim=256,
            device=torch.device("cpu:0"),
    ):
        super(BasicModel, self).__init__()

        self.device = device
        self.emb_dim = emb_dim

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(4)]
        )
        self.emb_fuse_weight = nn.Embedding(4, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.query = nn.Linear(4 * emb_dim, vocab_size[3])

        # graphs, bipartite matrix
        self.tensor_ddi_adj = ddi_adj
        self.init_weights()

    def forward(self, patient):
        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        adm = patient[-1]
        i1 = sum_embedding(
            self.dropout(
                self.embeddings[0](
                    torch.LongTensor(adm[5]).unsqueeze(dim=0).to(self.device)
                )
            )
        )  # (1,1,dim)
        i2 = sum_embedding(
            self.dropout(
                self.embeddings[1](
                    torch.LongTensor(adm[6]).unsqueeze(dim=0).to(self.device)
                )
            )
        )
        i3 = self.dropout(self.embeddings[2](torch.LongTensor(adm[7]).unsqueeze(dim=0).to(self.device)))
        i3 = sum_embedding(i3)

        if adm == patient[0]:
            i4 = torch.zeros(1, 1, self.emb_dim).to(self.device)
        else:
            adm_last = patient[-2]
            i4 = sum_embedding(self.dropout(self.embeddings[3](torch.LongTensor(adm_last[8]).unsqueeze(dim=0).to(self.device))))

        emb_fuse_weight = self.emb_fuse_weight(torch.tensor([0, 1, 2, 3]).to(self.device))
        patient_representations = torch.cat(
            [i1 * emb_fuse_weight[0], i2 * emb_fuse_weight[1], i3 * emb_fuse_weight[2], i4 * emb_fuse_weight[3]],
            dim=-1).squeeze(0)
        result = self.query(patient_representations)  # (1, dim)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx



class RaVSNet(nn.Module):
    def __init__(
            self,
            causal_graph,
            vocab_size,
            ddi_adj,
            ehr_adj_med_diag,
            ehr_adj_med_proc,
            ehr_adj_med_med,
            ehr_adj_med_sym,
            medication_list,
            pretrained_embedding,
            emb_dim=256,
            device=torch.device("cuda:1"),
    ):
        super(RaVSNet, self).__init__()

        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.medication_list = medication_list
        # self.average_projection = average_projection

        self.causal_graph = causal_graph

        # self.pretrained_embedding = pretrained_embedding
        self.embeddings = nn.ModuleList(
            pretrained_embedding
        )
        self.homo_graph = nn.ModuleList([
            homo_relation_graph(emb_dim, device),
            homo_relation_graph(emb_dim, device),
            homo_relation_graph(emb_dim, device),
            homo_relation_graph(emb_dim, device)
        ])

        self.emb_fuse_weight = nn.Embedding(4, 1)
        self.cross_att = ScaledDotProductAttention(4 * emb_dim, 4 * emb_dim, emb_dim, 4)
        self.drug_output = nn.Linear(emb_dim, emb_dim)
        self.drug_layernorm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.diag_gcn = GCN(voc_size=vocab_size[0] + vocab_size[3], emb_dim=emb_dim, adj=ehr_adj_med_diag,
                            device=device)
        self.proc_gcn = GCN(voc_size=vocab_size[1] + vocab_size[3], emb_dim=emb_dim, adj=ehr_adj_med_proc,
                            device=device)
        self.sym_gcn = GCN(voc_size=vocab_size[2] + vocab_size[3], emb_dim=emb_dim, adj=ehr_adj_med_sym,
                           device=device)
        self.med_gcn = GCN(voc_size=vocab_size[3], emb_dim=emb_dim, adj=ehr_adj_med_med, device=device)

        self.mab1 = MAB(emb_dim, emb_dim, emb_dim, 2, use_ln=True)
        self.mab2 = MAB(emb_dim, emb_dim, emb_dim, 2, use_ln=True)
        self.mab3 = MAB(emb_dim, emb_dim, emb_dim, 2, use_ln=True)
        self.mab4 = MAB(emb_dim, emb_dim, emb_dim, 2, use_ln=True)
        self.mab5 = MAB(emb_dim, emb_dim, emb_dim, 2, use_ln=True)
        self.mab6 = MAB(emb_dim, emb_dim, emb_dim, 2, use_ln=True)
        self.sab1 = SAB(emb_dim, emb_dim, 2, use_ln=True)
        self.sab2 = SAB(emb_dim, emb_dim, 2, use_ln=True)
        self.sab3 = SAB(emb_dim, emb_dim, 2, use_ln=True)
        self.sab4 = SAB(emb_dim, emb_dim, 2, use_ln=True)
        self.sab5 = SAB(emb_dim, emb_dim, 2, use_ln=True)
        self.sab6 = SAB(emb_dim, emb_dim, 2, use_ln=True)


        self.pat_fuse = nn.Linear(9 * emb_dim, emb_dim)
        self.med_fuse = nn.Linear(6 * emb_dim, emb_dim)
        self.fuse_weight = nn.Embedding(2, 1)
        self.recomb = nn.Sequential(
            nn.ReLU(),
            # nn.Linear(emb_dim, vocab_size[3])
            nn.Linear(2 * vocab_size[3], vocab_size[3])
        )
        self.recomd = nn.Sequential(
            nn.ReLU(),
            nn.Linear(9 * emb_dim, vocab_size[3])
        )
        self.docter_weight = nn.Embedding(2, vocab_size[3])
        self.review = CausaltyReview(self.causal_graph, vocab_size[3])
        self.tensor_ddi_adj = ddi_adj
        self.ehr_adj_med_diag = ehr_adj_med_diag
        self.ehr_adj_med_proc = ehr_adj_med_proc
        self.ehr_adj_med_sym = ehr_adj_med_sym
        self.ehr_adj_med_med = ehr_adj_med_med

        self.MLP_layer1 = nn.Linear(emb_dim * 4,1)
        self.MLP_layer2 = nn.Linear(2,1)
        self.gumbel_tau = 0.3
        self.att_tau = 20
        self.linear_layer = nn.Linear(4*emb_dim, 4*emb_dim)

    """ calculate target-aware attention """

    def calc_cross_visit_scores(self, embedding):
        """ embedding: (batch * visit_num * emb) """

        # Extract the current att value when calculating attention
        diag_keys = embedding[:, :]  # key: past visits and current visit
        diag_query = embedding[-1:,:]  # query: current visit
        diag_scores = torch.mm(self.linear_layer(diag_query), diag_keys.transpose(0, 1)) / math.sqrt(
            diag_query.size(-1))  # attention weight
        diag_scores_encoder = diag_scores
        scores = F.softmax(diag_scores / self.att_tau, dim=-1)
        scores_encoder = F.softmax(diag_scores_encoder / self.att_tau, dim=-1)
        return scores, scores_encoder

    def forward(self, patient, visit_embedding_table):
        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(0)  # (1,1,dim)

        diag_ehr = self.diag_gcn.forward()
        proc_ehr = self.proc_gcn.forward()
        med_ehr = self.med_gcn.forward()
        sym_ehr = self.sym_gcn.forward()


        diag_drug_emb = diag_ehr[:self.vocab_size[3]].unsqueeze(0)
        proc_drug_emb = proc_ehr[:self.vocab_size[3]].unsqueeze(0)
        med_drug_emb = med_ehr.unsqueeze(0)
        sym_drug_emb = sym_ehr[:self.vocab_size[3]].unsqueeze(0)

        d_d_p = self.mab1(diag_drug_emb, proc_drug_emb)
        d_d_p = self.sab1(d_d_p)
        d_d_m = self.mab2(diag_drug_emb, med_drug_emb)
        d_d_m = self.sab2(d_d_m)
        d_d_s = self.mab3(diag_drug_emb, sym_drug_emb)
        d_d_s = self.sab3(d_d_s)

        p_d_m = self.mab4(proc_drug_emb, med_drug_emb)
        p_d_m = self.sab4(p_d_m)
        p_d_s = self.mab5(proc_drug_emb, sym_drug_emb)
        p_d_s = self.sab5(p_d_s)

        m_d_s = self.mab6(med_drug_emb, sym_drug_emb)
        m_d_s = self.sab6(m_d_s)

        gl = torch.cat([d_d_p, d_d_m, d_d_s, p_d_m, p_d_s, m_d_s], dim=-1)  # (1,med_num,dim*6)
        medication = self.med_fuse(gl)  # (med_num,dim)
        i1_seq, i2_seq, i3_seq, i4_seq = [],[],[],[]
        for adm_id, adm in enumerate(patient):
            i1 = self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))
            graph_diag = self.causal_graph.get_graph(adm[4], "Diag")
            emb_diag = self.homo_graph[0](graph_diag, i1)

            i2 = self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)))
            graph_proc = self.causal_graph.get_graph(adm[4], "Proc")
            emb_proc = self.homo_graph[1](graph_proc, i2)

            i3 = self.dropout(self.embeddings[2](torch.LongTensor(adm[2]).unsqueeze(dim=0).to(self.device)))
            graph_sym = self.causal_graph.get_graph(adm[4], "Sym")
            emb_sym = self.homo_graph[2](graph_sym, i3)

            if adm == patient[0]:
                emb_med = torch.zeros(1, 1, self.emb_dim).to(self.device)
            else:
                adm_last = patient[adm_id - 1]
                i4 = self.dropout(self.embeddings[3](torch.LongTensor(adm_last[3]).unsqueeze(dim=0).to(self.device)))
                med_graph = self.causal_graph.get_graph(adm_last[4], "Med")
                emb_med = self.homo_graph[3](med_graph, i4)

            i1_seq.append(torch.sum(emb_diag, keepdim=True, dim=1))
            i2_seq.append(torch.sum(emb_proc, keepdim=True, dim=1))
            i3_seq.append(torch.sum(emb_sym, keepdim=True, dim=1))
            i4_seq.append(torch.sum(emb_med, keepdim=True, dim=1))

        i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)
        i3_seq = torch.cat(i3_seq, dim=1)  # (1,seq,dim)
        i4_seq = torch.cat(i4_seq, dim=1)  # (1,seq,dim)
        if len(patient) >= 2:
            #当前健康嵌入
            patient_representation = torch.concatenate([i1_seq, i2_seq, i3_seq, i4_seq], dim=-1).squeeze(dim=0)   #(seq,dim*4)
            input1 = self.MLP_layer1(patient_representation) #(seq,1)
            cur_query = input1[-1:,:]
            cur_query = cur_query.repeat(input1.size()[0], 1)
            concat_query = torch.cat([input1, cur_query], dim=-1)  # (seq,2)
            concat_query = torch.sigmoid(self.MLP_layer2(concat_query))
            gumbel_input = torch.cat([concat_query, 1 - concat_query], dim=-1)
            pre_gumbel = F.gumbel_softmax(gumbel_input, tau=self.gumbel_tau, hard=True)[:, 0]
            gumbel = torch.cat([pre_gumbel[:-1], torch.ones(1, device = self.device)])   #保证最后一次选中
            i1_seq = gumbel.unsqueeze(0).unsqueeze(-1)*i1_seq
            i2_seq = gumbel.unsqueeze(0).unsqueeze(-1) * i2_seq
            i3_seq = gumbel.unsqueeze(0).unsqueeze(-1) * i3_seq
            i4_seq = gumbel.unsqueeze(0).unsqueeze(-1) * i4_seq
        visit_diag_embedding = torch.concatenate([i1_seq, i2_seq, i3_seq, i4_seq], dim=-1).squeeze(dim=0)
        cross_visit_scores, scores_encoder = self.calc_cross_visit_scores(visit_diag_embedding)
        visit_diag_embedding = visit_diag_embedding * cross_visit_scores.T
        patient_representations = torch.sum(visit_diag_embedding, dim=0, keepdim=True)

        visit_j = torch.cat((visit_embedding_table[:adm[4]], visit_embedding_table[adm[4] + 1:]), dim=0)
        similar_score = torch.cosine_similarity(patient_representations, visit_j, dim=1)

        top_score, top_indices = torch.topk(similar_score, k=10)
        new_indices = []
        similar_patient_list = []
        for i in top_indices:
            similar_patient_list.append(visit_j[i.item()].unsqueeze(dim=0))
            if i.item() >= adm[4]:
                new_indices.append(i.item() + 1)
            else:
                new_indices.append(i.item())
        sim_patient = torch.cat(similar_patient_list, dim=0).unsqueeze(dim=0)  # (1,topk,4*dim)
        sim_patient = self.cross_att(patient_representations.unsqueeze(dim=0), sim_patient, sim_patient)  # (1,1,4*dim)
        patient_emb = torch.cat([patient_representations, sim_patient.squeeze(dim=0)], dim=-1)  # (1,12*dim)

        similar_medication = [self.medication_list[i] for i in new_indices]
        similar_medication_list = []
        for i in similar_medication:
            sm = self.dropout(
                self.embeddings[3](torch.LongTensor(i).unsqueeze(dim=0).to(self.device)))  # (1,1,dim)
            sm = sum_embedding(sm)
            similar_medication_list.append(sm)

        sim_med = torch.cat(similar_medication_list, dim=1).squeeze(dim=0)  # (10,dim)
        sim_med = torch.mm(top_score.unsqueeze(dim=0), sim_med)
        sim_med = self.drug_layernorm(sim_med + self.drug_output(sim_med))
        patient_emb = torch.cat([patient_emb, sim_med], dim=-1)  # (9,dim)

        patient_fuse = self.pat_fuse(patient_emb)
        fuse_weight = self.fuse_weight(torch.tensor([0, 1]).to(self.device))
        docter_weight = self.docter_weight(torch.tensor([0, 1]).to(self.device))
        similar = torch.cosine_similarity(patient_fuse, medication.squeeze(0), dim=1).unsqueeze(0)
        docter_direct = self.recomd(patient_emb)
        docter_recomb = self.recomb(torch.cat([docter_direct * fuse_weight[0], similar * fuse_weight[1]], dim=1))
        result = docter_direct * docter_weight[0] + docter_recomb * docter_weight[1]
        result = self.review(result, patient[-1][0], patient[-1][1], patient[-1][2])
        ehr_adj_med_diag = self.ehr_adj_med_diag[:self.vocab_size[3], :self.vocab_size[0]].t().to(self.device)
        md = torch.sum(ehr_adj_med_diag[adm[0]], keepdim=True, dim=0)
        ehr_adj_med_diag = self.ehr_adj_med_diag[:self.vocab_size[3], :self.vocab_size[1]].t().to(self.device)
        mp = torch.sum(ehr_adj_med_diag[adm[1]], keepdim=True, dim=0)
        ehr_adj_med_diag = self.ehr_adj_med_diag[:self.vocab_size[3], :self.vocab_size[2]].t().to(self.device)
        ms = torch.sum(ehr_adj_med_diag[adm[2]], keepdim=True, dim=0)
        result += F.sigmoid(md + mp + ms)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size[2], voc_size[2])
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg, patient_representations

