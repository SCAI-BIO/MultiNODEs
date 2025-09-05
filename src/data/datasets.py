import numpy as np
import torch
from torch.utils.data import Dataset


def make_list_static_BN(static_types, static_true_miss_mask):

    s_onehot_types = []
    for i in range(static_types.shape[0]):
        for j in range(static_types[i, 1]):
            s_onehot_types.append(static_types[i])
    s_onehot_types = np.array(s_onehot_types)

    s_onehot_missing = []
    for i in range(static_true_miss_mask.shape[1]):
        for j in range(static_types[i, 1]):
            s_onehot_missing.append(static_true_miss_mask[:, i])
    s_onehot_missing = torch.stack(s_onehot_missing)
    s_onehot_missing = torch.transpose(s_onehot_missing, 0, 1)

    return s_onehot_types, s_onehot_missing


class PPMI_Dataset(Dataset):
    def __init__(self, config, data):

        # X, W ---> Patients, num_visits, num_long_var
        self.X, self.W, self.T = data[0], data[1], data[2]
        self.static_onehot = data[3]
        self.static_types = data[4]
        self.static_true_miss_mask = data[5]

        self.static = config.static_data

        if config.batch_norm_static:
            self.bn_s = True
            data_ = make_list_static_BN(self.static_types,
                self.static_true_miss_mask)

            self.s_onehot_types =  data_[0]
            self.s_onehot_missing = data_[1]
        else:
            self.bn_s = False

        self.var_names_static = ["CSF_Abeta.42_VIS00", "CSF_p.Tau181P_VIS00",
            "CSF_Total.tau_VIS00","CSF_tTau.Abeta_VIS00","CSF_pTau.Abeta_VIS00",
            "CSF_pTau.tTau_VIS00","Biological_ALDH1A1..rep.1._VIS00","Biological_ALDH1A1..rep.2._VIS00",
            "Biological_GAPDH..rep.1._VIS00","Biological_GAPDH..rep.2._VIS00","Biological_HSPA8..rep.1._VIS00",
            "Biological_HSPA8..rep.2._VIS00","Biological_LAMB2..rep.1._VIS00","Biological_LAMB2..rep.2._VIS00",
            "Biological_PGK1..rep.1._VIS00","Biological_PGK1..rep.2._VIS00","Biological_PSMC4..rep.1._VIS00",
            "Biological_PSMC4..rep.2._VIS00","Biological_SKP1..rep.1._VIS00","Biological_SKP1..rep.2._VIS00",
            "Biological_UBE2K..rep.1._VIS00","Biological_UBE2K..rep.2._VIS00","Biological_Serum.IGF.1_VIS00",
            "PatDemo_HISPLAT","PatDemo_RAINDALS","PatDemo_RAASIAN","PatDemo_RABLACK","PatDemo_RAWHITE",
            "PatDemo_RANOS","PatDemo_EDUCYRS","PatDemo_HANDED","PatDemo_GENDER","PatPDHist_BIOMOMPD",
            "PatPDHist_BIODADPD","PatPDHist_FULSIBPD","PatPDHist_MAGPARPD","PatPDHist_PAGPARPD",
            "PatPDHist_MATAUPD","PatPDHist_PATAUPD","SA_Imaging_VIS00","SA_Enrollment_Age",
            "SA_CADD_filtered_impact_scores_VIS00","SA_Polygenetic_risk_scores_VIS00"]

        self.var_names_long = ["MedicalHistory_WGTKG","MedicalHistory_HTCM","MedicalHistory_TEMPC",
            "MedicalHistory_SYSSUP","MedicalHistory_DIASUP","MedicalHistory_HRSUP","MedicalHistory_SYSSTND",
            "MedicalHistory_DIASTND","MedicalHistory_HRSTND","NonMotor_DVT_TOTAL_RECALL","NonMotor_DVS_LNS",
            "NonMotor_BJLOT","NonMotor_ESS","NonMotor_GDS","NonMotor_QUIP","NonMotor_RBD","NonMotor_SCOPA",
            "NonMotor_SFT","NonMotor_STA","NonMotor_STAI.State","NonMotor_STAI.Trait","UPDRS_UPDRS1",
            "UPDRS_UPDRS2","UPDRS_UPDRS3","SA_CSF_CSF.Alpha.synuclein"]

    def __getitem__(self, idx):

        X, W = self.X[idx, :], self.W[idx, :]
        if self.static:
            S_OneHot = self.static_onehot[idx, :]
            S_True_MMask = self.static_true_miss_mask[idx, :]
        else:
            return X, W

        if self.bn_s:
            s_ohm = self.s_onehot_missing[idx, :]
            return X, W, S_OneHot.float(), S_True_MMask, s_ohm
        else:
            return X, W, S_OneHot.float(), S_True_MMask

    def __len__(self):
        return len(self.X)

    def get_T(self):
        return self.T

    def get_XW(self):
        return self.X, self.W

    def get_static(self):
        return self.static_onehot.float(), self.static_types, self.static_true_miss_mask

    def get_onehot_static(self):
        return self.s_onehot_types, self.s_onehot_missing

    def get_var_names(self):
        return self.var_names_long, self.var_names_static
