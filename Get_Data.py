import numpy as np
import torch
from pathlib import Path
import pickle
import mat73
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.signal import chirp
from utils import get_argparser
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm, colors, pyplot as plt
import scipy.io as sio
from torch.autograd import Variable


from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

parser = get_argparser()
args = parser.parse_args()

device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

class Datasets(object):

    def __init__(self,fold):
        self.fold = fold

    def HCP(self):

        path = Path('/data/users4/ziqbal5/DataSets/HCP1200realease/HCP830.pickle')
        if path.is_file() == True:
            with open(path, "rb") as infile:
                data = pickle.load(infile)     

        else:
        
            AllData = []
            filename2 = '/data/users4/ziqbal5/DataSets/HCP1200realease/correct_indices_HCP.csv'
            cor_ind = pd.read_csv(filename2, header=None)
            cor_ind = cor_ind.to_numpy()

            
            for i in tqdm(range(1, 834)):
                
                filename = '/data/qneuromark/Results/ICA/HCP/REST1_LR/HCP1_ica_c'+str(i)+'-1.mat'
                data = sio.loadmat(filename)
                data = data['tc']   #1200x100
                data = data.transpose()  #100x1200
                arr = []    
                for j in range(0,len(cor_ind)):
                    arr.append(data[int(cor_ind[j])-1])
                arr = np.array(arr)
                if arr.shape[1] < 1200:
                    print("Index ",i, "doesn't have full time steps: ", arr.shape[1])
                    continue
                AllData.append(arr)

            print('All subjects loaded successfully...')
            data = np.stack(AllData, axis=0)
            #print(data.shape)
            # with open(path, "wb") as outfile:
            #     pickle.dump(data, outfile)

        #labels
        X1 = torch.Tensor(data)
        L1 = np.zeros([len(data),])
        L1 = torch.tensor(L1)

        #Reverse Direction
        ccc = np.empty_like(X1)
        #ccc = np.empty([args.sasubjects, components, tp])
        for i in range(ccc.shape[0]):
            for j in range(ccc.shape[1]):
                ccc[i][j] = np.flip(data[i][j])
        print("Slices reversal is done")
        
        #print(X1[0][0])
        #print(ccc[0][0])
        X2 = torch.Tensor(ccc)
        L2 = np.ones([len(ccc),])
        L2 = torch.tensor(L2)
        data = torch.cat((X1, X2), 0)
        labels = torch.cat((L1, L2),0)
        return self.split_folds(data, labels)
     


    def FBIRN(self):
        path = Path('/data/users4/ziqbal5/DataSets/FBIRN/FBIRN_Pavel.h5')  #Pavel
        #path = Path('/data/users4/ziqbal5/DataSets/FBIRN/FBIRN.pickle')    #Old ones
        if path.is_file() == True:

            hf = h5py.File(path, "r")
            data = hf.get("FBIRN_dataset")
            data = np.array(data)
            print(data.shape)
            data = data.reshape(data.shape[0], 100, -1)

            # with open(path, "rb") as infile:
            #     data = pickle.load(infile)
            # print(data.shape)

        else:
            AllData = []
            for i in tqdm(range(1, args.samples+1)):
                filename = '/data/users2/zfu/Matlab/GSU/Neuromark/Results/ICA/FBIRN/FBIRN_ica_br'+str(i)+'.mat'
                data = mat73.loadmat(filename)
                data = data['compSet']['tc']
                data = data.T
                AllData.append(data[:, 0:args.tp])
            print('All subjects loaded successfully...')
            data = np.stack(AllData, axis=0)
            with open(path, "wb") as outfile:
                pickle.dump(data, outfile)

            

        ds = '/data/users4/ziqbal5/DataSets/FBIRN/transform_to_correct_GSP.csv'
        cor_ind = pd.read_csv(ds, header=None)
        cor_ind = cor_ind[1] # if you use 1 here then those are the indices coming from Dr. Fu.
        cor_ind = cor_ind.to_numpy()
        print("non-noise component indices used: ", cor_ind)
        

        data2 = np.zeros((args.samples, 53, args.tp))
        for i in range(len(data)):
            temp = data[i]
            data2[i] = temp[cor_ind]

        #import labels
        filename = '/data/users4/ziqbal5/DataSets/FBIRN/sub_info_FBIRN.mat'
        lab = mat73.loadmat(filename)
        ab = torch.empty((len(lab['analysis_SCORE']), 1))
        for i in range(len(lab['analysis_SCORE'])):
            a = lab['analysis_SCORE'][i][2]
            ab[i][0] = a
        labels = torch.squeeze(ab)

        labels[labels==1] = 0
        labels[labels==2] = 1
       
        print("HC: ", len(torch.argwhere(labels == 0)), "SZ: ",  len(torch.argwhere(labels == 1)))
        return self.split_folds(data2, labels)

    def COBRE(self):
        hf = h5py.File('/data/users4/ziqbal5/abc/MILC/data/COBRE_AllData.h5', 'r')
        data = hf.get('COBRE_dataset')
        data = np.array(data)
        data = data.reshape(len(data), 100, 140)
        
        ds = '/data/users4/ziqbal5/abc/MILC/data/bsnip/correct_indices_GSP.csv'
        cor_ind = pd.read_csv(ds, header=None)
        
        indices = pd.read_csv(ds, header=None)
        idx = indices[0].values
        print("non-noise component indices used: ", idx)
        data = data[:, idx, :]

        # fig = plt.figure(figsize=(20, 2))
        # plt.plot(data[0].T)
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('abc.png',transparent=True)
        # kckc
        filename = '/data/users4/ziqbal5/abc/MILC/data/labels_COBRE.csv'
        df = pd.read_csv(filename, header=None)
        all_labels = df.values
        all_labels = torch.from_numpy(all_labels).int()
        all_labels = all_labels.view(len(data))
        labels = all_labels - 1
        
        print("HC: ", len(np.where(labels == 0)[0]), "SZ: ",  len(np.argwhere(labels == 1)[0]))
        labels = torch.Tensor(labels)
        return self.split_folds(data, labels)
        
    def BSNIP(self):
        
        with np.load('/data/users4/ziqbal5/DataSets/bsnip/BSNIP_data.npz') as npzfile:
            data = npzfile['features']
            labels = npzfile['labels']


        #print(labels)
        #taking HC and SZ only. dropping the rest fo the classes
        ind = np.argwhere((labels== 0) | (labels == 1))

        data = data[ind]
        data = np.squeeze(data)
        labels = labels[ind]

        
        ds = '/data/users4/ziqbal5/DataSets/bsnip/transform_to_correct_GSP.csv'
        #cor_ind = pd.read_csv(ds, header=None)
        
        indices = pd.read_csv(ds, header=None)
        idx = indices[0].values
        print("non-noise component indices used: ",idx)
        data = data[:, idx, :]
        print(data.shape)
        print("HC: ", len(np.argwhere(labels == 0)), "SZ: ",  len(np.argwhere(labels == 1)))
        labels = torch.Tensor(labels)
        labels = torch.squeeze(labels)
        return self.split_folds(data, labels)

    def ADNI(self):
        
        with np.load('/data/users4/ziqbal5/DataSets/adni/ADNI_data_194.npz') as npzfile:
            data = npzfile['features']
            labels = npzfile['diagnoses']
            first_sessions = npzfile["early_indices"]

        indices_path: str = "/data/users4/ziqbal5/DataSets/adni/ICA_correct_order.csv"
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values
        print("non-noise component indices used: ",idx)

        only_first_sessions: bool = True
        multiclass: bool = False

        data = data[:, idx, :]

        if only_first_sessions:
            data = data[first_sessions, :, :]
            labels = labels[first_sessions]

        filter_array = []
        if multiclass:
            unique, counts = np.unique(labels, return_counts=True)
            counts = dict(zip(unique, counts))

            print(f"Number of classes in the data: {unique.shape[0]}")
            valid_labels = []
            for label, count in counts.items():
                if count > 10:
                    valid_labels += [label]
                else:
                    print(
                        f"There is not enough labels '{label}' in the dataset, filtering them out"
                    )

            if len(valid_labels) == unique.shape[0]:
                filter_array = [True] * labels.shape[0]
            else:
                for label in labels:
                    if label in valid_labels:
                        filter_array.append(True)
                    else:
                        filter_array.append(False)
        else:
            # leave subjects of class 0 and 1 only
            # {"Patient": 6, "LMCI": 2, "SMC": 5, "AD": 1, "EMCI": 4, "MCI": 3, "CN": 0}
            for label in labels:
                if label in (0, 1):
                    filter_array.append(True)
                else:
                    filter_array.append(False)

        data = data[filter_array, :, :]
        labels = labels[filter_array]

        unique = np.sort(np.unique(labels))
        shift_dict = dict(zip(unique, np.arange(unique.shape[0])))
        for i, _ in enumerate(labels):
            labels[i] = shift_dict[labels[i]]

        ####################

        #data = np.swapaxes(data, 1, 2)



        #print(labels)
        #taking HC and SZ only. dropping the rest fo the classes
        ind = np.argwhere((labels== 0) | (labels == 1))

        data = data[ind]
        data = np.squeeze(data)
        labels = labels[ind]
        labels = np.squeeze(labels)
        data, labels = self.data_balancing(data, labels, 'adni')
        print("HC: ", len(np.argwhere(labels == 0)), "AD: ",  len(np.argwhere(labels == 1)))
        labels = torch.Tensor(labels)
        labels = torch.squeeze(labels)
        return self.split_folds(data, labels)


    def data_balancing(self, data, labels, datasetname):
        hc_data = data[np.argwhere(labels == 0)].squeeze()
        ad_data = data[np.argwhere(labels == 1)].squeeze()
        print(f"HC data shape: {hc_data.shape}")
        print(f"AD data shape: {ad_data.shape}")
        print(type(hc_data))
   
     

        # Step 1: Extract embedding for each subject (mean over time)
        def extract_embedding(subject_data):
            return np.mean(subject_data, axis=-1)  # shape: (53,)

        X_hc = np.array([extract_embedding(subj) for subj in hc_data])  # (433, 53)
        X_ad = np.array([extract_embedding(subj) for subj in ad_data])  # (66, 53)

        # Step 2: Mahalanobis distance between HC and AD distribution
        mean_ad = np.mean(X_ad, axis=0)
        cov_ad = np.cov(X_ad, rowvar=False)
        inv_cov_ad = np.linalg.inv(cov_ad + 1e-6 * np.eye(cov_ad.shape[0]))  # regularized

        dists = np.array([
            distance.mahalanobis(x, mean_ad, inv_cov_ad)
            for x in X_hc
        ])  # shape: (433,)

        # Step 3: Select 100_adni/261_oasis HC closest to AD distribution
        if datasetname == 'adni':
            n_select = 100
        elif datasetname == 'oasis':
            n_select = 261
        selected_indices = np.argsort(dists)[:n_select]
        hc_selected_data = hc_data[selected_indices]  # shape: (66, 53, 199)

        # # Optional: PCA visualization
        # X_all = np.vstack([X_ad, X_hc, X_hc[selected_indices]])
        # labels = (
        #     ['AD'] * len(X_ad) +
        #     ['HC_rest'] * len(X_hc) +
        #     ['HC_selected'] * len(selected_indices)
        # )

        # pca = PCA(n_components=2)
        # X_2d = pca.fit_transform(X_all)

        # colors = {'AD': 'red', 'HC_rest': 'gray', 'HC_selected': 'green'}
        # for label in np.unique(labels):
        #     idx = [i for i, l in enumerate(labels) if l == label]
        #     plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=label, alpha=0.6, color=colors[label])

        # plt.legend()
        # plt.title("PCA: AD vs HC vs Selected HC")
        # plt.grid(True)
        # plt.xlabel("PC1")
        # plt.ylabel("PC2")
        # plt.savefig('abc.png')
        # # Already done PCA on the combined data:
     

        # # Get explained variance ratios
        # var_pc1 = pca.explained_variance_ratio_[0]
        # var_pc2 = pca.explained_variance_ratio_[1]

        # print(f"Variance explained by PC1: {var_pc1:.4f} ({var_pc1*100:.2f}%)")
        # print(f"Variance explained by PC2: {var_pc2:.4f} ({var_pc2*100:.2f}%)")


        # print(selected_indices)
        # Combine data
        combined_data = np.concatenate([hc_selected_data, ad_data], axis=0)  # shape: (132, 53, 199)

        # Create labels: 0 = HC, 1 = AD
        labels = np.array([0] * len(hc_selected_data) + [1] * len(ad_data))  # shape: (132,)
        combined_data, labels = shuffle(combined_data, labels, random_state=42)
        return combined_data, labels


        
    def OASIS(self):
        dataset_path: str = "/data/users4/ziqbal5/DataSets/oasis/OASIS3_AllData_allsessions.npz"
        indices_path: str = "/data/users4/ziqbal5/DataSets/oasis/ICA_correct_order.csv"
        labels_path: str = "/data/users4/ziqbal5/DataSets/oasis/labels_OASIS_6_classes.csv"
        sessions_path: str = "/data/users4/ziqbal5/DataSets/oasis/oasis_first_sessions_index.csv"
        multiclass: bool = False
        only_first_sessions: bool = True


        data = np.load(dataset_path)
        # 2826 - sessions - data.shape[0]
        # 100 - components - data.shape[1]
        # 160 - time points - data.shape[2]

        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values

        # filter the data: leave only correct components and the first 156 time points
        # (not all subjects have all 160 time points)
        data = data[:, idx, :156]
        # print(data.shape)
        # 53 - components - data.shape[1]
        # 156 - time points - data.shape[2]

        # get labels
        labels = pd.read_csv(labels_path, header=None)
        labels = labels.values.flatten().astype("int") - 1

        ####################
        ##### You can remove this block if you just need to load all data,
        ##### I needed this logic for my experiments

        if only_first_sessions:
            # leave only first sessions
            sessions = pd.read_csv(sessions_path, header=None)
            first_session = sessions[0].values - 1

            data = data[first_session, :, :]
            # 912 - sessions - data.shape[0] - only first session
            labels = labels[first_session]

        filter_array = []
        if multiclass:
            unique, counts = np.unique(labels, return_counts=True)
            counts = dict(zip(unique, counts))

            print(f"Number of classes in the data: {unique.shape[0]}")
            valid_labels = []
            for label, count in counts.items():
                if count > 10:
                    valid_labels += [label]
                else:
                    print(
                        f"There is not enough labels '{label}' in the dataset, filtering them out"
                    )

            if len(valid_labels) == unique.shape[0]:
                filter_array = [True] * labels.shape[0]
            else:
                for label in labels:
                    if label in valid_labels:
                        filter_array.append(True)
                    else:
                        filter_array.append(False)
        else:
            # leave subjects of class 0 and 1 only
            for label in labels:
                if label in (0, 1):
                    filter_array.append(True)
                else:
                    filter_array.append(False)

        data = data[filter_array, :, :]
        # 2559 - sessions - data.shape[0] - subjects of class 0 and 1
        # 823 - sessions - data.shape[0] - if only first sessions are considered
        labels = labels[filter_array]

        unique = np.sort(np.unique(labels))
        shift_dict = dict(zip(unique, np.arange(unique.shape[0])))
        for i, _ in enumerate(labels):
            labels[i] = shift_dict[labels[i]]
        data, labels = self.data_balancing(data, labels, 'oasis')
        print("HC: ", len(np.argwhere(labels == 0)), "AD: ",  len(np.argwhere(labels == 1)))
        labels = torch.from_numpy(labels)
        #print(type(data), type(labels), data.shape, labels.shape)
        return self.split_folds(data, labels)

    def ABIDE(self):
        
        data = np.load('/data/users4/ziqbal5/DataSets/abide869/ABIDE1_AllData_869Subjects_ICA.npz')
        print(len(data), data.shape)
        indices = pd.read_csv("/data/users4/ziqbal5/DataSets/adni/ICA_correct_order.csv", header=None)
        idx = indices[0].values
        
        print(idx)
        data = data[:, idx, :]

        labels = pd.read_csv('/data/users4/ziqbal5/DataSets/abide869/labels_ABIDE1_869Subjects.csv', header=None)
        labels = labels.values.flatten().astype("int") - 1
        print("HC: ", len(np.argwhere(labels == 0)), "Au: ",  len(np.argwhere(labels == 1)))
        labels = torch.from_numpy(labels)
        return self.split_folds(data, labels)


    def split_folds(self, data,labels):
        Adata = torch.Tensor(data)
        
        data = np.zeros((len(Adata), args.nw, 53, args.wsize))
        
        for i in range(len(Adata)):
            for j in range(args.nw):
                data[i, j, :, :] = Adata[i, :, (j * args.ws):(j * args.ws) + args.wsize]
        
        
        skf = StratifiedKFold(n_splits=5, random_state = 34, shuffle = True) #default 33
        skf.get_n_splits(data, labels)
        Folds_train_ind = []
        Folds_test_ind = []
        for i, (train_index, test_index) in enumerate(skf.split(data, labels)):
            Folds_train_ind.append(train_index)
            Folds_test_ind.append(test_index)
        
        #Folds_train_ind[self.fold], Folds_test_ind[self.fold]

        train_data, train_labels = data[Folds_train_ind[self.fold]], labels[Folds_train_ind[self.fold]]
        test_data, test_labels = data[Folds_test_ind[self.fold]], labels[Folds_test_ind[self.fold]] 
        tr_data, val_data, tr_labels, val_labels = train_test_split(train_data,train_labels , 
                                    random_state=100,  
                                    test_size=0.20,  
                                    shuffle=True, stratify=train_labels)
        
        return tr_data, tr_labels, val_data, val_labels,test_data, test_labels
        

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class LSTM_oldone(torch.nn.Module):
    #model = LSTM(X.shape[2], 256, 200, 121, 2, g)
    def __init__(self, enc_input_size, input_size, hidden_nodes, sequence_size, output_size, gain, onewindow):
        super(LSTM_oldone, self).__init__()
        self.sequence_size = sequence_size
        self.hidden = hidden_nodes
        self.onewindow = onewindow
        self.enc_out = input_size
        self.lstm = nn.LSTM(input_size, hidden_nodes, batch_first=True) #not using this for rnm. Fro rnn, using this and the encoder lstm layer as well.
        
        # input size for the top lstm is the hidden size for the lower
        
        self.encoder = nn.LSTM(enc_input_size, self.enc_out, batch_first = True)  #using only this for rnm

        # previously, I used 64
        self.attnenc = nn.Sequential(
                nn.Linear(self.enc_out, 64),
                nn.Linear(64, 1)
        )
        
        self.attn = nn.Sequential(
            nn.Linear(2*self.hidden, 128),
            nn.Linear(128, 1)
        )
        
        # Previously it was 64, now used 200
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden, 200),
            nn.Linear(200, output_size)
        )
        self.decoder2 = nn.Sequential(

            nn.Linear(256, 200),
            nn.Linear(200, output_size), 
            
            # #nn.Dropout(.10),
            # nn.Linear(256, 200),
            # #nn.LeakyReLU(0.1),
            # nn.Linear(200, 100),
            # nn.Linear(100, 50),            
            # nn.Linear(50, output_size),
            # nn.Sigmoid()

        )
        # self.dec1 = nn.Sequential(
        # nn.Linear(256,1),
        # nn.ReLU(),
        # )

        # self.dec2 = nn.Sequential(
        # nn.Linear(140,60),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(60, output_size),
        # nn.Sigmoid() 

        # )
        
        
        self.gain = gain
        
        self.init_weight()

        
    def init_weight(self):
        
        # For UFPT experiments, initialize only the decoder
        print('Initializing fresh components')
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attnenc.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.decoder.named_parameters():
            #print('Initializing Decoder:', name)
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.decoder2.named_parameters():
            #print('Initializing Decoder:', name)
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)

    def forward(self, x):
        print("input", x.shape)       
        b_size = x.size(0)
        s_size = x.size(1)
        #print('xx', x.shape)
        x = x.view(-1, x.shape[2], args.wsize)
        #print("input2", x.shape)
        x = x.permute(0, 2, 1)
        
        print('inputtoencoder: ', x.shape)
        
        out, hidden = self.encoder(x)
        print('outputofencoder: ', out.shape)
        
        
        out = self.get_attention_enc(out)
        print('enc_attn_output: ', out.shape)
        out = out.view(b_size, s_size, -1)


        # ######S: for one window ######        
        if self.onewindow == True:
            out = out.squeeze()
            lstm_out = self.decoder2(out)
            print(lstm_out.shape)
            
        # ######E: for one window ######
        else:
            lstm_out, hidden = self.lstm(out)
            #print('ext_lstm_output: ', lstm_out.shape)
            #lstm_out = out
            lstm_out = self.get_attention(lstm_out)
            #print("ext_attention_output: ",lstm_out.shape)
            lstm_out = lstm_out.view(b_size, -1)
            
            smax = torch.nn.Softmax(dim=1)

            lstm_out_smax = smax(lstm_out)
        #print("lstm_out", lstm_out.shape)
        return lstm_out #lstm_out_smax    #lstm_out_smax
        

    def get_attention(self, outputs):
        
        B= outputs[:,-1, :]
        #print('outputs: ', outputs.shape)
        B = B.unsqueeze(1).expand_as(outputs)
        outputs2 = torch.cat((outputs, B), dim=2)
        # For attention calculation
        b_size = outputs2.size(0)
        # out = outputs.view([-1, self.hidden])
        out = outputs2.reshape(-1, 2*self.hidden)
        #print("out", out.shape)
        weights = self.attn(out)
        #print("weigh", weights.shape)
        
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)
        #print('weights', weights.shape)

        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)
        # Batch-wise multiplication of weights and lstm outputs
        #print("bmm calcualted between : ", normalized_weights.shape, outputs.shape)
        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        #print("attapp", attn_applied.shape)
        attn_applied = attn_applied.squeeze()
        logits = self.decoder(attn_applied)


        return logits

    def get_attention_enc(self, outputs):
        
        b_size = outputs.size(0)

        ##### Start: Implementation of average hidden states and last hidden state
        # out = outputs

        
        # Mean_hidden_states = torch.mean(out,1,True)
        # for i in range(len(outputs)):
        #     out[i] = Mean_hidden_states[i]
        
        # # last_hidden_state = outputs[:, -1, :]
        # # last_hidden_state = last_hidden_state.unsqueeze(1)
        # # for i in range(len(outputs)):
        # #     out[i] = last_hidden_state[i]
        #out = out.reshape(-1, self.enc_out)
        ##### End: Implementation of average hidden states and last hidden state
        
        
        out = outputs.reshape(-1, self.enc_out)
        #print('1:',out.shape)
        
        weights = self.attnenc(out)
        #print('weights', weights.shape)
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)
        normalized_weights = F.softmax(weights, dim=1)
        #print('norm_weights', normalized_weights.shape)
        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        #print("attn_appliedd", attn_applied.shape)
        attn_applied = attn_applied.squeeze()
        return attn_applied


class LSTM2(torch.nn.Module):
    #model = LSTM(X.shape[2], 256, 200, 121, 2, g)
    def __init__(self, enc_input_size, input_size, hidden_nodes, sequence_size, output_size, gain, onewindow):
        super(LSTM2, self).__init__()
        self.sequence_size = sequence_size
        self.hidden = input_size
        self.onewindow = onewindow
        self.enc_out = input_size
        

        self.encoder = nn.LSTM(enc_input_size, self.enc_out, batch_first = True)
        #self.encoder = nn.GRU(enc_input_size, self.enc_out, batch_first = True)
        
        self.attn = nn.Sequential(
            nn.Linear(2*self.hidden, 128),
            nn.Linear(128, 1)
        )

        self.attention = nn.MultiheadAttention(embed_dim=self.hidden, 
                                                num_heads=8,
                                                batch_first=True, 
                                                dropout=0.1)
        
        # Previously it was 64, now used 200
        
        #this is the one with 81.64 accuracy in fbirn
        # self.decoder = nn.Sequential(
            
        #     nn.Linear(self.hidden, 200),
        #     nn.Dropout(.20),
        #     nn.Linear(200, output_size),
        #     nn.Sigmoid()
        # )
        
        self.decoder = nn.Sequential(
            
            nn.Linear(self.hidden, 200),
            nn.Dropout(.20),
            nn.Linear(200, output_size),
            nn.Sigmoid()
        )


        # self.decoder = nn.Sequential(

        #     #nn.Dropout(.10),
        #     nn.Linear(self.hidden, 200),
        #     #nn.LeakyReLU(0.1),
        #     nn.Linear(200, 100),
        #     nn.Linear(100, 50),            
        #     nn.Linear(50, output_size),
        #     nn.Sigmoid()
        # )


        # self.mlp_emb = nn.Sequential(nn.Linear(args.tp, args.tp),
        #                              nn.LayerNorm(args.tp),
        #                              nn.ELU(),
        #                              nn.Linear(args.tp, 1))      
        
        self.gain = gain
        
        self.init_weight()

        
    def init_weight(self):
        
        # For UFPT experiments, initialize only the decoder
        print('Initializing fresh components')
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.decoder.named_parameters():
            #print('Initializing Decoder:', name)
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)

    def forward(self, x):
                
        b_size = x.size(0)
        s_size = x.size(1)
        #print('aa',x.shape)
        #x = x.view(-1, x.shape[2], args.wsize)
        x = torch.squeeze(x)
        x = x.permute(0, 2, 1)
        out, hidden = self.encoder(x)
        h_lstm = out
        #print('hlstm :   : ',h_lstm.shape)
        
        ######MultiheadAttention#############
        # hidden_seq = []
        # hidden_seq += [out]
        # hidden_cat = torch.cat(hidden_seq, 1)
        # attn_output, attn_output_weights = self.attention(out, hidden_cat, hidden_cat)  # Q, K, V
        # attn_output = attn_output + out                
        # attn_output = attn_output.permute(0, 2, 1) 
        # out = self.mlp_emb(attn_output)
        # out = torch.squeeze(out)
        # lstm_out = self.decoder(out)
        ######MultiheadAttention#############

        lstm_out,atten_weights = self.get_attention(out)
        atten_weights = torch.unsqueeze(atten_weights, 2)
        #print(atten_weights.shape)


        lstm_out = lstm_out.view(b_size, -1)
        

        return lstm_out, h_lstm, atten_weights
        

    def get_attention(self, outputs):
        ##########
                #select the last hidden state for attention
        B= outputs[:,-1, :]
                #select average of all hidden states for attention
        #B = torch.mean(outputs, 1, True).squeeze()
        ##########

        B = B.unsqueeze(1).expand_as(outputs)
        
        outputs2 = torch.cat((outputs, B), dim=2)
        
        # For attention calculation
        b_size = outputs2.size(0)
        # out = outputs.view([-1, self.hidden])
        out = outputs2.reshape(-1, outputs2.shape[2])
        #print("out", out.shape)
        weights = self.attn(out)
        #print("weights1: ", weights.shape)
        
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)
        #print('weights2: ', weights.shape)

        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)
        #print('normalized_weights: ', normalized_weights.shape)
        # Batch-wise multiplication of weights and lstm outputs
        #print("bmm calcualted between : ", normalized_weights.shape, outputs.shape)
        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        #print("attention_output: ", attn_applied.shape)
        attn_applied = attn_applied.squeeze()
        logits = self.decoder(attn_applied)
        #print("decoder output: ", logits.shape)


        return logits, normalized_weights


class LSTMCG(nn.Module):
    #def __init__(self, input_size, hidden_size, input_size_nouse, sequence_size, num_classes, gain, onewindow):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMCG, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)  # Attention mechanism
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        #print('aa',x.shape)
        #x = x.view(-1, x.shape[2], args.wsize)
        x = torch.squeeze(x)
        x = x.permute(0, 2, 1)
        #x = x.view(-1, x.shape[2], args.wsize)
        #x = x.permute(0,2,1)
        h_lstm, _ = self.lstm(x)  # h_lstm: (batch, time, hidden)
        #print(h_lstm.shape)
        #print(self.attention(h_lstm).shape)
        attention_weights = torch.softmax(self.attention(h_lstm), dim=1)  # (batch, time, 1)
        weighted_sum = torch.sum(attention_weights * h_lstm, dim=1)  # (batch, hidden)
        #print(weighted_sum.shape)
        out = self.fc(weighted_sum)  # Classification output
        #print(out.shape)
        #abcc
        return out, h_lstm, attention_weights
        #return out


class LSTM(torch.nn.Module):

    def __init__(self, enc_input_size=53, ext_lstm_input_size=256, hidden=200, output_size=2, gain = 2222):
        super(LSTM, self).__init__()
        self.gain = gain
        self.ext_lstm_input_size = ext_lstm_input_size
        self.hidden = hidden
        self.lstm = nn.LSTM(self.ext_lstm_input_size, self.hidden, batch_first=True)
        
        # input size for the top lstm is the hidden size for the lower
        
        self.encoder = nn.LSTM(enc_input_size, self.ext_lstm_input_size, batch_first = True)  

        self.attnenc = nn.Sequential(
             nn.Linear(2*self.ext_lstm_input_size, 64),
             nn.ReLU(),
             nn.Linear(64, 1)
        )
     
        self.attn = nn.Sequential(
            nn.Linear(2*self.hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden, 200),
            nn.ReLU(),
            nn.Linear(200, output_size),
            nn.Sigmoid()
        )
        
        #self.init_weight()

        
    def init_weight(self):
        
        
        print('Initializing All components')
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attnenc.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.decoder.named_parameters():
           # print('Initializing Decoder:', name)
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
                
    def init_hidden(self, batch_size, device):
        
        h0 = Variable(torch.zeros(1, batch_size, self.hidden, device=device))
        c0 = Variable(torch.zeros(1, batch_size, self.hidden, device=device))
        
        return (h0, c0)
    
    def init_hidden_enc(self, batch_size, device):
        
        h0 = Variable(torch.zeros(1, batch_size, self.ext_lstm_input_size, device=device))
        c0 = Variable(torch.zeros(1, batch_size, self.ext_lstm_input_size, device=device))
        
        return (h0, c0)


    def forward(self, x):
        # Normalize input
        sx = []
        for episode in x:
            mean = episode.mean()
            sd = episode.std()
            episode = (episode - mean) / sd
            sx.append(episode)
        x = torch.stack(sx)
        
        # Reshape input
        b_size = x.size(0)
        s_size = x.size(1)
        args.wsize
        x = x.view(-1, x.shape[2], 20)
        x = x.permute(0, 2, 1)
        
        # Encoder processing
        enc_batch_size = x.size(0)
        self.enc_hidden = self.init_hidden_enc(enc_batch_size, x.device)
        #print(x.shape, self.enc_hidden[0].shape)
        enc_out, self.enc_hidden = self.encoder(x, self.enc_hidden)
        #print("encoder: ", enc_out.shape)
        
        
        # Window-level attention
        enc_attn_applied, enc_attn_weights = self.get_attention_enc(enc_out)
     
        enc_attn_applied = enc_attn_applied.view(b_size, s_size, -1)
     
        
        # Main LSTM processing
        self.lstm_hidden = self.init_hidden(b_size, x.device)
        lstm_out, self.lstm_hidden = self.lstm(enc_attn_applied, self.lstm_hidden)
       
        # Global attention
        latent_representation, lstm_attn_weights = self.get_attention(lstm_out)
        
        # Decoder
        output = self.decoder(latent_representation)
        #print(output)
        #print(latent_representation.shape, enc_attn_weights.shape, lstm_attn_weights.shape)
    
        return output, latent_representation, enc_attn_weights, lstm_attn_weights

    def get_attention_enc(self, outputs):
        # For anchor point
        B = outputs[:, -1, :]
        #print(outputs.shape, B.shape)

        B = B.unsqueeze(1).expand_as(outputs)
        outputs2 = torch.cat((outputs, B), dim=2)
        # For attention calculation
        b_size = outputs.size(0)
        out = outputs2.reshape(-1, 2 * self.ext_lstm_input_size)
      
        
        weights = self.attnenc(out)
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)
        
        
        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)
        
        # Batch-wise multiplication of weights and outputs
        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        attn_applied = attn_applied.squeeze()
        return attn_applied, normalized_weights

    def get_attention(self, outputs):
        # For anchor point
        B = outputs[:, -1, :]
        #print('aa', outputs.shape, B.shape)
        
        B = B.unsqueeze(1).expand_as(outputs)
        outputs2 = torch.cat((outputs, B), dim=2)
        
        # For attention calculation
        b_size = outputs2.size(0)
        
        out = outputs2.reshape(-1, 2 * self.hidden)
        #print(outputs2.shape, out.shape)
        
        weights = self.attn(out)
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)
        
        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)
        #print(normalized_weights.shape, outputs.shape)
        # Batch-wise multiplication of weights and outputs
        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        attn_applied = attn_applied.squeeze()
        return attn_applied, normalized_weights




