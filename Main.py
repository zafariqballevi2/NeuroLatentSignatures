import numpy as np
import torch
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import pickle
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from captum.attr import IntegratedGradients, GradientShap, NoiseTunnel, Occlusion
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from sklearn.metrics import roc_auc_score
import datetime
import pandas as pd
import sys
from sklearn.metrics import accuracy_score
import os
import gc
from pathlib import Path

from wholeMILC import NatureOneCNN
from lstm_attn import subjLSTM
from All_Architecture import combinedModel
from matplotlib import cm, colors, pyplot as plt
from scipy.signal import chirp
from utils import get_argparser
from sklearn.metrics import accuracy_score
from Get_Data import Datasets, EarlyStopping, LSTM, LSTM2, LSTMCG
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
region_contributions = defaultdict(list)
from group_lasso import GroupLasso




early_stop = True
parser = get_argparser()
args = parser.parse_args()
print("JOBID: ", args.jobid)
print("Batch Size and Learning rate: ", args.batch_size, args.lr)
device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
#print("Device: ", device)
seed_values = [22,143,65,39,4,5,6,7,8,9]
fold_values = [0,1,2,3,4,5,6,7,8,9]
seed = seed_values[args.seeds]
fold = fold_values[args.fold_v]
print("[Seed Fold]: ", seed, fold)
print("conv size: ", args.convsize)


#Reproduceability
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


eppochs = args.epp
Gain = [args.gainn]
dd = 00 #this is evein he.
Data = args.daata
enc = [args.encoder]     # 'cnn', 'lstmM'
attr_alg = args.attr_alg

#aa = args.windowspecs
l_ptr = args.l_ptr
if args.encoder == 'rnm':
    onewindow = True
else:
    onewindow = False
subjects, tp, components = args.samples, args.tp, 53  #
sample_y, window_shift = args.wsize, args.ws
samples_per_subject = args.nw #no of windows

path_dd = '/data/users4/ziqbal5/ILR/Data/'
#path_b = str(args.jobid) + '-F' + str(fold)+str(args.encoder) + args.daata + 'wsh'+str(args.ws)+ 'wsi'+str(args.wsize)+'nw'+str(args.nw) + 'S' + str(seed) +  'g'+str(args.gainn) + 'tp' + str(args.tp) + 'samp'+ str(args.samples) + 'ep'+ str(args.epp)+ 'ptr'+ str(args.l_ptr)+'-'+args.attr_alg
path_b = str(args.jobid) + '_'+str(args.encoder)+'_'+ args.daata + '_ws_'+str(args.ws)+ '_wsize_'+str(args.wsize)+'_nw_'+str(args.nw) + '_seed_' + str(seed) +  '_gain_'+str(args.gainn) + '_tp_' + str(args.tp) + '_samp_'+ str(args.samples) + '_ep_'+ str(args.epp)+'_'+args.attr_alg+ '_fold_' + str(fold)+ '_ptr_'+ str(args.l_ptr)
path_a = path_dd + path_b
print("Results will be saved here: ", path_a)

Path(path_a).mkdir(parents=True, exist_ok=True)
#start_time = time.time()
print("Data: ", Data)


# def find_indices_of_each_class(all_labels):
#     HC_index = (all_labels == 0).nonzero()
#     SZ_index = (all_labels == 1).nonzero()

#     return HC_index, SZ_index





def train_model(model, loader_train, loader_Validation, loader_test, epochs, learning_rate):
    loss = torch.nn.CrossEntropyLoss()

    
    #optimizer = Adam(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), momentum=0.5)

    # model.cuda()
    model.to(device)
    
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min',
    factor=0.5,   # halve the learning rate
    patience=5,   # wait 5 epochs before reducing
    min_lr=1e-6,  # don't go below this
    verbose=True
)


    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(dataLoaderTrain), epochs=10)
    
   
   
    #Earlystopping: 1/2
    pathtointer =  os.path.join(path_a, 'checkpoint.pt')
    early_stopping = EarlyStopping(patience=50,delta = 0.0001, path = pathtointer, verbose=True)

    
    #start: Freeze all the layers except for decoder
    # for name, param in model.named_parameters():
    #     if not name.startswith('decoder'):  # Only keep decoder trainable
    #         param.requires_grad = False

    # # Verify which parameters are trainable
    # for name, param in model.named_parameters():
    #     print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")
    #end: Freeze all the layers except for decoder
    
    train_losses, valid_losses, avg_train_losses, avg_valid_losses = [],[], [],[]
    train_accuracys, valid_accuracys, avg_train_accuracys, avg_valid_accuracys = [],[], [],[]
    train_rocs, valid_rocs, avg_train_rocs, avg_valid_rocs = [],[], [],[]
    for epoch in range(epochs):
                

        #for training
        # running_loss = 0.0
        # running_accuracy = 0.0
        model.train()
        #with torch.autograd.detect_anomaly(False):
        for i, data in enumerate(loader_train):
            #print('Batch: ',i+1)
            x, y = data
            optimizer.zero_grad()
            x = torch.squeeze(x)
            outputs, _, _, _= model(x)
            l = loss(outputs, y)
            _, preds = torch.max(outputs.data, 1)
            l.backward()
            optimizer.step()
            train_losses.append(l.item())
            accuracy = accuracy_score(y.cpu(), preds.cpu(), normalize=True)
            train_accuracys.append(accuracy)
            #print('t_acc: ', accuracy)
            sig = F.softmax(outputs, dim=1).to(device)
            y_scores = sig.detach()[:, 1]
            #print('yscores',y_scores)
            #roc = roc_auc_score(y.to('cpu'), y_scores.to('cpu'))
            #train_rocs.append(roc)
            try:
                roc = roc_auc_score(y.to('cpu'), y_scores.to('cpu'))
                train_rocs.append(roc)
            except ValueError:
                pass
            # if i % 100 == 0:
            #     print(f"GPU Mem: {torch.cuda.memory_allocated()/1024**3:.1f}GB / "
            #         f"{torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
            #     print(f"Utilization: {torch.cuda.utilization()}%")
            
        #for validation
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader_Validation):
                x, y = data
                outputs, _, _, _ = model(x)
                l = loss(outputs, y)
                _, preds = torch.max(outputs.data, 1)
                valid_losses.append(l.item())
                accuracy = accuracy_score(y.cpu(), preds.cpu(), normalize=True)
                valid_accuracys.append(accuracy)
                sig = F.softmax(outputs, dim=1).to(device)
                y_scores = sig.detach()[:, 1]
                try:
                    roc = roc_auc_score(y.to('cpu'), y_scores.to('cpu'))
                    valid_rocs.append(roc)
                except ValueError:
                    pass
                
        
        train_loss,valid_loss = np.average(train_losses),np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        train_accuracy,valid_accuracy = np.average(train_accuracys),np.average(valid_accuracys)
        avg_train_accuracys.append(train_accuracy)
        avg_valid_accuracys.append(valid_accuracy)
    

        train_roc,valid_roc = np.average(train_rocs),np.average(valid_rocs)
        avg_train_rocs.append(train_roc)
        avg_valid_rocs.append(valid_roc)

        print("epoch: " + str(epoch) + ", train_loss: " + str(train_loss) + ", val_loss: " + str(valid_loss) +", train_auc: " + str(train_roc) + ", val_auc: " + str(valid_roc) +", train_acc: " + str(train_accuracy) +" , val_acc: " + str(valid_accuracy))
        train_losses, valid_losses =[], []
        train_accuracys, valid_accuracys =[], []
        train_rocs, valid_rocs =[], []
        
        #Earlystopping: 2/2
        
        early_stopping(valid_loss, model)
        if early_stop:
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
    model.load_state_dict(torch.load(pathtointer))        
    #Start: test set results
    x_test, y_test = next(iter(loader_test))
    outputs, _ ,_, _ = model(x_test)
    _, preds = torch.max(outputs.data, 1)
    
    
    print("Test (Predicted--) :", preds)
    print("Test (GroundTruth) :", y_test)
    
    accuracy_test = accuracy_score(y_test.cpu(), preds.cpu(), normalize=True)
    sig = F.softmax(outputs, dim=1).to(device)
    y_scores = sig.detach()[:, 1]
    roc_test = roc_auc_score(y_test.to('cpu'), y_scores.to('cpu'))
    #End: test set results
        
   
   
    

    #Plot training and validation loss curves
    # fig = plt.figure(figsize=(10,8))
    # plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
    # plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')
    # # find position of lowest validation loss
    # minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
    # plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # #plt.ylim(0.5, 1) # consistent scale
    # plt.xlim(0, len(avg_train_losses)+1) # consistent scale
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # fig.savefig('loss_plot.png', bbox_inches='tight')
    # plt.close()
    #print("length", len(train_loss_acc), len(val_loss_acc), val_loss_acc)
    return optimizer, accuracy_test, roc_test, model



obj_datasets = Datasets(fold)
if args.daata == 'HCP':
    tr_data, tr_labels, val_data, val_labels,test_data, test_labels = obj_datasets.HCP()
elif args.daata == 'FBIRN':
    tr_data, tr_labels, val_data, val_labels,test_data, test_labels = obj_datasets.FBIRN()
elif args.daata == 'BSNIP':
    tr_data, tr_labels, val_data, val_labels,test_data, test_labels = obj_datasets.BSNIP()
elif args.daata == 'COBRE':
    tr_data, tr_labels, val_data, val_labels,test_data, test_labels = obj_datasets.COBRE()
elif args.daata == 'ADNI':
    tr_data, tr_labels, val_data, val_labels,test_data, test_labels = obj_datasets.ADNI()
elif args.daata == 'OASIS':
    tr_data, tr_labels, val_data, val_labels,test_data, test_labels = obj_datasets.OASIS()
elif args.daata == 'ABIDE':
    tr_data, tr_labels, val_data, val_labels,test_data, test_labels = obj_datasets.ABIDE()


print('Dataset: ', args.daata, "Tr_data, Tr_labels, val_data, val_labels, test_data, test_labels: ",tr_data.shape, tr_labels.shape, val_data.shape, val_labels.shape,test_data.shape, test_labels.shape )

print(test_labels)
#Start: These are only to extract latent representations.
tr_dataset = TensorDataset(torch.from_numpy(tr_data).float(), tr_labels.long())
val_dataset = TensorDataset(torch.from_numpy(val_data).float(), val_labels.long())
test_dataset = TensorDataset(torch.from_numpy(test_data).float(), test_labels.long())

tr_val_dataset = ConcatDataset([tr_dataset, val_dataset])
tr_val_loader = DataLoader(tr_val_dataset, batch_size = args.batch_size)


combined_dataset = ConcatDataset([tr_dataset, val_dataset, test_dataset])
combined_loader = DataLoader(combined_dataset, batch_size = args.batch_size)
#End: These are only to extract latent representations.

#print(tr_labels)
# c1_index = torch.where(tr_labels == 0)
# c2_index = torch.where(tr_labels == 1)
# c1_index = c1_index[0][0:15]
# c2_index = c2_index[0][0:15]
# c_index = torch.cat([c1_index, c2_index])

# tr_data, tr_labels = tr_data[c_index], tr_labels[c_index]


# with open(os.path.join(path_a, 'test_data.pickle'), "wb") as outfile:
#     pickle.dump(test_data, outfile)
# with open(os.path.join(path_a, 'test_labels.pickle'), "wb") as outfile:
#     pickle.dump(test_labels, outfile)

def get_data_loader(X, Y, batch_size):
    dataLoader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle = True)

    return dataLoader


encoderr = enc[0]
print("Encoder: ", encoderr)

test_data = torch.from_numpy(test_data)
dataLoaderTest = get_data_loader(test_data.float().to(device), test_labels.long().to(device), test_data.float().shape[0])




accMat = []
aucMat = []

#start_time = time.time()

#print('Gain Values Chosen:', Gain)

dir = args.exp   # NPT or UFPT


    
        
g = Gain[0]
print("Gain: ",g)


tr_data = torch.from_numpy(tr_data)
val_data = torch.from_numpy(val_data)
dataLoaderTrain =      get_data_loader(tr_data.float().to(device), tr_labels.long().to(device), args.batch_size)
#dataLoaderTrainCheck = get_data_loader(tr_data.float().to(device), tr_labels.long().to(device),  32)
dataLoaderValidation = get_data_loader(val_data.float().to(device), val_labels.long().to(device),  len(val_data))


encoder = NatureOneCNN(53, args)
lstm_model = subjLSTM(
                            device,
                            args.feature_size,
                            args.lstm_size,
                            num_layers=args.lstm_layers,
                            freeze_embeddings=True,
                            gain=g,
                        ) 

if encoderr == 'CNN':
    
    model = combinedModel(
    encoder,
    lstm_model,
    gain=g,
    PT=args.pre_training,
    exp=args.exp,
    device=device,
    oldpath=args.oldpath,
    complete_arc=args.complete_arc,
)
elif encoderr == 'LSTM': #will go with rnn #double attention
    model = LSTM(gain = g).float()
elif encoderr == 'LSTM2': #will go with rnm #single attention
    model = LSTM2(gain = g).float()
elif encoderr == 'LSTMCG':    
    model = LSTMCG(53, 128, 2, 2)

if l_ptr == 'T':
    
    #path_m ='/data/users4/ziqbal5/ILR/Data/7605899_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_300_IG_fold_0_ptr_F/PretrainedModel.pt'
    #path_m ='/data/users4/ziqbal5/ILR/Data/7605900_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_300_IG_fold_1_ptr_F/PretrainedModel.pt' #worked for fbirn
    #path_m ='/data/users4/ziqbal5/ILR/Data/7605900_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_300_IG_fold_1_ptr_F/checkpoint.pt'
    #path_m ='/data/users4/ziqbal5/ILR/Data/7605909_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_300_IG_fold_9_ptr_F/PretrainedModel.pt'
    
    #Create a mapping of convsize to path suffix
    #Old one with 10 folds
    # path_mapping = {
    #     0: '7605899_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_300_IG_fold_0_ptr_F/PretrainedModel.pt',
    #     1: '7605900_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_300_IG_fold_1_ptr_F/PretrainedModel.pt',
    #     2: '7605901_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_300_IG_fold_2_ptr_F/PretrainedModel.pt',
    #     3: '7605902_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_300_IG_fold_3_ptr_F/PretrainedModel.pt',
    #     4: '7605903_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_300_IG_fold_4_ptr_F/PretrainedModel.pt',
    #     5: '7605904_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_300_IG_fold_5_ptr_F/PretrainedModel.pt',
    #     6: '7605905_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_300_IG_fold_6_ptr_F/PretrainedModel.pt',
    #     7: '7605906_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_300_IG_fold_7_ptr_F/PretrainedModel.pt',
    #     8: '7605907_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_300_IG_fold_8_ptr_F/PretrainedModel.pt',
    #     9: '7605909_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_300_IG_fold_9_ptr_F/PretrainedModel.pt'
    # }

    path_mapping = {
        0: '383612_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_1500_IG_fold_0_ptr_F/PretrainedModel.pt',
        1: '383613_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_1500_IG_fold_1_ptr_F/PretrainedModel.pt',
        2: '383614_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_1500_IG_fold_2_ptr_F/PretrainedModel.pt',
        3: '383615_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_1500_IG_fold_3_ptr_F/PretrainedModel.pt',
        4: '383616_LSTM_HCP_ws_20_wsize_20_nw_60_seed_1_gain_0.9_tp_1200_samp_830_ep_1500_IG_fold_4_ptr_F/PretrainedModel.pt'

    }

    # Get the path based on convsize
    if args.convsize in path_mapping:
        path_m = f'/data/users4/ziqbal5/ILR/Data/{path_mapping[args.convsize]}'
    else:
        print("Couldn't find the pretrained model. Exiting...")
        exit()
    print("pretrained model loaded from: ", path_m)
    model.load_state_dict(torch.load(path_m, map_location=torch.device('cpu')))    
    

starttime = datetime.datetime.now()
optimizer, accuracy_test, auc_test,model = train_model(model, dataLoaderTrain, dataLoaderValidation, dataLoaderTest, eppochs, args.lr)#3e-4  #.0005
print("Auc and Accuracy: ", auc_test, accuracy_test)

endtime = datetime.datetime.now()
elapsed_time = endtime-starttime

torch.save(model.state_dict(),  os.path.join(path_a, 'PretrainedModel.pt'))

if args.daata == 'HCP':
    accDataFrame = pd.DataFrame(accMat, aucMat)
    accDataFrame = pd.DataFrame({'Accuracy':[accuracy_test], 'AUC':[auc_test]})
    accfname = os.path.join(path_a, 'ACC_AUC.csv')
    accDataFrame.to_csv(accfname)
    print("auc: " ,auc_test)
    print("Exiting after training and saving model...")

    exit()


brain_functional_networks_mapping = {
        "sub-cortical": [0,1,2,3,4],
        "auditory": [5,6],
        "sensorimotor": list(range(7,16)),
        "visual": list(range(16,25)),
        "cognitive control": list(range(25,42)),
        "default mode": list(range(42,49)),
        "cerebellar": [49,50,51,52]
    }


#device = next(model.parameters()).device

def mapping_component_latent(dataloader):
    all_enc_attn, all_lstm_attn, all_latentRep, all_y = [], [], [],[]
    model.eval()
    with torch.no_grad():
        
        for i, data in enumerate(dataloader):
            x_batch, y = data
            #print('aaaaaa')
            X_batch = x_batch.to(device)
            _, l_representations, enc_attn, lstm_attn = model(X_batch)
            #print(enc_attn.shape, lstm_attn.shape)
            # Reshape enc_attn: (batch*num_windows, 20) → (batch, num_windows, 20)
            batch_size, num_windows = X_batch.shape[0], X_batch.shape[1]
            #print('aa', enc_attn.shape, lstm_attn.shape, l_representations.shape)
            enc_attn = enc_attn.view(batch_size, num_windows, -1)
            #print('cc', enc_attn.shape)
            all_enc_attn.append(enc_attn.to(device))
            all_lstm_attn.append(lstm_attn.to(device))
            all_latentRep.append(l_representations)
            all_y.append(y)
        
        #print(all_enc_attn[0].shape, len(all_enc_attn))
        #print(len(all_latentRep), all_latentRep[0].shape)
        combined_y = torch.cat(all_y, dim=0)
        combined_latent = torch.cat(all_latentRep, dim=0)
    
        # Average across batches
        global_enc_attn = torch.cat(all_enc_attn).mean(dim=0)  # (num_windows, 20)
        global_lstm_attn = torch.cat(all_lstm_attn).mean(dim=0)  # (num_windows,)
        #print('abc', global_enc_attn.shape, global_lstm_attn.shape)
        

        W_ih_enc = model.encoder.weight_ih_l0[:256, :].detach()  # (256, 53)
        W_ih_main = model.lstm.weight_ih_l0[:200, :].detach()      #(200, 256)
        #print("def", W_ih_enc.shape, W_ih_main.shape)
        # Reshape attentions
        global_enc_attn = global_enc_attn.view(-1, 1, 1)  # (num_windows=7, 20) → (7*20, 1, 1)
        global_lstm_attn = global_lstm_attn.view(-1, 1, 1)  # (num_windows=7, 1, 1)
       # print('abc', global_enc_attn.shape, global_lstm_attn.shape)
        # Combine weights with attention
        component_to_latent = torch.zeros(200, 53)
     
        # For each window and time step
        num_windows, window_size = 7, 20
        for window in range(num_windows):
            # Main LSTM weights scaled by global attention
            lstm_weights = W_ih_main * global_lstm_attn[window]
            for t in range(window_size):
                # Encoder weights scaled by window-level attention
                enc_weights = W_ih_enc * global_enc_attn[window * window_size + t]
                #print('b', enc_weights.shape, W_ih_enc.shape, global_enc_attn[window * window_size + t].shape)
                
                # Accumulate contributions over all time steps
                lstm_weights,enc_weights =  lstm_weights.to('cpu'), enc_weights.to('cpu')
                component_to_latent += torch.matmul(lstm_weights, enc_weights)
               # print(torch.matmul(lstm_weights, enc_weights).shape)
    return component_to_latent, combined_latent, combined_y
    #print('component_to_latent: ', component_to_latent.shape)



component_to_latent, combined_latent, combined_y = mapping_component_latent(dataLoaderTest)
#print('component_to_latent: ', component_to_latent.shape, combined_latent.shape)


for latent_idx in range(200):
    latent_weights = component_to_latent[latent_idx]
    #print(latent_weights.shape)
    region_weights = {}
    
    for region, components in brain_functional_networks_mapping.items():
        region_weights[region] = latent_weights[components].sum().item()
        #print(region, region_weights[region])
    
    # Find dominant region
    dominant_region = max(region_weights, key=region_weights.get)
    region_contributions[dominant_region].append(latent_idx)
print(region_contributions)

with open(os.path.join(path_a, "region_contributions.pkl"), "wb") as outfile:
    pickle.dump(region_contributions, outfile)
with open(os.path.join(path_a, "combined_y.pkl"), "wb") as outfile:
    pickle.dump(combined_y, outfile)
with open(os.path.join(path_a, "combined_latent.pkl"), "wb") as outfile:
    pickle.dump(combined_latent, outfile)
print(combined_latent.shape, combined_y.shape)


#print(groups, mean_vals)
accDataFrame = pd.DataFrame({'Accuracy':[accuracy_test], 'AUC':[auc_test]})
#print('df', accDataFrame)
accfname = os.path.join(path_a, 'ACC_AUC.csv')
accDataFrame.to_csv(accfname)