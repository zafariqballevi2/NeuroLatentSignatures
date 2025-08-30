import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lstm_attn import subjLSTM
from All_Architecture import combinedModel
import pickle
from pathlib import Path
import os

sample_x = 53
from utils import get_argparser
parser = get_argparser()

args = parser.parse_args()
tags = ["pretraining-only"]
config = {}
config.update(vars(args))

if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
        
else:
        device = torch.device("cpu")

print('device = ', device)

if args.exp == 'FPT':
    gain = [0.1, 0.05, 0.05]  # FPT
elif args.exp == 'UFPT':
    gain = [0.05, 0.45, 0.65]  # UFPT
else:
    gain = [0.25, 0.35, 0.65]  # NPT
ID = args.script_ID - 1
current_gain = gain[ID]
args.gain = current_gain

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class NatureOneCNN(nn.Module):
    def init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    def init_hidden_enc(self, batch_size, device):
        
                h0 = Variable(torch.zeros(1, batch_size, 256, device=device))
                c0 = Variable(torch.zeros(1, batch_size, 256, device=device))
        
                return (h0, c0)

    def get_attention_enc(self, outputs):
        
        # For anchor point
        B= outputs[:,-1, :]
        B = B.unsqueeze(1).expand_as(outputs)
        outputs2 = torch.cat((outputs, B), dim=2)
        
        
        # For attention calculation
        b_size = outputs.size(0)
        # out = outputs.view([-1, self.hidden])
        out = outputs2.reshape(-1, 2* self.enc_out)

        weights = self.attnenc(out)
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)

        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)

        # Batch-wise multiplication of weights and lstm outputs

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        attn_applied = attn_applied.squeeze()

        return attn_applied

    def __init__(self, input_channels, args):
        super().__init__()
        self.feature_size = args.feature_size
        #print("__________", self.feature_size)
        self.input_size = 53
        self.enc_out = 256
        #self.encoderr = args.encoder

        #self.encoderr = str(self.encoderr)
        self.hidden_size = self.feature_size
        self.downsample = not args.no_downsample
        self.input_channels = input_channels
        self.twoD = args.fMRI_twoD
        self.end_with_relu = args.end_with_relu
        self.args = args
        init_ = lambda m: self.init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        self.flatten = Flatten()

        if self.downsample:
            self.final_conv_size = 32 * 7 * 7
            self.final_conv_shape = (32, 7, 7)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),  #0
                nn.ReLU(),                                          #1
                init_(nn.Conv2d(32, 64, 4, stride=2)),              #2
                nn.ReLU(),                                          #3
                init_(nn.Conv2d(64, 32, 3, stride=1)),              #4
                nn.ReLU(),                                          #5
                Flatten(),                                          #6
                init_(nn.Linear(self.final_conv_size, self.feature_size)), #7
                # nn.ReLU()   #8
            )

        else:
            #print("fet size", self.feature_size)
            self.final_conv_size =   args.convsize #8400 #200*12 #26400 #6400 #14400 12400 8400
            self.final_conv_shape = (200, 12)


            self.main = nn.Sequential(
                init_(nn.Conv1d(input_channels, 64, 4, stride=1)),
                nn.ReLU(),
                init_(nn.Conv1d(64, 128, 4, stride=1)),
                nn.ReLU(),
                init_(nn.Conv1d(128, 200, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                init_(nn.Conv1d(200, 128, 3, stride=1)),
                nn.ReLU(),
            )

    def forward(self, inputs, fmaps=False, five=False):

        f5 = self.main[:6](inputs)
        out = f5
        #print("f5",f5.shape)
        out = self.main[6:8](f5)
        #print(out.shape)
        f5 = self.main[8:](f5)
        if self.end_with_relu:
            assert (
                self.args.method != "vae"
            ), "can't end with relu and use vae!"
            out = F.relu(out)
        if five:
            return f5.permute(0, 2, 1)
        if fmaps:
            return {
                "f5": f5.permute(0, 2, 1),
                # 'f7': f7.permute(0, 2, 1),
                "out": out,
            }
        
        return out
