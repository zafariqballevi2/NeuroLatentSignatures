# NeuroLantentSignatures

### Pretraining: 
   ```bash
     python -m Main --fold_v 0 --daata HCP --encoder LSTM --seeds 1 --ws 20 --nw 60 --wsize 20 --convsize 1 --epp 1000 --tp 1200 --samples 311 --l_ptr T
```


### Downstream training:
  ```bash
  python -m Main --fold_v 0 --daata FBIRN --encoder LSTM --seeds 1 --ws 20 --nw 7 --wsize 20 --convsize 1 --epp 1000 --tp 140 --samples 311 --l_ptr T
  ```

### T-test and Probe analysis:
  ```bash
  python -m ttest_probing
  ```

### Description of Arguments:

 ```bash
  --fold_v: Number of fold (Eg. If using 5 fold cv, then pass each fold index to train a single model).
  --daata: Pass the data in the shape of:  (Subjects, components, time points). For pretraining, pass HCP. For downstream, pass any one of these datasets (FBIRN, ADNI, B-SNIP, ABIDE, OASIS)
  --encoder: Model used for training (LSTM: Modified wholeMILC)
  --ws: Window shift (Should be equal to window size (20)
  --wsize: window size (20)
  --convsize: Pass seed value
  --epp: Number of epochs
  --tp: total time points
  --samples: Number of subjects
  --l_ptr: T (with pretraining), F (w/o pretraining)
```
