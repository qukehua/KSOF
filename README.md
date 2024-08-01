# KSOF
**KSOF: Leveraging Kinematics and Spatio-temporal Optimal Fusion for Human Motion Prediction** 

### Abstract
------
Ignoring the meaningful kinematics law, which generates improbable or impractical predictions, is one of the obstacles
to human motion prediction. Current methods attempt to tackle this problem by taking simple kinematics information
as auxiliary features to improve predictions. It remains challenging to utilize human prior knowledge deeply, such
as the trajectory formed by the same joint should be smooth and continuous on this task. In this paper, we advocate
explicitly describing kinematics information via velocity and acceleration by proposing a novel loss called joint point
smoothness (JPS) loss, which calculates the acceleration of joints to smooth the sudden change in joint velocity. In
addition, capturing spatio-temporal dependencies to make feature representations more informative is also one of the
obstacles in this task. Therefore, we propose a dual-path network (KSOF) that models the temporal and spatial dependencies from kinematic temporal convolutional network (K-TCN) and spatial graph convolutional networks (S-GCN),
respectively. Moreover, we propose a novel multi-scale fusion module named spatio-temporal optimal fusion (SOF)
to better capture the essential correlation and important features at different scales from spatio-temporal coupling
features. We evaluate our approach on three standard benchmark datasets, including Human3.6M, CMU-Mocap, and
3DPW datasets. For both short-term and long-term predictions, our method achieves outstanding performance on all
these datasets, confirming its effectiveness.

### Network Architecture
------
![image](images/architecture.png)

### Requirements
------
- PyTorch = 1.8.0
- Numpy
- CUDA = 11.4
- Python = 3.1.0

### Data Preparation
------
Download all the data and put them in the [dataset path].

[H3.6M](https://drive.google.com/file/d/15OAOUrva1S-C_BV8UgPORcwmWG2ul4Rk/view?usp=share_link)

Directory structure: 
```shell script
[dataset path]
|-- h36m
|   |-- S1
|   |-- S5
|   |-- S6
|   |-- ...
|   |-- S11
```


[CMU mocap](http://mocap.cs.cmu.edu/) 

[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)

Directory structure: 
```shell script
[dataset path]
|-- 3dpw
|   |-- sequenceFiles
|   |   |-- test
|   |   |-- train
|   |   |-- validation
```

### Training
------
+ Train on Human3.6M:

`
python main_h36m.py
--data_dir
[dataset path]
--num_gcn
4
--dct_n
15
--input_n
10
--output_n
10
--skip_rate
1
--batch_size
32
--test_batch_size
64
--node_n
66
--cuda_idx
cuda:0
--d_model
16
--lr_now
0.0005
--epoch
100
--test_sample_num
-1
`

+ Train on CMU-MoCap:

`
python main_cmu_3d.py
--data_dir
[dataset path]
--num_gcn
4
--dct_n
15
--input_n
10
--output_n
25
--skip_rate
1
--batch_size
16
--test_batch_size
32
--node_n
75
--cuda_idx
cuda:0
--d_model
16
--lr_now
0.005
--epoch
100
--test_sample_num
-1
`

+ Train on 3DPW:

`
--data_dir
[dataset path]
--num_gcn
4
--dct_n
15
--input_n
10
--output_n
30
--skip_rate
1
--batch_size
32
--test_batch_size
64
--node_n
69
--cuda_idx
cuda:0
--d_model
16
--lr_now
0.001
--epoch
100
--test_sample_num
-1
`
未选择任何文件


## Evaluation
------
Add `--is_eval` after the above training commands.

The test result will be saved in `./checkpoint/`.

#### Ackowlegments
Our code is based on [PGBIG](https://github.com/705062791/PGBIG) and [Dpnet](https://ieeexplore.ieee.org/document/10025861).
