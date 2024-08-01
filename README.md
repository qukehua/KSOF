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
Download all the data and put them in the `./data` directory.

[H3.6M](https://drive.google.com/file/d/15OAOUrva1S-C_BV8UgPORcwmWG2ul4Rk/view?usp=share_link)

[Original stanford link](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip) has crashed, this link is a backup.

Directory structure:
```shell script
data
|-- h36m
|   |-- S1
|   |-- S5
|   |-- S6
|   |-- ...
|   |-- S11
```


[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)

Directory structure: 
```shell script
data
|-- 3dpw
|   |-- sequenceFiles
|   |   |-- test
|   |   |-- train
|   |   |-- validation
```

### Training
------
#### H3.6M
```bash
cd exps/baseline_h36m/
sh run.sh
```

#### AMASS
```bash
cd exps/baseline_amass/
sh run.sh
```

## Evaluation
------
#### H3.6M
```bash
cd exps/baseline_h36m/
python test.py --model-pth your/model/path
```

#### AMASS
```bash
cd exps/baseline_amass/
#Test on AMASS
python test.py --model-pth your/model/path 
#Test on 3DPW
python test_3dpw.py --model-pth your/model/path 
```

### Citation
If you find this project useful in your research, please consider cite:
```bash
@article{guo2022back,
  title={Back to MLP: A Simple Baseline for Human Motion Prediction},
  author={Guo, Wen and Du, Yuming and Shen, Xi and Lepetit, Vincent and Xavier, Alameda-Pineda and Francesc, Moreno-Noguer},
  journal={arXiv preprint arXiv:2207.01567},
  year={2022}
}
```

### Contact
Feel free to contact [Wen](wen.guo@inria.fr) or [Me](yuming.du@enpc.fr) or open a new issue if you have any questions.
