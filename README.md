# IP-UIE
This is a Tensorflow implement of IP-UIE.
The original structure of our code is based on Retinex-Net(https://github.com/weichen582/RetinexNet).

### Requirements ###
1. Tensorflow >= 1.12.0
2. python 3.6
3. Numpy
4. pillow

### Data ###
The data for training IP-UIE is available at [Baidu Cloud](https://pan.baidu.com/s/1Gl10C_u1yCZLB-I6JA8I8A)(code:ag4p).

You can also download the detection dataset for evaluation from [Baidu Cloud](https://pan.baidu.com/s/18cm_MO2CezWQLmcZEf_ZnA)(dneb).

### Testing ###
```shell
python main.py --phase=test
```

### Training ###
```shell
python main.py --phase=train
```

### Citation ###
