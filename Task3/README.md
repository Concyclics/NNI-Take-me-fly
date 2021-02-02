# Task 3 进阶任务 Feature Engineering

## Task 3.1

* 跑通 NNI [Feature Engineering Sample](https://github.com/SpongebBob/tabular_automl_NNI)

### 1. 特征工程简介

有这么一句话在业界广泛流传：**对于一个机器学习问题，数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已**。

特征工程，顾名思义，是对原始数据进行一系列工程处理，将其提炼为特征，作为输入供算法和模型使用。从本质上来讲，特征工程是一个表示和展现数据的过程。在实际工作中，特征工程旨在去除原始数据中的杂质和冗余，设计更高效的特征以刻画求解的问题与预测模型之间的关系。

### 2. 自动化特征工程简介

自动化特征工程是旨在通过从数据集中自动创建候选特征，且从中选择若干最佳特征进行训练的一种方式。

利用NNI中的特征工程工具，我们能够方便地实现特征工程的自动调优。

### 3. NNI特征工程样例

#### 3.1 配置文件

* 配置搜索空间：

  ```json
  {
  
      "count":[
  
          "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
  
          "C11","C12","C13","C14","C15","C16","C17","C18","C19",
  
          "C20","C21","C22","C23","C24","C25","C26"
  
      ],
  
      "aggregate":[
  
          ["I9","I10","I11","I12"],
  
          [
  
              "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
  
              "C11","C12","C13","C14","C15","C16","C17","C18","C19",
  
              "C20","C21","C22","C23","C24","C25","C26"
  
          ]
  
      ],
  
      "crosscount":[
  
          [
  
              "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
  
              "C11","C12","C13","C14","C15","C16","C17","C18","C19",
  
              "C20","C21","C22","C23","C24","C25","C26"
  
          ],
  
          [
  
              "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
  
              "C11","C12","C13","C14","C15","C16","C17","C18","C19",
  
              "C20","C21","C22","C23","C24","C25","C26"
  
          ]
  
      ]
  
  }
  ```

* 配置实验：

  ```
  authorName: default
  
  experimentName: example-auto-fe
  
  trialConcurrency: 1
  
  maxExecDuration: 10h
  
  maxTrialNum: 2000
  
  #choice: local, remote
  
  trainingServicePlatform: local
  
  searchSpacePath: search_space.json
  
  #choice: true, false
  
  useAnnotation: false
  
  tuner:
  
    codeDir: .
  
    classFileName: autofe_tuner.py
  
    className: AutoFETuner
  
    classArgs:
  
      optimize_mode: maximize
  
  trial:
  
    command: python3 main.py
  
    codeDir: .
  
    gpuNum: 0
  
  ```

#### 3.2 实验结果

* **Top 10 Trails：**

  ![](imgs/top10.png)

* **Default Metric：**

  ![](imgs/metric.png)

* **Hyper-parameter：**

  ![](imgs/hyperparam.PNG)

  

