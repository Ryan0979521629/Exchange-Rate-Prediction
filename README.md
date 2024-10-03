# 預測外匯匯率 
## 學號:110201532 姓名:范植緯 系級:資工3A
## 使用版本
Python 3.11.5
使用的是jupyter notebooks
The version of the notebook server is: 6.5.4

## 解釋程式碼執行
### 1.函示庫
以下程式碼是此次做predict時需要用到的函式庫
```
import torch
import torch.nn as nn
import numpy as np
from torch.optim import optimizer
import pandas as pd
import matplotlib.pyplot as plt
import os
```
### 2.定義
```
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, input, target):
        return torch.sqrt(torch.mean((input - target) ** 2))
```
這段程式定義了LinearRegression的模型還有我自定義的RMSE的Loss function，在LinearRegression中，利用__init__來初始化，self.linear用來執行線性變換，將特徵與權重相乘
### 3.Load data
```
def load_data(folder_path):
  load_df = pd.DataFrame()
  # 遍歷文件夾中的所有文件
  name_list=[]
  for filename in os.listdir(folder_path):
      if filename.endswith(".csv"):
          file_path = os.path.join(folder_path, filename)
          df = pd.read_csv(file_path)
          load_df = pd.concat([load_df, df["現鈔買入"], df["現鈔賣出"], df["即期買入"], df["即期賣出"]], axis=1)
          filename = filename.replace(".csv", "")
          name_list.extend([f"{filename}現鈔買入", f"{filename}現鈔賣出", f"{filename}即期買入", f"{filename}即期賣出"])
  load_df.columns = name_list

  # 反向排序
  load_df = load_df.iloc[::-1]

  # 排序column
  load_df = load_df.reindex(sorted(load_df.columns), axis=1)

  # 處理空值
  load_df.replace("-", 0, inplace=True)

  return load_df

train_df = load_data("C:/Users/User")
train_df
```
這段定義了load_data的function，藉由給路徑讓電腦幫你找這個路徑的所有結尾為.csv的檔案，組成一個DataFrame，並藉由反向排序、處理空值來讓資料更好被使用

### 3.設定不同condiction
```
# 設定輸入資料的天數範圍
input_date_data_size = 4
#要到第幾項
power=6
# 設定 seed
torch.manual_seed(1234)
np.random.seed(1234)
```
這一段程式碼的input_date_data_size 代表著我們要用幾天的資料來做預測
power代表著我們總共要用到第幾個次方向，若power=3，就代表我們可以到W~i~^3^，i為我們的feature，下面的設定seed用來確保在不同的運行中獲得相同的結果

### 4.所有data進行整合
```
train = train_df.to_numpy()
train_size, feature_size = train.shape
# 以一段時間的資料當作輸入，故資料數量要扣掉輸入天數範圍
train_size = train_size - input_date_data_size

train_x = np.empty([train_size, feature_size * input_date_data_size*power], dtype = float)#乘power讓資料集擴展到次方項
train_y = np.empty([train_size, feature_size], dtype = float)

for idx in range(train_size):
    temp_data = np.array([])
    for count in range(input_date_data_size):
        temp_data = np.hstack([temp_data, train[idx + count]])#把前幾天的資料合併
        for nth_term in range(2,power+1):
            temp_data = np.hstack([temp_data, train[idx + count]**nth_term])#把前幾天的資料合併
    train_x[idx, :] = temp_data
    train_y[idx, :] = train[idx + input_date_data_size]#應該要預測到的實際第幾天資料

    # y值只留下現鈔買入
filtered_columns = [train_df.columns.get_loc(col) for col in train_df.columns if '現鈔買入' in col]
train_y = train_y[:, filtered_columns]
```
train_x 為所有資料中的feature創造時的 feature_size * input_date_data_size*power意思是原本要有的feature項再乘上我們一次要用幾筆的資料做預測再乘上整體需要多少的項
train_y為資料的正確結果
底下的三層for迴圈，第一個idx用來輸入該行的值，count用來把4天的資料合併到一行，nth_term用來把值的power項乘進去，最後一起塞入到train_x[idx]中，最後讓train_y只留下現鈔買入來做為一個指標

### 5.Feature Scaling
```
#Feature scaling
mean_x = np.mean(train_x, axis = 0)
std_x = np.std(train_x, axis = 0)
for i in range(len(train_x)):
    for j in range(len(train_x[0])):
        if std_x[j] != 0:
            train_x[i][j] = (train_x[i][j] - mean_x[j]) / std_x[j]
```
這段程式碼是利用Feature scaling讓所有的feature都在0~1之間，目的是為了讓所有的feature的權重都差不多，不會有feature對預測特別有影響力

### 6. Validation
```
split_ratio = 0.2

# 計算驗證集的大小
val_size = int(train_size * split_ratio)

# 隨機生成索引以選擇要包含在驗證集中的樣本
indices = np.random.permutation(train_size)

# 使用索引切分數據
val_indices = indices[:val_size]
train_indices = indices[val_size:]

# 創建驗證集
val_x = train_x[val_indices]
val_y = train_y[val_indices]
# 創建訓練集
train_x = train_x[train_indices]
train_y = train_y[train_indices]
```
split_ratio用來表示切幾分之幾作為Validation data，並用random來打亂順序，切成val_x、train_x和val_y、train_y

### 7.轉為張量
```
#轉為pytorch張量，才可以餵進pytorch中
train_x = torch.from_numpy(train_x.astype(np.float32))
train_y = torch.from_numpy(train_y.astype(np.float32))
val_x = torch.from_numpy(val_x.astype(np.float32))
val_y = torch.from_numpy(val_y.astype(np.float32))

```
使用這一段程式碼轉型態才可以用到pytorch進行training

### 8.Setting model
```
model = LinearRegression(feature_size * input_date_data_size*power, 8)

best_validation_loss = float('inf')

learning_rates =0.01
criterion =RMSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rates)
patience = 10 
counter = 0  
```
將model設成我們先前所設定的LinearRegression，輸入為input的維度，輸出維度8是因為我們有八國的資料要輸出
底下的learning rate代表經過一次gradient_decent要跑多遠
criterion用來計算Loss
optimizer 代表我們要使用的方法，在這裡是使用SGD
patience和counter和best_validation_loss負責紀錄在gradient_decent時會不會越learning validation_loss越大。

### 9.training
```
epochs = 100000
val_loss_history = []
train_loss_history=[]
for epoch in range(epochs):
    model.train()
    # forward pass and loss
    train_y_predicted = model(train_x)
    loss = criterion(train_y_predicted, train_y)
    train_loss_history.append(loss.item())
    # backward pass
    loss.backward()
    # update
    optimizer.step()
    # init optimizer
    optimizer.zero_grad()

    # 驗證資料集
    model.eval()
    val_y_predicted = model(val_x)
    val_loss = criterion(val_y_predicted, val_y)
    val_loss_history.append(val_loss.item())
    
    if val_loss < best_validation_loss:
        best_validation_loss = val_loss
        counter = 0  # 重置
    else:
        counter += 1
    if counter >= patience:
        break
    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch+1}, train_loss = {loss.item(): .4f}, val_loss = {val_loss.item(): .4f}')
```
epochs代表要iteration幾次，vak_loss_history和train_loss_history用來記錄過程的Loss，在迴圈中，train完出來的值會放到train_y_predicted，再藉由criterion得到loss，並且藉由 optimizer.step()來更新數據，optimizer.zero_grad()來讓梯度歸零，底下也是相同的做法，只是是用再validation上，並且使用counter 和patience來判斷learning過程中會不會越來越糟，如果越來越糟就會自動停止迴圈。

### 10.load_test
```
test_df = load_data("C:/Users/User/test")

```
使用一樣的load_data，將測試集的資料都載入

### 11.處理資料
```
test = test_df.to_numpy()
test_size, feature_size = test.shape
# 因為 test 資料已經事先切割好範圍，故需要明確切分每段資料
test_size = test_size//input_date_data_size
test_x = np.empty([test_size, feature_size * input_date_data_size*power], dtype = float)

for idx in range(test_size):
  temp_data = np.array([])
  for count in range(input_date_data_size):
    temp_data = np.hstack([temp_data, test[idx * input_date_data_size + count]])
    for nth_term in range(2,power+1):
        temp_data = np.hstack([temp_data, test[idx * input_date_data_size + count]**nth_term])
  test_x[idx, :] = temp_data

# test 資料也需要照 training 方式做正規化
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
```
跟先前第4步第5步做得差不多，把資料處理成我們要的形式

### 12.predicted
```
test_x = torch.from_numpy(test_x.astype(np.float32))
predicted = model(test_x)

ids = [x for x in range(len(predicted))]
output_df = pd.DataFrame({'id': ids})
# 要按照規定順序設定col
currency_columns = ["AUD", "CAD", "EUR", "GBP", "HKD", "JPY", "KRW", "USD"]

for i, column_name in enumerate(currency_columns):
    output_df[column_name] = [x[i] for x in predicted.tolist()]


output_df.to_csv('answer/submission.csv', index=False)
```
將我們training完的model用來predict test_data，並將資料按照順序排好，並且用output_df.to_csv('answer/submission.csv', index=False)把結果存到我們要的特定位置。