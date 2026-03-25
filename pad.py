import pandas as pd
from sklearn.model_selection import train_test_split

data_file = 'C:/Users/HuaFei/Desktop/task/aes_bert-bisru++-attention-作文评分asap/data/asap-aes/training_set_rel3.xlsx'
raw_datas = pd.read_excel(data_file)

pre_data = {"1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": []}

for rowIndex in range(len(raw_datas)):
    essay_set = raw_datas.iloc[rowIndex]['essay_set']
    pre_data[str(essay_set)].append(raw_datas.iloc[rowIndex])

print("pre_data:",pre_data)
# print(len(pre_data["1"]))
# print(len(pre_data["2"]))
# print(len(pre_data["3"]))
# print(len(pre_data["4"]))
# print(len(pre_data["5"]))
# print(len(pre_data["6"]))
# print(len(pre_data["7"]))
# print(len(pre_data["8"]))

trains = []
tests = []
test_size = 0.2
for key in pre_data.keys():

    train, test = train_test_split(pre_data[key], test_size=test_size, random_state=42)
    # print(type(train)) # <class 'list'>
    # print(type(train[0])) # <class 'pandas.core.series.Series'>
    # print("len(train):",len(train))
    # print("len(test):", len(test))

    trains.extend(train)
    tests.extend(test)

print("len(trains):",len(trains))
print("len(tests):",len(tests))

# 重新写入
trains = pd.DataFrame(trains)
trains.to_excel('train_asap.xlsx', index=False)

tests = pd.DataFrame(tests)
tests.to_excel('test_asap.xlsx', index=False)