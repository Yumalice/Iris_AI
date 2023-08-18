import pandas as pd
import torch


def read_data():
    # データの読み込み
    IRIS_csv = pd.read_csv("IRIS.csv", index_col=None, header=0)
    # 読み込んだデータの確認
    # print(tips_csv.head())

    # NNで処理できるようにデータを変換
    IRIS_data = IRIS_csv.replace(
        {
            "species": {"Iris-setosa": 0, "Iris-versicolor": 1 ,"Iris-virginica": 2 },
        }
    )
    #  IRIS_data["sepal_length"] = IRIS_data["sepal_length"] / 10 いらない？？？？
    # 変換後のデータの確認
    # print(tips_data.head())

    return IRIS_data


# データをPyTorchでの学習に利用できる形式に変換
def create_dataset_from_dataframe(IRIS_data, target_tag="sepal_length"):
    # "tip"の列を目的にする
    target = torch.tensor(IRIS_data[target_tag].values, dtype=torch.float32).reshape(-1, 1)
    # "tip"以外の列を入力にする
    input = torch.tensor(IRIS_data.drop(target_tag, axis=1).values, dtype=torch.float32)
    return input, target


# 4層順方向ニューラルネットワークモデルの定義
class FourLayerNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h1 = torch.tanh(self.l1(x))
        h2 = torch.tanh(self.l2(h1))
        o = self.l3(h2)
        return o


def train_model(nn_model, input, target):
    # データセットの作成
    tips_dataset = torch.utils.data.TensorDataset(input, target)
    # バッチサイズ=25として学習用データローダを作成
    train_loader = torch.utils.data.DataLoader(tips_dataset, batch_size=25, shuffle=True)

    # オプティマイザ
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.01, momentum=0.9)

    # データセット全体に対して10000回学習
    for epoch in range(10000):
        # バッチごとに学習する
        for x, y_hat in train_loader:
            y = nn_model(x)
            loss = torch.nn.functional.mse_loss(y, y_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 1000回に1回テストして誤差を表示
        if epoch % 1000 == 0:
            with torch.inference_mode():  # 推論モード（学習しない）
                y = nn_model(input)
                loss = torch.nn.functional.mse_loss(y, target)
                print(epoch, loss)


# データの準備
IRIS_data = read_data()
input, target = create_dataset_from_dataframe(IRIS_data)

# NNのオブジェクトを作成
nn_model = FourLayerNN(input.shape[1], 30, 1)
train_model(nn_model, input, target)

# 学習後のモデルの保存
# torch.save(nn_model.state_dict(), "nn_model.pth")

# 学習後のモデルのテスト
test_data = torch.tensor(
    [
        [
         # sepal_length
            3.2 ,  # sepal_width
            1.4,  # petal_length
            0.1,  # petal_width
            2,  # species
        ]
    ],
    dtype=torch.float32,
)
with torch.inference_mode():  # 推論モード（学習しない）
    print(nn_model(test_data))