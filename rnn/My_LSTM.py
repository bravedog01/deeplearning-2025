import torch
import torch.nn as nn
import math


class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 输入门
        self.W_xi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # 遗忘门
        self.W_xf = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # 候选记忆
        self.W_xc = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # 输出门
        self.W_xo = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, h_t, c_t):
        i_t = torch.sigmoid(x @ self.W_xi + h_t @ self.W_hi + self.b_i)
        f_t = torch.sigmoid(x @ self.W_xf + h_t @ self.W_hf + self.b_f)
        g_t = torch.tanh(x @ self.W_xc + h_t @ self.W_hc + self.b_c)
        o_t = torch.sigmoid(x @ self.W_xo + h_t @ self.W_ho + self.b_o)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = LSTMBlock(input_size, hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=1)
        )

    def init_hidden(self, batch_size=1):
        device = next(self.parameters()).device
        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=device)
        return h_t, c_t

    def forward(self, x, h_t, c_t):
        """
        x: (batch_size, seq_len, input_size)
        h_t, c_t: (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        for t in range(seq_len):
            h_t, c_t = self.lstm(x[:, t, :], h_t, c_t)
        output = self.classifier(h_t)  # 只使用最后一个时间步的输出
        return output, h_t, c_t


if __name__ == "__main__":
    # 测试模型运行
    input_size = 57   # 如：one-hot 字母编码
    hidden_size = 128
    output_size = 18  # 类别数量

    model = LSTMModel(input_size, hidden_size, output_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 造一个输入
    x = torch.randn(1, 6, input_size).to(device)  # batch=1, seq_len=6
    h, c = model.init_hidden(batch_size=1)
    output, h, c = model(x, h, c)
    print("Output shape:", output.shape)  # 应为 (1, output_size)
