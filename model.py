import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # RNN 레이어 정의
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        
        # 완전 연결층 정의
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # RNN 레이어를 통한 순전파
        output, hidden = self.rnn(input, hidden)
        
        # 출력을 (배치 크기 * 시퀀스 길이, 은닉 크기)로 변환
        output = output.view(-1, self.hidden_size)
        
        # 완전 연결층 적용
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        # 숨겨진 상태를 모두 0으로 초기화
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CharLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # LSTM 레이어 정의
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        
        # 완전 연결층 정의
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # LSTM 레이어를 통한 순전파
        output, hidden = self.lstm(input, hidden)
        
        # 출력을 (배치 크기 * 시퀀스 길이, 은닉 크기)로 변환
        output = output.view(-1, self.hidden_size)
        
        # 완전 연결층 적용
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        # 숨겨진 상태와 셀 상태를 모두 0으로 초기화
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
