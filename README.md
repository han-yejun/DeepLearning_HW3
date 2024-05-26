# DeepLearning_HW3
딥러닝 수업 과제3

## 1.dataset.py 

        import torch
        from torch.utils.data import Dataset
        
        class Shakespeare(Dataset):
            """ 셰익스피어 데이터셋 """
        
            def __init__(self, input_file):
                # Step 1: 입력 파일을 로드하고 문자 사전을 구축합니다.
                with open(input_file, 'r') as f:
                    text = f.read()
                self.chars = sorted(list(set(text)))
                self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
                self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
                # Step 2: 문자 사전을 사용하여 문자 인덱스의 리스트를 만듭니다.
                self.data = [self.char_to_idx[ch] for ch in text]
        
                # Step 3: 데이터를 30 길이의 시퀀스로 분할합니다.
                self.seq_length = 30
                self.data_len = len(self.data) - self.seq_length
                self.input_sequences = [self.data[i:i+self.seq_length] for i in range(self.data_len)]
                self.target_sequences = [self.data[i+self.seq_length] for i in range(self.data_len)]
        
                # Save character mappings
                self.save_char_mappings()
        
            def __len__(self):
                return self.data_len
        
            def __getitem__(self, idx):
                input_seq = torch.tensor(self.input_sequences[idx])
                target = torch.tensor(self.target_sequences[idx])
                return input_seq, target
        
            def save_char_mappings(self):
                # Save character mappings as numpy files
                with open('char_to_idx.npy', 'wb') as f:
                    np.save(f, self.char_to_idx)
                with open('idx_to_char.npy', 'wb') as f:
                    np.save(f, self.idx_to_char)
        
        if __name__ == '__main__':
            # 데이터셋을 테스트합니다.
            input_file = "shakespeare_train.txt"  # 여기를 셰익스피어 데이터셋 파일의 경로로 변경하세요
            dataset = Shakespeare(input_file)
        
            # 몇 가지 통계 정보를 출력합니다.
            print("총 문자 개수:", len(dataset.chars))
            print("문자 사전:", dataset.char_to_idx)
            print("입력 시퀀스 길이:", dataset.seq_length)
            print("총 데이터 샘플 수:", len(dataset))
        
            # __getitem__ 메서드를 테스트합니다.
            idx = 0
            input_seq, target = dataset[idx]
            print("입력 시퀀스:", input_seq)
            print("타겟:", target)


결과

        총 문자 개수: 62
        문자 사전: {'\n': 0, ' ': 1, '!': 2, '&': 3, "'": 4, ',': 5, '-': 6, '.': 7, ':': 8, ';': 9, '?': 10, 'A': 11, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19, 'J': 20, 'K': 21, 'L': 22, 'M': 23, 'N': 24, 'O': 25, 'P': 26, 'Q': 27, 'R': 28, 'S': 29, 'T': 30, 'U': 31, 'V': 32, 'W': 33, 'Y': 34, 'Z': 35, 'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45, 'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55, 'u': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61}
        입력 시퀀스 길이: 30
        총 데이터 샘플 수: 268300
        입력 시퀀스: tensor([16, 44, 53, 54, 55,  1, 13, 44, 55, 44, 61, 40, 49,  8,  0, 12, 40, 41, 50, 53, 40,  1, 58, 40,  1, 51, 53, 50, 38, 40])
        타겟: tensor(40)


출력 결과를 보면 데이터셋이 제대로 구성되었고, 문자 개수는 62개이며 문자 사전이 올바르게 생성되었다. 입력 시퀀스의 길이는 30이며, 총 데이터 샘플 수는 268,300개다. 데이터셋이 제대로 구성되었으니 이제 이를 사용하여 신경망 모델을 훈련할 수 있겠다!


## 2.model.py 

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



## 3.main.py 

        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, SubsetRandomSampler
        from dataset import Shakespeare
        from model import CharRNN, CharLSTM
        import numpy as np
        
        def train(model, trn_loader, device, criterion, optimizer):
            """ Train function """
            model.train()
            trn_loss = 0.0
        
            for inputs, targets in trn_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 입력 데이터를 one-hot 인코딩
                inputs = nn.functional.one_hot(inputs, num_classes=62).float()
                
                # 초기화
                hidden = model.init_hidden(inputs.size(0))
                if isinstance(hidden, tuple):
                    hidden = tuple(h.to(device) for h in hidden)
                else:
                    hidden = hidden.to(device)
                
                optimizer.zero_grad()
                
                # 순전파
                outputs, hidden = model(inputs, hidden)
                
                # 손실 계산 및 역전파
                loss = criterion(outputs, targets.view(-1))
                loss.backward()
                
                # 옵티마이저 단계
                optimizer.step()
                
                trn_loss += loss.item()
            
            trn_loss /= len(trn_loader)
            return trn_loss
        
        def validate(model, val_loader, device, criterion):
            """ Validate function """
            model.eval()
            val_loss = 0.0
        
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # 입력 데이터를 one-hot 인코딩
                    inputs = nn.functional.one_hot(inputs, num_classes=62).float()
                    
                    # 초기화
                    hidden = model.init_hidden(inputs.size(0))
                    if isinstance(hidden, tuple):
                        hidden = tuple(h.to(device) for h in hidden)
                    else:
                        hidden = hidden.to(device)
                    
                    # 순전파
                    outputs, hidden = model(inputs, hidden)
                    
                    # 손실 계산
                    loss = criterion(outputs, targets.view(-1))
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            return val_loss
        
        def main():
            """ Main function """
            # 하이퍼파라미터 설정
            input_size = 62
            hidden_size = 128
            output_size = 62
            num_layers = 2
            batch_size = 64
            num_epochs = 10
            learning_rate = 0.002
        
            # 디바이스 설정
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
            # 데이터셋 및 데이터 로더 생성
            dataset = Shakespeare('shakespeare_train.txt')
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(0.2 * dataset_size))
        
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]
        
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
        
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        
            # 모델 초기화 (CharRNN 또는 CharLSTM 중 하나 선택)
            model = CharLSTM(input_size, hidden_size, output_size, num_layers).to(device)
            # model = CharRNN(input_size, hidden_size, output_size, num_layers).to(device)
            
            # 손실 함수 및 옵티마이저 설정
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
            for epoch in range(num_epochs):
                trn_loss = train(model, train_loader, device, criterion, optimizer)
                val_loss = validate(model, val_loader, device, criterion)
        
                print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {trn_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        if __name__ == '__main__':
            main()

결과

        Epoch 1/10, Training Loss: 1.7882, Validation Loss: 1.4490
        Epoch 2/10, Training Loss: 1.3485, Validation Loss: 1.2822
        Epoch 3/10, Training Loss: 1.2275, Validation Loss: 1.1935
        Epoch 4/10, Training Loss: 1.1512, Validation Loss: 1.1333
        Epoch 5/10, Training Loss: 1.0976, Validation Loss: 1.0932
        Epoch 6/10, Training Loss: 1.0582, Validation Loss: 1.0607
        Epoch 7/10, Training Loss: 1.0279, Validation Loss: 1.0341
        Epoch 8/10, Training Loss: 1.0040, Validation Loss: 1.0190
        Epoch 9/10, Training Loss: 0.9848, Validation Loss: 1.0013
        Epoch 10/10, Training Loss: 0.9685, Validation Loss: 0.9856


## 4.compare.py

        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, SubsetRandomSampler
        from dataset import Shakespeare
        from model import CharRNN, CharLSTM
        import numpy as np
        import matplotlib.pyplot as plt
        
        def train(model, trn_loader, device, criterion, optimizer):
            """ Train function """
            model.train()
            trn_loss = 0.0
        
            for inputs, targets in trn_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 입력 데이터를 one-hot 인코딩
                inputs = nn.functional.one_hot(inputs, num_classes=62).float()
                
                # 초기화
                hidden = model.init_hidden(inputs.size(0))
                if isinstance(hidden, tuple):
                    hidden = tuple(h.to(device) for h in hidden)
                else:
                    hidden = hidden.to(device)
                
                optimizer.zero_grad()
                
                # 순전파
                outputs, hidden = model(inputs, hidden)
                
                # 손실 계산 및 역전파
                loss = criterion(outputs, targets.view(-1))
                loss.backward()
                
                # 옵티마이저 단계
                optimizer.step()
                
                trn_loss += loss.item()
            
            trn_loss /= len(trn_loader)
            return trn_loss
        
        def validate(model, val_loader, device, criterion):
            """ Validate function """
            model.eval()
            val_loss = 0.0
        
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # 입력 데이터를 one-hot 인코딩
                    inputs = nn.functional.one_hot(inputs, num_classes=62).float()
                    
                    # 초기화
                    hidden = model.init_hidden(inputs.size(0))
                    if isinstance(hidden, tuple):
                        hidden = tuple(h.to(device) for h in hidden)
                    else:
                        hidden = hidden.to(device)
                    
                    # 순전파
                    outputs, hidden = model(inputs, hidden)
                    
                    # 손실 계산
                    loss = criterion(outputs, targets.view(-1))
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            return val_loss
        
        def main():
            """ Main function """
            # 하이퍼파라미터 설정
            input_size = 62
            hidden_size = 128
            output_size = 62
            num_layers = 2
            batch_size = 64
            num_epochs = 10
            learning_rate = 0.002
        
            # 디바이스 설정
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
            # 데이터셋 및 데이터 로더 생성
            dataset = Shakespeare('shakespeare_train.txt')
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(0.2 * dataset_size))
        
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]
        
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
        
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        
            # 모델 초기화
            models = {
                'RNN': CharRNN(input_size, hidden_size, output_size, num_layers).to(device),
                'LSTM': CharLSTM(input_size, hidden_size, output_size, num_layers).to(device)
            }
            
            # 손실 함수 및 옵티마이저 설정
            criterion = nn.CrossEntropyLoss()
        
            train_losses = {'RNN': [], 'LSTM': []}
            val_losses = {'RNN': [], 'LSTM': []}
        
            for model_name, model in models.items():
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                
                print(f"Training {model_name} model")
                for epoch in range(num_epochs):
                    trn_loss = train(model, train_loader, device, criterion, optimizer)
                    val_loss = validate(model, val_loader, device, criterion)
        
                    train_losses[model_name].append(trn_loss)
                    val_losses[model_name].append(val_loss)
        
                    print(f'Epoch {epoch+1}/{num_epochs}, {model_name} Training Loss: {trn_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
            # 손실 값 플로팅
            epochs = range(1, num_epochs + 1)
            plt.figure(figsize=(12, 6))
            
            for model_name in models.keys():
                plt.plot(epochs, train_losses[model_name], label=f'{model_name} Train Loss')
                plt.plot(epochs, val_losses[model_name], label=f'{model_name} Validation Loss')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss for RNN and LSTM')
            plt.legend()
            plt.show()
        
        if __name__ == '__main__':
            main()

결과

        Training RNN model
        Epoch 1/10, RNN Training Loss: 1.7259, Validation Loss: 1.4999
        Epoch 2/10, RNN Training Loss: 1.4569, Validation Loss: 1.4265
        Epoch 3/10, RNN Training Loss: 1.4044, Validation Loss: 1.3875
        Epoch 4/10, RNN Training Loss: 1.3762, Validation Loss: 1.3730
        Epoch 5/10, RNN Training Loss: 1.3586, Validation Loss: 1.3578
        Epoch 6/10, RNN Training Loss: 1.3468, Validation Loss: 1.3519
        Epoch 7/10, RNN Training Loss: 1.3383, Validation Loss: 1.3441
        Epoch 8/10, RNN Training Loss: 1.3317, Validation Loss: 1.3352
        Epoch 9/10, RNN Training Loss: 1.3268, Validation Loss: 1.3316
        Epoch 10/10, RNN Training Loss: 1.3230, Validation Loss: 1.3282
        Training LSTM model
        Epoch 1/10, LSTM Training Loss: 1.8739, Validation Loss: 1.5230
        Epoch 2/10, LSTM Training Loss: 1.4257, Validation Loss: 1.3491
        Epoch 3/10, LSTM Training Loss: 1.2932, Validation Loss: 1.2536
        Epoch 4/10, LSTM Training Loss: 1.2092, Validation Loss: 1.1870
        Epoch 5/10, LSTM Training Loss: 1.1497, Validation Loss: 1.1419
        Epoch 6/10, LSTM Training Loss: 1.1042, Validation Loss: 1.1008
        Epoch 7/10, LSTM Training Loss: 1.0690, Validation Loss: 1.0693
        Epoch 8/10, LSTM Training Loss: 1.0411, Validation Loss: 1.0508
        Epoch 9/10, LSTM Training Loss: 1.0183, Validation Loss: 1.0338
        Epoch 10/10, LSTM Training Loss: 0.9996, Validation Loss: 1.0167

![LeNet5 - Loss & Accuracy](https://github.com/han-yejun/DeepLearning_HW2/blob/main/LeNet5%20-%20Loss%20%26%20Accuracy.png)
