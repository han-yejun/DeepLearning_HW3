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

![그래프](https://github.com/han-yejun/DeepLearning_HW3/blob/main/Training%20and%20Validation%20Loss%20for%20RNN%20and%20LSTM.png)


## 5-1.bestmodel.py
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
        
            best_val_loss = float('inf')  # 초기화
            best_model_path = 'best_model.pth'  # 최적 모델의 경로
        
            for epoch in range(num_epochs):
                trn_loss = train(model, train_loader, device, criterion, optimizer)
                val_loss = validate(model, val_loader, device, criterion)
        
                print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {trn_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
                # 현재 검증 손실이 이전까지의 최적 손실보다 낮으면 모델 저장
                if val_loss < best_val_loss:
                    torch.save(model.state_dict(), best_model_path)
                    best_val_loss = val_loss
                    print("Best model saved!")
        
        if __name__ == '__main__':
            main()

챗지피티한테 generate.py 해주는대로 진행할 거면 best_model.pth파일 필요하다며 협박당해서... 만들었다...

## 5-2.generate.py

        import torch
        import numpy as np
        from model import CharRNN, CharLSTM
        
        def generate(model, seed_characters, temperature, char_to_idx, idx_to_char):
            # seed_input을 (1, 시퀀스 길이, 입력 크기)의 형태로 생성
            seed_input = torch.tensor([[char_to_idx[ch] for ch in seed_characters]]).to(device)
            seed_input = nn.functional.one_hot(seed_input, num_classes=len(char_to_idx)).float()
        
            # hidden 상태 초기화
            hidden = model.init_hidden(1)
        
            # 생성된 문자열 초기화
            generated_chars = seed_characters
        
            # 최소 100자 이상 생성
            for _ in range(len(seed_characters), 100):
                output, hidden = model(seed_input, hidden)
                # 다항 분포로부터 샘플링
                output_dist = output.div(temperature).exp().squeeze()  # squeeze 함수를 사용하여 크기를 (출력 크기,)로 변경
                top_char_idx = torch.multinomial(output_dist, 1)[0].item()  # [0]을 사용하여 첫 번째 차원을 제거하여 스칼라 값을 추출합니다.
                # 생성된 문자를 추가
                generated_char = idx_to_char[top_char_idx]
                generated_chars += generated_char
                # 다음 입력으로 사용할 문자를 준비
                seed_input = torch.tensor([[top_char_idx]]).to(device)
                seed_input = nn.functional.one_hot(seed_input, num_classes=len(char_to_idx)).float()
        
            return generated_chars
        
        def main():
            # 훈련된 모델 로드
            # 최상의 검증 결과를 보인 모델을 선택
            model = CharLSTM(input_size, hidden_size, output_size, num_layers).to(device)
            # 혹은: model = CharRNN(input_size, hidden_size, output_size, num_layers).to(device)
            model.load_state_dict(torch.load('best_model.pth'))  # 최상의 모델 경로로 업데이트
            
            # 문자 매핑 로드
            with open('char_to_idx.npy', 'rb') as f:
                char_to_idx = np.load(f, allow_pickle=True).item()
            with open('idx_to_char.npy', 'rb') as f:
                idx_to_char = np.load(f, allow_pickle=True).item()
        
            # 초기 문자열 설정하고 텍스트 생성
            seed_characters = ['T', 'h', 'e', ' ', 's']
            temperature = 0.5  # 원하는 대로 온도 조정 가능
            generated_samples = generate(model, seed_characters, temperature, char_to_idx, idx_to_char)
        
            # 생성된 샘플 출력
            print(f'생성된 샘플: {generated_samples}')
        
        if __name__ == '__main__':
            main()

결과

        생성된 샘플: ['T', 'h', 'e', ' ', 's', 'h', 'o', 'u', 'l', 'd', ' ', 'h', 'a', 'v', 'e', ' ', 'b', 'e', 'e', 'n', ' ', 't', 'h', 'e', ' ', 'm', 'i', 'n', 'i', 's', 't', 'e', 'r', 's', ' ', 't', 'o', ' ', 'm', 'y', ' ', 'f', 'r', 'i', 'e', 'n', 'd', 's', ',', '\n', 'W', 'h', 'o', ' ', 't', 'h', 'a', 't', ' ', 'w', 'a', 's', ' ', 'w', 'i', 't', 'h', 'a', 'l', '!', ' ', 'T', 'h', 'i', 's', ' ', 'm', 'o', 'r', 'n', 'i', 'n', 'g', ' ', 'f', 'o', 'r', ' ', 'm', 'e', ',', '\n', 'A', 'n', 'd', ' ', 'l', 'o', 'o', 'k']

시작 부분에 "The should have been the ministers to my friends," 라는 문장이 생성되었음. 이어지는 텍스트도 의미 있는 문장처럼 보임. 잘 된건지 모르겠음.. 뭐하는 건지 모르겠지만 뭔가 진행되고 있어서 무섭다.

## 6.temperatures.py
온도를 다양하게 바꾸어야 한다고 해서 온도라고 이름을 지엇다.. 아 온도가 뭐지..

        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset
        
        class Shakespeare(Dataset):
            """ 셰익스피어 데이터셋 """
        
            def __init__(self, input_file):
                # 입력 파일을 로드하고 문자 사전을 구축합니다.
                with open(input_file, 'r') as f:
                    text = f.read()
                self.chars = sorted(list(set(text)))
                self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
                self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
                # 문자 사전을 사용하여 문자 인덱스의 리스트를 만듭니다.
                self.data = [self.char_to_idx[ch] for ch in text]
        
                # 데이터를 30 길이의 시퀀스로 분할합니다.
                self.seq_length = 30
                self.data_len = len(self.data) - self.seq_length
                self.input_sequences = [self.data[i:i+self.seq_length] for i in range(self.data_len)]
                self.target_sequences = [self.data[i+self.seq_length] for i in range(self.data_len)]
        
            def __len__(self):
                return self.data_len
        
            def __getitem__(self, idx):
                input_seq = torch.tensor(self.input_sequences[idx])
                target = torch.tensor(self.target_sequences[idx])
                return input_seq, target
        
        class CharLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(CharLSTM, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
        
            def forward(self, input, hidden):
                out, hidden = self.lstm(input, hidden)
                out = self.fc(out[:, -1, :])
                return out, hidden
        
            def init_hidden(self, batch_size):
                return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                        torch.zeros(self.num_layers, batch_size, self.hidden_size))
        
        
        def generate_with_temperature(model, seed_characters, temperatures, char_to_idx, idx_to_char):
            """ Generate characters with different temperatures
        
            Args:
                model: trained model
                seed_characters: seed characters
                temperatures: list of temperatures to try
                char_to_idx: character to index mapping
                idx_to_char: index to character mapping
        
            Returns:
                generated_samples: dictionary containing generated samples for each temperature
            """
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
            generated_samples = {}
        
            # 각 온도에 대해 샘플을 생성합니다.
            for temp in temperatures:
                # 시드 입력을 사용하여 샘플을 생성합니다.
                generated_chars = seed_characters.copy()
                hidden, cell = model.init_hidden(1)
                hidden = hidden.to(device)
                cell = cell.to(device)
        
                # 시드 입력을 준비합니다.
                seed_input = torch.zeros((1, len(seed_characters), len(characters)), dtype=torch.float32).to(device)
                for i, ch in enumerate(seed_characters):
                    seed_input[0, i, char_to_idx[ch]] = 1.0
        
                # 생성된 샘플을 반복해서 생성합니다.
                for _ in range(len(seed_characters), 100):  # 최소 100자 이상 생성
                    output, hidden = model(seed_input, hidden)
        
                    # 다항 분포로부터 샘플링합니다.
                    output_dist = output.div(temp).exp().squeeze()  # squeeze 함수를 사용하여 크기를 (출력 크기,)로 변경
                    top_char_idx = torch.multinomial(output_dist, 1).item()
        
                    # 생성된 문자를 추가합니다.
                    generated_char = idx_to_char[top_char_idx]
                    generated_chars.append(generated_char)
        
                    # 다음 입력을 준비합니다. (이전 문자를 포함하여)
                    seed_input = torch.tensor([[char_to_idx[ch] for ch in generated_chars[-len(seed_characters):]]]).to(device)
        
                # 생성된 샘플을 저장합니다.
                generated_samples[temp] = ''.join(generated_chars)
        
            return generated_samples
        
        if __name__ == '__main__':
            # 데이터셋을 로드합니다.
            input_file = "shakespeare_train.txt"  # 여기를 셰익스피어 데이터셋 파일의 경로로 변경하세요
            dataset = Shakespeare(input_file)
        
            # 총 문자 개수, 문자 사전, 입력 시퀀스 길이, 총 데이터 샘플 수를 출력합니다.
            print("총 문자 개수:", len(dataset.chars))
            print("문자 사전:", dataset.char_to_idx)
            print("입력 시퀀스 길이:", dataset.seq_length)
            print("총 데이터 샘플 수:", len(dataset))
        
결과

        총 문자 개수: 62
        문자 사전: {'\n': 0, ' ': 1, '!': 2, '&': 3, "'": 4, ',': 5, '-': 6, '.': 7, ':': 8, ';': 9, '?': 10, 'A': 11, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19, 'J': 20, 'K': 21, 'L': 22, 'M': 23, 'N': 24, 'O': 25, 'P': 26, 'Q': 27, 'R': 28, 'S': 29, 'T': 30, 'U': 31, 'V': 32, 'W': 33, 'Y': 34, 'Z': 35, 'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45, 'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55, 'u': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61}
        입력 시퀀스 길이: 30
        총 데이터 샘플 수: 268300


총 문자 개수: 데이터셋에 있는 고유한 문자의 총 수는 62.
문자 사전: 각 문자에 대한 인덱스를 가진 사전이 있음. 이것은 각 문자를 모델에 입력할 때 사용됨.
입력 시퀀스 길이: 데이터가 입력될 때 사용되는 시퀀스의 길이는 30. 이는 모델이 한 번에 처리하는 문자의 수.
총 데이터 샘플 수: 데이터셋에는 총 268,300개의 샘플이 있음. 이는 모델 학습에 사용되는 전체 데이터의 크기.

이 정보는 모델을 설정하고 데이터를 로드하는 데 필요하다고한다(?). 



The should have been the ministers to my friends라는 문장이 나왔을 때만 해도 뭔가 기뻤는데, 이후로 오류 세례와 챗지피티와의 밀당을 하면서 점점 잘못됨을 느꼈다.. 결국 마지막엔 정말 뭘 한건지 모르게 됐다.







