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
