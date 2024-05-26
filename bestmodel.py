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
