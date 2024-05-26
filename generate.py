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
