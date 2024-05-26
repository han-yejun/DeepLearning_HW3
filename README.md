# DeepLearning_HW3
딥러닝 수업 과제3

## dataset.py 

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


총 문자 개수: 62
문자 사전: {'\n': 0, ' ': 1, '!': 2, '&': 3, "'": 4, ',': 5, '-': 6, '.': 7, ':': 8, ';': 9, '?': 10, 'A': 11, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19, 'J': 20, 'K': 21, 'L': 22, 'M': 23, 'N': 24, 'O': 25, 'P': 26, 'Q': 27, 'R': 28, 'S': 29, 'T': 30, 'U': 31, 'V': 32, 'W': 33, 'Y': 34, 'Z': 35, 'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45, 'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55, 'u': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61}
입력 시퀀스 길이: 30
총 데이터 샘플 수: 268300
입력 시퀀스: tensor([16, 44, 53, 54, 55,  1, 13, 44, 55, 44, 61, 40, 49,  8,  0, 12, 40, 41,
        50, 53, 40,  1, 58, 40,  1, 51, 53, 50, 38, 40])
타겟: tensor(40)
