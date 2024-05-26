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
