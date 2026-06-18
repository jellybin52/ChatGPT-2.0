import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import urllib.request
import time
from tqdm import tqdm  # tqdm 라이브러리 임포트

if torch.cuda.is_available():
    print(f"✅ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
else:
    print("❌ GPU 감지 실패. CPU로만 학습합니다.")

# [1] 데이터 파이프라인
DATA_FILE = "custom_data.txt"
if not Path(DATA_FILE).exists():
    url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
    urllib.request.urlretrieve(url, DATA_FILE)

text = open(DATA_FILE, "r", encoding="utf-8").read()
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

class NextTokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data, self.block_size = data, block_size
    def __len__(self): return len(self.data) - self.block_size
    def __getitem__(self, idx):
        return self.data[idx:idx+self.block_size], self.data[idx+1:idx+self.block_size+1]

dataset = NextTokenDataset(data, 64)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# [2] 모델 정의
class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 128)
        self.pos = nn.Embedding(64, 128)
        self.blocks = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True) for _ in range(2)])
        self.head = nn.Linear(128, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        x = self.blocks(x)
        return self.head(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyGPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# [3] 학습 루틴 (50 에포크 단위 tqdm 업데이트)
TOTAL_EPOCHS = 1500
UPDATE_INTERVAL = 50
print(f"학습 시작 ({device})")

running_loss = 0.0
# tqdm 객체 생성
pbar = tqdm(total=TOTAL_EPOCHS, desc="학습 진행도", unit="epoch")

for epoch in range(1, TOTAL_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    running_loss += (epoch_loss / len(loader))
    
    # 50 에포크마다 진행바 업데이트 및 평균 Loss 출력
    if epoch % UPDATE_INTERVAL == 0:
        avg_loss = running_loss / UPDATE_INTERVAL
        pbar.update(UPDATE_INTERVAL)
        pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        running_loss = 0.0

pbar.close()
print("학습 완료!")