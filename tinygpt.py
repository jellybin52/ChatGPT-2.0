import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import urllib.request
import time
from tqdm import tqdm

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

# [3] 학습 루틴 (에포크 단위 실시간 진행률 표시)
TOTAL_EPOCHS = 1500
start_time = time.time()
print(f"학습 시작 ({device})")

# tqdm을 사용하여 전체 1500 에포크 진행률을 표시
pbar = tqdm(range(TOTAL_EPOCHS), desc="학습 진행도", unit="epoch")

for epoch in pbar:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # 에포크마다 평균 loss를 계산하여 진행바 옆에 출력
    avg_loss = total_loss / len(loader)
    elapsed = time.time() - start_time
    pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Time": f"{elapsed:.0f}s"})

print(f"\n학습 완료! 총 소요 시간: {time.time() - start_time:.1f}초")

# [4] 결과 저장
@torch.no_grad()
def generate():
    model.eval()
    idx = torch.zeros((1, 64), dtype=torch.long, device=device)
    out = []
    for _ in range(1000):
        logits = model(idx)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx[:, 1:], next_idx], dim=1)
        out.append(itos[next_idx.item()])
    return "".join(out)

with open("result_for_professor.txt", "w", encoding="utf-8") as f:
    f.write(generate())
print("결과가 'result_for_professor.txt' 파일로 저장되었습니다.")