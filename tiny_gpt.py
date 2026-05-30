import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import urllib.request

try:
    from tqdm import tqdm
except ImportError:
    # tqdm이 없어도 set_postfix 메서드를 호출할 수 있도록 더미 클래스를 정의합니다.
    class tqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def set_postfix(self, **kwargs):
            pass

# =====================================================================
# [1] 데이터 파이프라인: 텍스트 다운로드 및 토큰화 데이터셋 구축
# =====================================================================

# 데이터셋(Tiny Shakespeare)이 없을 경우 원격 저장소에서 순수 파이썬 코드로 다운로드
if not Path("input.txt").exists():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, "input.txt")

# 텍스트 로드 및 고유 문자 집합을 활용한 어휘 사전(Vocabulary) 빌드
text = open("input.txt", "r", encoding="utf-8").read()
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}  # 문자 -> 고유 인덱스 매핑
itos = {i: ch for ch, i in stoi.items()}      # 인덱스 -> 문자 역매핑
vocab_size = len(chars)

# 전체 말뭉치(Corpus)를 숫자 인덱스 텐서로 변환
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

class NextTokenDataset(Dataset):
    """ 언어 모델 학습을 위해 현재 문맥(x)과 다음 타겟 문자(y) 쌍을 생성하는 클래스 """
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # x는 현재 위치부터 block_size 크기만큼, y는 우측으로 한 칸 밀린 정답 시퀀스
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

# 배치 및 블록 크기 설정 후 데이터 로더 인스턴스화
block_size = 64
dataset = NextTokenDataset(data, block_size)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
xb, yb = next(iter(loader))


# =====================================================================
# [2] 어텐션 레이어: Multi-Head 및 인과 관계 마스킹(Causal Masking) 구현
# =====================================================================

class Head(nn.Module):
    """ 단일 어텐션 헤드: 문맥 내 토큰 간의 가중 관계 스코어를 계산하는 최소 단위 """
    def __init__(self, emb_dim, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        # 생성 모델의 핵심인 미래 토큰 차단용 하삼각 행렬(Tril) 등록
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # [B, T, head_size]
        q = self.query(x) # [B, T, head_size]
        v = self.value(x) # [B, T, head_size]

        # 스케일드 닷 프로덕트 어텐션(Scaled Dot-Product Attention) 스코어 계산
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5) # [B, T, T]
        # 미래 시점에 해당하는 위치를 음의 무한대로 밀어내어 소프트맥스 시 확률을 0으로 차단
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # 가중치와 가치(Value) 벡터를 결합하여 최종 컨텍스트 벡터 도출
        out = wei @ v # [B, T, head_size]
        return out

class MultiHeadAttention(nn.Module):
    """ 여러 개의 독립적인 어텐션 헤드를 병렬로 운용하여 다각도의 문맥을 포착하는 레이어 """
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList([Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(emb_dim, emb_dim)  # 병렬 연산 결과를 다시 통합하는 선형 레이어
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 각각의 헤드가 연산한 결과를 채널 차원 축으로 단순 결합(Concatenation)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


# =====================================================================
# [3] 네트워크 확장: FeedForward 및 트랜스포머 블록(Block) 구성
# =====================================================================

class FeedForward(nn.Module):
    """ 각 토큰별 포지션에 독립적으로 적용되는 고차원 비선형 표현 레이어 (MLP) """
    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),  # 표현력을 위해 일시적으로 차원을 4배 확장
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),  # 원래 모델 차원으로 다시 축소
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ 어텐션 레이어와 피드포워드를 잔차 연결(Residual Connection)로 이어붙인 기본 아키텍처 단위 """
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.sa = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffwd = FeedForward(emb_dim, dropout)

    def forward(self, x):
        # Pre-Layer Normalization 패턴 및 스킵 커넥션을 통한 그라디언트 흐름 안정화
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# =====================================================================
# [4] 엔드투엔드 모델 정의: Tiny GPT 아키텍처 조립
# =====================================================================

class TinyGPT(nn.Module):
    """ 완성된 모듈들을 적층하여 빌드한 미니 사양의 GPT 디코더 모델 """
    def __init__(self, vocab_size, block_size, emb_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(block_size, emb_dim)
        # 설계된 트랜스포머 블록을 수직으로 깊게 쌓아 올림 (Deep Stacking)
        self.blocks = nn.Sequential(*[
            Block(emb_dim, num_heads, block_size, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)  # 최종 확률 매핑 헤드

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        
        # 문자의 의미적 표현 공간과 순서적 위치 표현 공간을 더해 초기 텐서 구성
        tok = self.token_embedding(x)
        pos = self.position_embedding(pos)[None]  # 파이토치 브로드캐스팅용 차원 추가
        h = tok + pos
        
        # 딥 스택 블록과 정규화를 차례로 거쳐 최종 예측 로짓 출력
        h = self.blocks(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits

model = TinyGPT(vocab_size, block_size)
logits = model(xb)


# =====================================================================
# [5] 학습 루틴: 손실 함수 설정 및 에포크 최적화 제어
# =====================================================================

def sequence_cross_entropy(logits, targets):
    """ 3차원 시퀀스 데이터 처리를 위한 크로스 엔트로피 비용 함수 """
    return F.cross_entropy(logits.transpose(1, 2), targets)

def train_one_epoch(model, loader, optimizer, device, max_steps=None):
    """ 1 에포크 단위의 순전파-역전파 루프 가중치 최적화 함수 """
    model.train()
    total_loss, total_count = 0.0, 0
    pbar = tqdm(enumerate(loader), total=max_steps if max_steps else len(loader), desc="Training")
    for step, (xb, yb) in pbar:
        xb, yb = xb.to(device), yb.to(device)
        
        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
        pbar.set_postfix(loss=loss.item())
        if max_steps is not None and step + 1 >= max_steps:
            break
    return total_loss / total_count

# 디바이스 바인딩 및 가중치 최적화용 AdamW 옵티마이저 할당
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyGPT(vocab_size, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 100 에포크 동안 모델 학습 가동
for epoch in range(100):
    train_loss = train_one_epoch(model, loader, optimizer, device, max_steps=300)
    if epoch % 10 == 0 or epoch == 99:
        print(f"epoch {epoch:2d} | train loss {train_loss:.4f}")


# =====================================================================
# [6] 생성 및 추론: 자동 문장 완성(Auto-regressive Generation) 테스트
# =====================================================================

@torch.no_grad()
def sample_gpt(model, block_size, stoi, itos, device, start_text="ROMEO:", max_new_tokens=400):
    """ 주어진 시작 프롬프트로부터 순차적으로 다음 문자를 예측 및 샘플링하는 함수 """
    model.eval()
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)
    for ch in start_text:
        if ch in stoi:
            ix = torch.tensor([[stoi[ch]]], device=device)
            context = torch.cat([context[:, 1:], ix], dim=1)
            
    out = list(start_text)
    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :]  # 마지막 타입 스텝의 출력 분포만 추려냄
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)  # 확률 분포 기반 무작위 확률 추출
        out.append(itos[ix.item()])
        context = torch.cat([context[:, 1:], ix], dim=1)  # 생성된 문자를 다시 입력 버퍼에 밀어 넣음
    return "".join(out)

# 최종 셰익스피어 풍의 문자 생성 추론 테스트 출력
print("\n[추론 결과 샘플링]")
print(sample_gpt(model, block_size, stoi, itos, device, start_text="ROMEO:", max_new_tokens=500))
