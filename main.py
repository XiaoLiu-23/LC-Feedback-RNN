import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. 随机种子
# =========================
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. 数据生成
#    输入是一维随时间变化的量
#    一旦首次超过 threshold，就生成一个先升后降的三角 bump target
# =========================
def make_triangle_bump(T, start_idx, rise_len=15, fall_len=20, peak=1.0):
    """
    T: 总时长
    start_idx: bump 开始时刻
    rise_len: 上升段长度
    fall_len: 下降段长度
    peak: 峰值
    """
    y = np.zeros(T, dtype=np.float32)

    # 上升
    for i in range(rise_len):
        t = start_idx + i
        if 0 <= t < T:
            y[t] = peak * (i + 1) / rise_len

    # 下降
    for i in range(fall_len):
        t = start_idx + rise_len + i
        if 0 <= t < T:
            y[t] = peak * max(0.0, 1.0 - (i + 1) / fall_len)

    return y


def generate_one_sequence(
    T=120,
    threshold=0.7,
    noise_std=0.03,
    rise_len=15,
    fall_len=20,
    min_cross_time=20,
    max_cross_time=70
):
    """
    生成一个一维输入序列 u(t)
    让它在某个时间点附近超过阈值
    target 是 threshold crossing 后触发的 triangle bump
    """
    u = np.zeros(T, dtype=np.float32)

    # 先生成一个平滑变化的一维输入
    cross_time = np.random.randint(min_cross_time, max_cross_time)

    baseline = np.random.uniform(0.0, 0.3)
    pre_slope = np.random.uniform(0.002, 0.01)
    post_jump = np.random.uniform(0.4, 0.8)

    for t in range(T):
        if t < cross_time:
            u[t] = baseline + pre_slope * t
        else:
            # crossing 后抬高
            u[t] = baseline + pre_slope * cross_time + post_jump + 0.002 * (t - cross_time)

    u += np.random.randn(T).astype(np.float32) * noise_std

    # 找到第一次超过阈值的位置
    above = np.where(u > threshold)[0]
    if len(above) == 0:
        # 如果没超过，就强制在 cross_time 后超过
        u[cross_time:] += (threshold + 0.1)
        above = np.where(u > threshold)[0]

    trigger_t = int(above[0])

    # 目标输出：三角 bump
    y = make_triangle_bump(
        T=T,
        start_idx=trigger_t,
        rise_len=rise_len,
        fall_len=fall_len,
        peak=1.0
    )

    # shape -> [T, 1]
    u = u[:, None]
    y = y[:, None]

    return u.astype(np.float32), y.astype(np.float32), trigger_t


class ThresholdBumpDataset(Dataset):
    def __init__(self, n_samples=2000, T=120, threshold=0.7):
        self.inputs = []
        self.targets = []
        self.triggers = []

        for _ in range(n_samples):
            u, y, trig = generate_one_sequence(T=T, threshold=threshold)
            self.inputs.append(u)
            self.targets.append(y)
            self.triggers.append(trig)

        self.inputs = np.stack(self.inputs, axis=0)   # [N, T, 1]
        self.targets = np.stack(self.targets, axis=0) # [N, T, 1]
        self.triggers = np.array(self.triggers)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


# =========================
# 3. E/I Balance RNN
#    - rate RNN
#    - output feedback
#    - Dale's law
# =========================
class EIRNN(nn.Module):
    def __init__(
        self,
        n_e=80,
        n_i=20,
        input_dim=1,
        output_dim=1,
        tau=10.0,
        dt=1.0,
        sigma_rec=0.02,
    ):
        super().__init__()

        self.n_e = n_e
        self.n_i = n_i
        self.n = n_e + n_i
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = dt / tau
        self.sigma_rec = sigma_rec

        # 输入权重
        self.w_in = nn.Parameter(torch.randn(self.n, input_dim) / math.sqrt(input_dim))

        # feedback 权重：把上一时刻输出送回网络
        self.w_fb = nn.Parameter(torch.randn(self.n, output_dim) / math.sqrt(output_dim))

        # 输出层
        self.w_out = nn.Parameter(torch.randn(output_dim, self.n) / math.sqrt(self.n))
        self.b_out = nn.Parameter(torch.zeros(output_dim))

        # recurrent 权重的“原始参数”
        # 通过 softplus + sign mask 实现 E/I 约束
        self.w_rec_raw = nn.Parameter(torch.randn(self.n, self.n) / math.sqrt(self.n))

        # bias
        self.bias = nn.Parameter(torch.zeros(self.n))

        # 符号 mask：按照列来控制 presynaptic neuron 的输出符号
        # 前 n_e 列是 excitatory，后 n_i 列是 inhibitory
        sign = torch.ones(self.n)
        sign[n_e:] = -1.0
        self.register_buffer("dale_sign", sign)

        # 用于初始化时减弱自连接
        self.register_buffer("eye_mask", 1.0 - torch.eye(self.n))

    def recurrent_weight(self):
        """
        W_rec[:, j] 表示第 j 个神经元投射到所有 postsynaptic 单元的权重
        j 属于 E -> 非负
        j 属于 I -> 非正
        """
        w_mag = F.softplus(self.w_rec_raw)
        w = w_mag * self.dale_sign[None, :]  # 按列施加符号
        w = w * self.eye_mask                # 去掉自连接
        return w

    def forward(self, x, y_teacher=None, teacher_forcing_ratio=0.5):
        """
        x: [B, T, 1]
        y_teacher: [B, T, 1]
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.n, device=x.device)
        r = torch.tanh(h)

        w_rec = self.recurrent_weight()

        ys = []
        y_prev = torch.zeros(B, self.output_dim, device=x.device)

        for t in range(T):
            xt = x[:, t, :]  # [B, 1]

            # teacher forcing 的 feedback
            if (y_teacher is not None) and (t > 0):
                use_teacher = (torch.rand(1).item() < teacher_forcing_ratio)
                if use_teacher:
                    y_fb = y_teacher[:, t - 1, :]
                else:
                    y_fb = y_prev
            else:
                y_fb = y_prev

            rec_term = r @ w_rec.T
            inp_term = xt @ self.w_in.T
            fb_term = y_fb @ self.w_fb.T

            noise = self.sigma_rec * torch.randn_like(h)

            dh = -h + rec_term + inp_term + fb_term + self.bias + noise
            h = h + self.alpha * dh
            r = torch.tanh(h)

            y_t = r @ self.w_out.T + self.b_out
            ys.append(y_t.unsqueeze(1))
            y_prev = y_t

        y_seq = torch.cat(ys, dim=1)  # [B, T, 1]
        return y_seq

    def ei_balance_regularization(self):
        """
        一个简单的 balance 正则：
        希望每个 postsynaptic neuron 接收到的 E 与 I 总输入大致平衡
        """
        w = self.recurrent_weight()  # [N, N]

        w_e = w[:, :self.n_e].sum(dim=1)          # 正向总和
        w_i = -w[:, self.n_e:].sum(dim=1)         # 取负后表示抑制强度的正量

        # 希望 E 和 I 规模接近
        reg = ((w_e - w_i) ** 2).mean()
        return reg

    def firing_rate_regularization(self, x):
        """
        可选：限制活动不要太大
        """
        with torch.no_grad():
            pass
        return 0.0


# =========================
# 4. 训练
# =========================
def train_model():
    # 超参数
    T = 120
    threshold = 0.7
    batch_size = 64
    epochs = 80
    lr = 1e-3

    train_ds = ThresholdBumpDataset(n_samples=3000, T=T, threshold=threshold)
    val_ds = ThresholdBumpDataset(n_samples=400, T=T, threshold=threshold)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = EIRNN(
        n_e=80,
        n_i=20,
        input_dim=1,
        output_dim=1,
        tau=10.0,
        dt=1.0,
        sigma_rec=0.01,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        # teacher forcing 比例逐渐下降
        tf_ratio = max(0.1, 0.8 * (1 - epoch / epochs))

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x, y_teacher=y, teacher_forcing_ratio=tf_ratio)

            loss_task = F.mse_loss(pred, y)
            loss_balance = model.ei_balance_regularization()

            loss = loss_task + 1e-3 * loss_balance

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x, y_teacher=None, teacher_forcing_ratio=0.0)
                loss_task = F.mse_loss(pred, y)
                loss_balance = model.ei_balance_regularization()
                loss = loss_task + 1e-3 * loss_balance
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_ds)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | tf_ratio={tf_ratio:.3f}")

    return model, val_ds


# =========================
# 5. 可视化
# =========================
def plot_examples(model, dataset, n_show=5):
    model.eval()
    plt.figure(figsize=(12, 3 * n_show))

    idxs = np.random.choice(len(dataset), size=n_show, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(idxs):
            x, y = dataset[idx]
            x_in = x.unsqueeze(0).to(device)
            pred = model(x_in, y_teacher=None, teacher_forcing_ratio=0.0)[0].cpu().numpy()

            x_np = x.numpy().squeeze(-1)
            y_np = y.numpy().squeeze(-1)
            pred_np = pred.squeeze(-1)

            ax = plt.subplot(n_show, 1, i + 1)
            ax.plot(x_np, label="input u(t)")
            ax.plot(y_np, label="target bump")
            ax.plot(pred_np, label="pred output", linestyle="--")
            ax.axhline(0.7, color="gray", linestyle=":", label="threshold" if i == 0 else None)
            ax.legend(loc="upper right")
            ax.set_title(f"Example {idx}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model, val_ds = train_model()
    plot_examples(model, val_ds, n_show=5)