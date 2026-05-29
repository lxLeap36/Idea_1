"""
标准单分区 FDAF 线性自适应滤波器。

本实现采用标准 overlap-save 频域卷积结构。

注意：
    这里的“无重叠帧”指外部实验按 block_size 个新样本一帧输入；
    FDAF 内部仍然需要 overlap-save 缓存历史输入，
    否则频域卷积会产生循环卷积混叠。

第一版约束：
    为了保证实现清晰，建议 filter_length <= block_size。
    如果后续要支持 filter_length >> block_size，
    应该升级为 partitioned FDAF。
"""

import numpy as np


class FDAF:
    """
    单分区 Frequency-Domain Adaptive Filter。

    参数
    ----
    filter_length : int
        线性滤波器长度。第一版建议等于 block_size。

    block_size : int
        每帧新输入样本数。外部无重叠流式处理时，一帧就是 block_size 个样本。

    step_size : float
        频域 NLMS 更新步长。

    eps : float
        防止除零的小常数。

    leakage : float
        泄漏因子。0 表示不泄漏。
    """

    def __init__(
        self,
        filter_length: int,
        block_size: int,
        step_size: float = 0.1,
        eps: float = 1e-6,
        leakage: float = 0.0,
    ):
        self.filter_length = int(filter_length)
        self.block_size = int(block_size)
        self.step_size = float(step_size)
        self.eps = float(eps)
        self.leakage = float(leakage)

        if self.filter_length <= 0:
            raise ValueError("filter_length 必须为正数。")
        if self.block_size <= 0:
            raise ValueError("block_size 必须为正数。")
        if self.filter_length > self.block_size:
            raise ValueError(
                "当前单分区 FDAF 第一版要求 filter_length <= block_size。"
                "如果需要更长滤波器，请后续实现 partitioned FDAF。"
            )

        # overlap-save 常用 2B 点 FFT。
        self.fft_size = 2 * self.block_size

        self.reset()

    def reset(self):
        """重置 FDAF 状态。"""
        # 频域权重。
        self.W = np.zeros(self.fft_size, dtype=np.complex128)

        # overlap-save 的历史输入缓存，长度为 block_size。
        self.x_hist = np.zeros(self.block_size, dtype=np.float64)

    def _constrain_time_domain_filter(self):
        """
        对频域权重做时域约束。

        频域更新后，权重可能在 filter_length 之外产生非零尾巴。
        这里把时域冲激响应中 filter_length 之后的部分清零，
        再变回频域，保证滤波器有效长度受控。
        """
        h = np.fft.ifft(self.W).real

        h[self.filter_length:] = 0.0

        self.W = np.fft.fft(h, n=self.fft_size)

    def process_block(self, x_block, d_block):
        """
        处理一个 block。

        输入
        ----
        x_block : array, shape=(block_size,)
            当前远端输入块。

        d_block : array, shape=(block_size,)
            当前目标信号块，例如 echo_signal。

        返回
        ----
        y_block : array, shape=(block_size,)
            FDAF 估计的线性回声。

        e_block : array, shape=(block_size,)
            线性残差 d_block - y_block。
        """
        x_block = np.asarray(x_block, dtype=np.float64)
        d_block = np.asarray(d_block, dtype=np.float64)

        if len(x_block) != self.block_size:
            raise ValueError(
                f"x_block 长度应为 {self.block_size}，当前为 {len(x_block)}"
            )
        if len(d_block) != self.block_size:
            raise ValueError(
                f"d_block 长度应为 {self.block_size}，当前为 {len(d_block)}"
            )

        # overlap-save 输入：[上一块历史输入, 当前新输入]
        x_fft_buf = np.concatenate([self.x_hist, x_block], axis=0)

        X = np.fft.fft(x_fft_buf, n=self.fft_size)

        # 频域滤波。
        y_full = np.fft.ifft(self.W * X).real

        # overlap-save：前 block_size 点丢弃，后 block_size 点为有效输出。
        y_block = y_full[self.block_size:]

        e_block = d_block - y_block

        # 误差频域表示。前半部分补零，只用有效误差更新。
        e_fft_buf = np.concatenate(
            [np.zeros(self.block_size, dtype=np.float64), e_block],
            axis=0,
        )
        E = np.fft.fft(e_fft_buf, n=self.fft_size)

        # 频域 NLMS 更新。
        power = np.abs(X) ** 2 + self.eps

        if self.leakage > 0.0:
            self.W *= (1.0 - self.leakage)

        self.W += self.step_size * np.conj(X) * E / power

        # 时域约束，保证有效滤波器长度不超过 filter_length。
        self._constrain_time_domain_filter()

        # 更新 overlap-save 历史输入。
        self.x_hist = x_block.copy()

        return y_block, e_block