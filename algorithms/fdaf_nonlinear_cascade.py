"""
FDAF + 非线性残差分支级联模型。

结构：
    x(n)
      ↓
    FDAF 长线性滤波器
      ↓
    线性回声估计 y_lin(n)
      ↓
    线性残差 e_lin(n) = d(n) - y_lin(n)
      ↓
    短非线性滤波器 WL-LMS / GH-WL-LMS-Fast
      ↓
    非线性残差估计 y_nl(n)
      ↓
    最终回声估计 y_hat(n) = y_lin(n) + y_nl(n)
    最终残差 e_final(n) = e_lin(n) - y_nl(n)

说明：
    FDAF 负责主要线性 echo path；
    非线性分支只学习 FDAF 未消除的非线性残差。
"""

import numpy as np


class FDAFNonlinearCascade:
    """
    FDAF + 可选非线性残差分支。

    nonlinear_filter 为 None 时，模型退化为 FDAF only。
    """

    def __init__(
        self,
        fdaf,
        nonlinear_filter=None,
        nonlinear_order: int = 16,
    ):
        self.fdaf = fdaf
        self.nonlinear_filter = nonlinear_filter
        self.nonlinear_order = int(nonlinear_order)

        if self.nonlinear_order <= 0:
            raise ValueError("nonlinear_order 必须为正数。")

        self.reset_nonlinear_buffer()

    def reset_nonlinear_buffer(self):
        """重置非线性分支的短记忆输入缓存。"""
        self.x_nl_buf = np.zeros(self.nonlinear_order, dtype=np.float64)

    def reset(self):
        """重置级联模型状态。"""
        if hasattr(self.fdaf, "reset"):
            self.fdaf.reset()

        if self.nonlinear_filter is not None and hasattr(self.nonlinear_filter, "reset"):
            try:
                self.nonlinear_filter.reset()
            except TypeError:
                self.nonlinear_filter.reset(seed=getattr(self.nonlinear_filter, "seed", 0))

        self.reset_nonlinear_buffer()

    def _predict_nonlinear(self, x_vec):
        """
        非线性分支预测。

        如果算法有 predict 接口，则直接使用；
        否则返回 None，后续通过 update 返回误差反推输出。
        """
        if self.nonlinear_filter is None:
            return 0.0

        if hasattr(self.nonlinear_filter, "predict"):
            return float(self.nonlinear_filter.predict(x_vec))

        return None

    def _update_nonlinear(self, x_vec, target):
        """
        非线性分支更新。

        target 是 FDAF 线性残差 e_lin(k)。
        """
        if self.nonlinear_filter is None:
            return 0.0

        y_pred = self._predict_nonlinear(x_vec)

        if y_pred is None:
            # 若无 predict 接口，则假设 update 返回 e = target - y。
            e_nl = float(self.nonlinear_filter.update(x_vec, float(target)))
            y_pred = float(target) - e_nl
        else:
            self.nonlinear_filter.update(x_vec, float(target))

        return float(y_pred)

    def process_block(self, x_block, d_block):
        """
        处理一个 block。

        返回字典，便于实验脚本分别统计线性残差、非线性补偿和最终残差。
        """
        x_block = np.asarray(x_block, dtype=np.float64)
        d_block = np.asarray(d_block, dtype=np.float64)

        y_lin_block, e_lin_block = self.fdaf.process_block(x_block, d_block)

        y_nl_block = np.zeros_like(e_lin_block)

        if self.nonlinear_filter is not None:
            for k, xk in enumerate(x_block):
                # 更新短记忆输入向量：
                # [x(k), x(k-1), ..., x(k-p+1)]
                self.x_nl_buf[1:] = self.x_nl_buf[:-1]
                self.x_nl_buf[0] = float(xk)

                # 非线性分支拟合 FDAF 线性残差。
                y_nl_block[k] = self._update_nonlinear(
                    self.x_nl_buf,
                    target=e_lin_block[k],
                )

        y_hat_block = y_lin_block + y_nl_block
        e_final_block = d_block - y_hat_block

        return dict(
            y_hat=y_hat_block,
            e_final=e_final_block,
            y_lin=y_lin_block,
            e_lin=e_lin_block,
            y_nl=y_nl_block,
        )