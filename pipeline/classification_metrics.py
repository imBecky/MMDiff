"""分类指标：OA / AA / Kappa 等。独立模块名，避免与第三方 `metrics` 包或目录冲突。"""
import numpy as np


def accuracies(cm):
    """
    返回 ovr_acc, usr_acc, prod_acc, kappa, s_sqr, aa。

    AA（Average Accuracy）：各类别召回率（对角/行和）在真值有样本的类上取平均。
    """
    num_class = np.shape(cm)[0]
    n = np.sum(cm)

    P = cm / n
    ovr_acc = np.trace(P)

    p_plus_j = np.sum(P, axis=0)
    p_i_plus = np.sum(P, axis=1)

    usr_acc = np.divide(np.diagonal(P), p_i_plus, out=np.zeros_like(p_i_plus), where=p_i_plus != 0)
    prod_acc = np.divide(np.diagonal(P), p_plus_j, out=np.zeros_like(p_plus_j), where=p_plus_j != 0)
    mask = p_i_plus > 0
    aa = float(np.mean(usr_acc[mask])) if np.any(mask) else 0.0

    theta1 = np.trace(P)
    theta2 = np.sum(p_plus_j * p_i_plus)
    theta3 = np.sum(np.diagonal(P) * (p_plus_j + p_i_plus))
    theta4 = 0
    for i in range(num_class):
        for j in range(num_class):
            theta4 = theta4 + P[i, j] * (p_plus_j[i] + p_i_plus[j]) ** 2

    kappa = (theta1 - theta2) / (1 - theta2)

    t1 = theta1 * (1 - theta1) / (1 - theta2) ** 2
    t2 = 2 * (1 - theta1) * (2 * theta1 * theta2 - theta3) / (1 - theta2) ** 3
    t3 = ((1 - theta1) ** 2) * (theta4 - 4 * theta2 ** 2) / (1 - theta2) ** 4

    s_sqr = (t1 + t2 + t3) / n

    return ovr_acc, usr_acc, prod_acc, kappa, s_sqr, aa


__all__ = ['accuracies']
