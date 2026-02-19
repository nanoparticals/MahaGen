# data.py
# 支持两种输入：
#   mode="smiles": 字符级分词（逐字符）
#   mode="selfies": token级分词（逐 [ ... ] 作为一个 token）
#
# 暴露接口：
#   - self.all_characters: 词表（list[str]，元素是字符或SELFIES token）
#   - self.n_characters:   词表大小
#   - char_tensor(tok):    把单个“符号”（字符或token）转为 LongTensor 索引
#   - random_training_set(...):
#       返回 (inp, tgt) 或 (inp, tgt, idx)（当 return_index=True 时）
#       其中 idx 是样本在原 CSV 中的行号（去除空行/越界后的索引）
#
# 注意：开始/结束符固定为 "<" 和 ">"，作为词表中的两个特殊符号。

from typing import List, Tuple
import csv
import random
import torch

START_TOKEN = "<"
END_TOKEN = ">"


def _read_csv_column(path: str, col_idx: int, delimiter: str = ",", keep_header: bool = False) -> List[str]:
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        first = True
        for row in reader:
            if first and keep_header:
                first = False
                continue
            if not row or col_idx >= len(row):
                continue
            s = (row[col_idx] or "").strip()
            if s:
                seqs.append(s)
    return seqs


def _split_selfies_tokens(s: str) -> List[str]:
    """把 '[C][O][Ring1]' 解析为 ['[C]', '[O]', '[Ring1]']；保留可能出现的 '<' '>' """
    tokens = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "[":
            j = s.find("]", i + 1)
            if j == -1:
                # 非法字符串，尽量跳过
                break
            tokens.append(s[i:j+1])
            i = j + 1
        elif c in (START_TOKEN, END_TOKEN):
            tokens.append(c)
            i += 1
        elif c.isspace():
            i += 1
        else:
            # SELFIES 理论上不会有裸字符，这里忽略
            i += 1
    return tokens


class GeneratorData:
    def __init__(
        self,
        path: str,
        max_len: int = 120,
        cols_to_read: List[int] = [0],
        delimiter: str = ",",
        keep_header: bool = False,
        mode: str = "smiles",  # 'smiles' 字符级；'selfies' token级
    ):
        assert len(cols_to_read) == 1, "只支持读取单列序列数据"
        self.mode = mode.lower()
        self.max_len = int(max_len)

        # 读取原始序列
        raw = _read_csv_column(
            path=path, col_idx=int(cols_to_read[0]),
            delimiter=delimiter, keep_header=keep_header
        )

        # 解析为符号序列（字符或token）
        seq_tokens_list: List[List[str]] = []
        for s in raw:
            if self.mode == "selfies":
                toks = _split_selfies_tokens(s)
            else:  # smiles 字符级
                toks = list(s)

            # 去掉内部可能混入的起止符，统一我们自己加
            toks = [t for t in toks if t not in (START_TOKEN, END_TOKEN)]

            # 截断到 max_len（注意要给起止符留位置，所以这里最多保留 max_len-2）
            if len(toks) > max_len - 2:
                toks = toks[: max_len - 2]

            # 包装起止符
            toks = [START_TOKEN] + toks + [END_TOKEN]
            seq_tokens_list.append(toks)

        # 保存处理后的序列（带 < >）
        self.sequences: List[List[str]] = seq_tokens_list

        # 构建词表
        vocab = set([START_TOKEN, END_TOKEN])
        for toks in self.sequences:
            vocab.update(toks)

        # 固定词表顺序：保证可重复性
        self.all_characters = sorted(vocab)
        self.n_characters = len(self.all_characters)
        self.char2index = {ch: i for i, ch in enumerate(self.all_characters)}

        # 简要信息
        avg_len = (sum(len(x) for x in self.sequences) / max(1, len(self.sequences))) if self.sequences else 0.0
        print(f"[data] mode={self.mode}  samples={len(self.sequences)}  "
              f"vocab={self.n_characters}  avg_len={avg_len:.1f}")

    # ========== 编码接口 ==========
    def char_tensor(self, ch: str) -> torch.LongTensor:
        """把单个 符号(字符或token) 转成索引张量（形状 [1]）。"""
        idx = self.char2index[ch]
        return torch.tensor([idx], dtype=torch.long)

    def _seq_to_tensor(self, seq_tokens: List[str]) -> torch.LongTensor:
        """把 整个符号序列 转为 1D LongTensor。"""
        return torch.tensor([self.char2index[t] for t in seq_tokens], dtype=torch.long)

    # ========== 新增：按索引取样本（便于对齐 z_bank[idx]） ==========
    def tensors_by_index(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """给定样本索引，返回 (inp, tgt)。"""
        n = len(self.sequences)
        if n == 0:
            raise IndexError("Empty dataset")
        if idx < 0:
            idx = 0
        if idx >= n:
            idx = n - 1
        seq = self.sequences[idx]
        inp_tokens = seq[:-1]
        tgt_tokens = seq[1:]
        inp = self._seq_to_tensor(inp_tokens)
        tgt = self._seq_to_tensor(tgt_tokens)
        return inp, tgt

    # ========== 采样训练对（支持返回索引） ==========
    def random_training_set(self, smiles_augmentation=None, return_index: bool = False):
        """返回 (inp, tgt) 或 (inp, tgt, idx)。
        inp 是去掉最后一个符号（>），tgt 是去掉第一个符号（<）。
        当 return_index=True 时，同时返回本样本在内部列表中的索引 idx，
        供上层与 z_bank[idx] 对齐使用。
        """
        n = len(self.sequences)
        if n == 0:
            raise RuntimeError("Dataset is empty; please check CSV path/column.")
        idx = random.randrange(n)
        inp, tgt = self.tensors_by_index(idx)
        if return_index:
            return inp, tgt, idx
        return inp, tgt
