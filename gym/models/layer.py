import numpy as np
import torch


def get_position_angle_vec(position, d_hid):
    return [
        position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)
    ]


def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""
    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i, d_hid) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def get_position_angle_vec_torch(position, d_hid):
    hid_j = torch.arange(d_hid, dtype=torch.float32).to(position.device)
    return position.repeat(d_hid, 1).permute(1, 0) / torch.pow(
        torch.tensor(10000).to(position.device), 2 * (hid_j // 2) / d_hid
    )


def get_sinusoid_encoding(position, d_hid):
    pe = get_position_angle_vec_torch(position, d_hid)
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe
