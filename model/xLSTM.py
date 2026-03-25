import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalConv1D(nn.Module):
    """
    Causal 1D convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CausalConv1D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding)
    
    def forward(self, x):
        # x: [batch_size, channels, seq_len] or [batch_size, seq_len]
        if x.dim() == 2:
            # 2D input: [batch_size, seq_len] -> [batch_size, 1, seq_len]
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        out = self.conv(x)
        # 移除右侧的padding，保持因果性
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        
        if squeeze_output:
            out = out.squeeze(1)  # [batch_size, seq_len]
        
        return out


class BlockDiagonal(nn.Module):
    """
    Block diagonal linear layer
    """
    def __init__(self, in_features, out_features, num_blocks):
        super(BlockDiagonal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        self.block_in_features = in_features // num_blocks
        self.block_out_features = out_features // num_blocks
        
        # 创建块对角权重
        self.weights = nn.Parameter(torch.Tensor(num_blocks, self.block_in_features, self.block_out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # 初始化参数
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # x: [batch_size, in_features] or [batch_size, seq_len, in_features]
        if x.dim() == 2:
            # 2D input: [batch_size, in_features]
            batch_size = x.shape[0]
            
            # 重塑为块格式 [batch_size, num_blocks, block_in_features]
            x_reshaped = x.view(batch_size, self.num_blocks, self.block_in_features)
            
            # 应用块对角线性变换
            # 使用 einsum: [batch_size, num_blocks, block_in_features] @ [num_blocks, block_in_features, block_out_features]
            # -> [batch_size, num_blocks, block_out_features]
            out = torch.einsum('bni,nio->bno', x_reshaped, self.weights)
            
            # 重塑回原始格式
            out = out.reshape(batch_size, self.out_features)
            out = out + self.bias
            
        else:
            # 3D input: [batch_size, seq_len, in_features]
            batch_size, seq_len, _ = x.shape
            
            # 重塑为块格式
            x_reshaped = x.view(batch_size, seq_len, self.num_blocks, self.block_in_features)
            
            # 应用块对角线性变换
            # 使用 einsum: [batch_size, seq_len, num_blocks, block_in_features] @ [num_blocks, block_in_features, block_out_features]
            # -> [batch_size, seq_len, num_blocks, block_out_features]
            out = torch.einsum('bsni,nio->bsno', x_reshaped, self.weights)
            
            # 重塑回原始格式
            out = out.reshape(batch_size, seq_len, self.out_features)
            out = out + self.bias
        
        return out


class sLSTMBlock(nn.Module):
    """
    Scalar LSTM Block with exponential gating and residual connections
    参考论文实现，包含layer norm, causal conv, block diagonal layers等
    """
    def __init__(self, input_size, hidden_size, num_heads=4, proj_factor=4/3):
        super(sLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.proj_factor = proj_factor
        
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        assert proj_factor > 0
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_size)
        
        # Causal convolution
        self.causal_conv = CausalConv1D(1, 1, 4)
        
        # Block diagonal layers for gates
        self.Wz = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wi = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wf = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wo = BlockDiagonal(input_size, hidden_size, num_heads)
        
        self.Rz = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ri = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Rf = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ro = BlockDiagonal(hidden_size, hidden_size, num_heads)
        
        # Group normalization
        self.group_norm = nn.GroupNorm(num_heads, hidden_size)
        
        # Up and down projections
        self.up_proj_left = nn.Linear(hidden_size, int(hidden_size * proj_factor))
        self.up_proj_right = nn.Linear(hidden_size, int(hidden_size * proj_factor))
        self.down_proj = nn.Linear(int(hidden_size * proj_factor), input_size)
    
    def forward(self, x, prev_state):
        """
        Args:
            x: [batch_size, input_size]
            prev_state: (h, c, n, m) where
                h: [batch_size, hidden_size]
                c: [batch_size, hidden_size]
                n: [batch_size, hidden_size] - normalizer
                m: [batch_size, hidden_size] - stabilizer
        """
        assert x.size(-1) == self.input_size
        h_prev, c_prev, n_prev, m_prev = prev_state
        
        # Layer normalization
        x_norm = self.layer_norm(x)
        
        # Causal convolution with SiLU activation
        # CausalConv1D 现在可以处理 2D 输入
        x_conv = F.silu(self.causal_conv(x_norm))
        
        # Compute gates
        z = torch.tanh(self.Wz(x) + self.Rz(h_prev))
        o = torch.sigmoid(self.Wo(x) + self.Ro(h_prev))
        
        i_tilde = self.Wi(x_conv) + self.Ri(h_prev)
        f_tilde = self.Wf(x_conv) + self.Rf(h_prev)
        
        # Stabilizer update
        m_t = torch.max(f_tilde + m_prev, i_tilde)
        
        # Exponential gating
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)
        
        # Cell and normalizer update
        c_t = f * c_prev + i * z
        n_t = f * n_prev + i
        
        # Hidden state
        h_t = o * c_t / (n_t + 1e-6)
        
        # Output processing
        output = h_t
        output_norm = self.group_norm(output)
        
        # Gated projection
        output_left = self.up_proj_left(output_norm)
        output_right = self.up_proj_right(output_norm)
        output_gated = F.gelu(output_right)
        output = output_left * output_gated
        
        # Down projection
        output = self.down_proj(output)
        
        # Residual connection
        final_output = output + x
        
        return final_output, (h_t, c_t, n_t, m_t)


class sLSTMCell(nn.Module):
    """
    Simplified Scalar LSTM Cell for compatibility
    返回隐藏状态而不是 Block 的残差输出
    """
    def __init__(self, input_size, hidden_size, num_heads=4):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.block = sLSTMBlock(input_size, hidden_size, num_heads)
    
    def forward(self, x, states):
        # Block 返回 (output, states)，其中 output 是残差输出 [batch_size, input_size]
        # states 是 (h_t, c_t, n_t, m_t)，其中 h_t 是隐藏状态 [batch_size, hidden_size]
        _, new_states = self.block(x, states)
        h_t = new_states[0]  # 提取隐藏状态
        return h_t, new_states


class mLSTMBlock(nn.Module):
    """
    Matrix LSTM Block with covariance update rule
    参考论文实现，包含layer norm, causal conv, block diagonal layers等
    """
    def __init__(self, input_size, hidden_size, num_heads=4, proj_factor=2):
        super(mLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.proj_factor = proj_factor
        
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        assert proj_factor > 0
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_size)
        
        # Up projections
        self.up_proj_left = nn.Linear(input_size, int(input_size * proj_factor))
        self.up_proj_right = nn.Linear(input_size, hidden_size)
        
        # Down projection
        self.down_proj = nn.Linear(hidden_size, input_size)
        
        # Causal convolution
        self.causal_conv = CausalConv1D(1, 1, 4)
        
        # Skip connection
        self.skip_connection = nn.Linear(int(input_size * proj_factor), hidden_size)
        
        # Block diagonal layers for Q, K, V
        self.Wq = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        self.Wk = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        self.Wv = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        
        # Gates
        self.Wi = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wf = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wo = nn.Linear(int(input_size * proj_factor), hidden_size)
        
        # Group normalization
        self.group_norm = nn.GroupNorm(num_heads, hidden_size)
    
    def forward(self, x, prev_state):
        """
        Args:
            x: [batch_size, input_size]
            prev_state: (h, c, n, m) where
                h: [batch_size, hidden_size]
                c: [batch_size, hidden_size, hidden_size] - matrix memory
                n: [batch_size, hidden_size] - normalizer
                m: [batch_size, hidden_size] - stabilizer
        """
        h_prev, c_prev, n_prev, m_prev = prev_state
        assert x.size(-1) == self.input_size
        
        # Layer normalization
        x_norm = self.layer_norm(x)
        
        # Up projections
        x_up_left = self.up_proj_left(x_norm)
        x_up_right = self.up_proj_right(x_norm)
        
        # Causal convolution with SiLU activation
        # CausalConv1D 现在可以处理 2D 输入
        x_conv = F.silu(self.causal_conv(x_up_left))
        
        # Skip connection
        x_skip = self.skip_connection(x_conv)
        
        # Compute Q, K, V
        q = self.Wq(x_conv)
        k = self.Wk(x_conv) / (self.head_size ** 0.5)
        v = self.Wv(x_up_left)
        
        # Compute gates
        i_tilde = self.Wi(x_conv)
        f_tilde = self.Wf(x_conv)
        o = torch.sigmoid(self.Wo(x_up_left))
        
        # Stabilizer update
        m_t = torch.max(f_tilde + m_prev, i_tilde)
        
        # Exponential gating
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)
        
        # Matrix memory update: c_t = f * c_prev + i * (v ⊗ k^T)
        # 使用外积更新矩阵记忆
        c_t = f.unsqueeze(2) * c_prev + i.unsqueeze(2) * (v.unsqueeze(2) @ k.unsqueeze(1))
        
        # Normalizer update
        n_t = f * n_prev + i * k
        
        # Retrieve from memory: h_tilde = (c @ q) / max{|n.T @ q|, 1}
        numerator = torch.bmm(c_t, q.unsqueeze(2)).squeeze(2)  # [batch_size, hidden_size]
        denominator = torch.max(torch.abs((n_t * q).sum(dim=1, keepdim=True)), torch.ones_like((n_t * q).sum(dim=1, keepdim=True)))
        h_tilde = numerator / (denominator + 1e-6)
        
        # Output processing
        output = h_tilde
        output_norm = self.group_norm(output)
        output = output_norm + x_skip
        output = output * F.silu(x_up_right)
        
        # Down projection
        output = self.down_proj(output)
        
        # Residual connection
        final_output = output + x
        
        return final_output, (h_tilde, c_t, n_t, m_t)


class mLSTMCell(nn.Module):
    """
    Simplified Matrix LSTM Cell for compatibility
    返回隐藏状态而不是 Block 的残差输出
    """
    def __init__(self, input_size, hidden_size, num_heads=4):
        super(mLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.block = mLSTMBlock(input_size, hidden_size, num_heads)
    
    def forward(self, x, states):
        # Block 返回 (output, states)，其中 output 是残差输出 [batch_size, input_size]
        # states 是 (h_tilde, c_t, n_t, m_t)，其中 h_tilde 是隐藏状态 [batch_size, hidden_size]
        _, new_states = self.block(x, states)
        h_t = new_states[0]  # 提取隐藏状态
        return h_t, new_states


class xLSTMLayer(nn.Module):
    """
    xLSTM Layer that can use either sLSTM or mLSTM
    """
    def __init__(self, input_size, hidden_size, cell_type='slstm', bidirectional=False, num_heads=4, mlstm_hidden_size=None):
        super(xLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.num_heads = num_heads
        
        # mLSTM 使用更小的 hidden_size 以节省内存
        if mlstm_hidden_size is None:
            # 默认使用 hidden_size 的 1/4，但至少为 128
            mlstm_hidden_size = max(128, hidden_size // 4)
        self.mlstm_hidden_size = mlstm_hidden_size
        
        # 确保 hidden_size 能被 num_heads 整除
        while hidden_size % num_heads != 0:
            num_heads -= 1
        self.num_heads = num_heads
        
        # 确保 mlstm_hidden_size 能被 num_heads 整除
        mlstm_num_heads = num_heads
        while mlstm_hidden_size % mlstm_num_heads != 0:
            mlstm_num_heads -= 1
        self.mlstm_num_heads = mlstm_num_heads
        
        if cell_type == 'slstm':
            self.cell_forward = sLSTMCell(input_size, hidden_size, num_heads)
            if bidirectional:
                self.cell_backward = sLSTMCell(input_size, hidden_size, num_heads)
        elif cell_type == 'mlstm':
            self.cell_forward = mLSTMCell(input_size, mlstm_hidden_size, mlstm_num_heads)
            if bidirectional:
                self.cell_backward = mLSTMCell(input_size, mlstm_hidden_size, mlstm_num_heads)
            # 添加线性层将 mlstm_hidden_size 转换为 hidden_size
            num_directions = 2 if bidirectional else 1
            self.mlstm_proj = nn.Linear(mlstm_hidden_size * num_directions, hidden_size * num_directions)
        elif cell_type == 'all':
            # 同时使用slstm和mlstm
            self.cell_forward_slstm = sLSTMCell(input_size, hidden_size, num_heads)
            self.cell_forward_mlstm = mLSTMCell(hidden_size, mlstm_hidden_size, mlstm_num_heads)
            if bidirectional:
                self.cell_backward_slstm = sLSTMCell(input_size, hidden_size, num_heads)
                self.cell_backward_mlstm = mLSTMCell(hidden_size, mlstm_hidden_size, mlstm_num_heads)
            # 添加线性层将 mlstm_hidden_size 转换为 hidden_size
            self.mlstm_proj_forward = nn.Linear(mlstm_hidden_size, hidden_size)
            if bidirectional:
                self.mlstm_proj_backward = nn.Linear(mlstm_hidden_size, hidden_size)
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}")
    
    def _init_states(self, batch_size, device):
        """Initialize hidden states"""
        if self.cell_type == 'slstm':
            h = torch.zeros(batch_size, self.hidden_size, device=device)
            c = torch.zeros(batch_size, self.hidden_size, device=device)
            n = torch.zeros(batch_size, self.hidden_size, device=device)
            m = torch.zeros(batch_size, self.hidden_size, device=device)
            return (h, c, n, m)
        elif self.cell_type == 'mlstm':
            # mLSTM states with matrix memory (使用更小的 mlstm_hidden_size)
            h = torch.zeros(batch_size, self.mlstm_hidden_size, device=device)
            C = torch.zeros(batch_size, self.mlstm_hidden_size, self.mlstm_hidden_size, device=device)
            n = torch.zeros(batch_size, self.mlstm_hidden_size, device=device)
            m = torch.zeros(batch_size, self.mlstm_hidden_size, device=device)
            return (h, C, n, m)
        elif self.cell_type == 'all':
            # 同时初始化slstm和mlstm的状态
            # slstm states
            h_slstm = torch.zeros(batch_size, self.hidden_size, device=device)
            c_slstm = torch.zeros(batch_size, self.hidden_size, device=device)
            n_slstm = torch.zeros(batch_size, self.hidden_size, device=device)
            m_slstm = torch.zeros(batch_size, self.hidden_size, device=device)
            slstm_states = (h_slstm, c_slstm, n_slstm, m_slstm)
            
            # mlstm states with matrix memory (使用更小的 mlstm_hidden_size)
            h_mlstm = torch.zeros(batch_size, self.mlstm_hidden_size, device=device)
            C_mlstm = torch.zeros(batch_size, self.mlstm_hidden_size, self.mlstm_hidden_size, device=device)
            n_mlstm = torch.zeros(batch_size, self.mlstm_hidden_size, device=device)
            m_mlstm = torch.zeros(batch_size, self.mlstm_hidden_size, device=device)
            mlstm_states = (h_mlstm, C_mlstm, n_mlstm, m_mlstm)
            
            return (slstm_states, mlstm_states)
        else:
            raise ValueError(f"Unknown cell_type: {self.cell_type}")
    
    def forward(self, x):
        """
        Args:
            x: [seq_len, batch_size, input_size] or [batch_size, seq_len, input_size]
        Returns:
            output: [seq_len, batch_size, hidden_size * num_directions]
        """
        # Handle both input formats
        if x.dim() == 3 and x.size(0) != x.size(1):
            # Assume [batch_size, seq_len, input_size], convert to [seq_len, batch_size, input_size]
            if x.size(1) > x.size(0):
                x = x.transpose(0, 1)
        
        seq_len, batch_size, _ = x.size()
        device = x.device
        
        # Forward direction
        if self.cell_type == 'all':
            # 同时使用slstm和mlstm
            slstm_states_forward, mlstm_states_forward = self._init_states(batch_size, device)
            outputs_forward_slstm = []
            outputs_forward_mlstm = []
            
            for t in range(seq_len):
                # 先通过slstm处理
                h_t_slstm, slstm_states_forward = self.cell_forward_slstm(x[t], slstm_states_forward)
                # 再通过mlstm处理slstm的输出
                h_t_mlstm, mlstm_states_forward = self.cell_forward_mlstm(h_t_slstm, mlstm_states_forward)
                # 将 mlstm 输出转换为 hidden_size
                h_t_mlstm = self.mlstm_proj_forward(h_t_mlstm)
                # 保存两个输出
                outputs_forward_slstm.append(h_t_slstm)
                outputs_forward_mlstm.append(h_t_mlstm)
            
            # 拼接slstm和mlstm的输出
            outputs_forward_slstm = torch.stack(outputs_forward_slstm, dim=0)  # [seq_len, batch_size, hidden_size]
            outputs_forward_mlstm = torch.stack(outputs_forward_mlstm, dim=0)  # [seq_len, batch_size, hidden_size]
            outputs_forward = torch.cat([outputs_forward_slstm, outputs_forward_mlstm], dim=2)  # [seq_len, batch_size, hidden_size*2]
            
            if self.bidirectional:
                # Backward direction
                slstm_states_backward, mlstm_states_backward = self._init_states(batch_size, device)
                outputs_backward_slstm = []
                outputs_backward_mlstm = []
                
                for t in range(seq_len - 1, -1, -1):
                    # 先通过slstm处理
                    h_t_slstm, slstm_states_backward = self.cell_backward_slstm(x[t], slstm_states_backward)
                    # 再通过mlstm处理slstm的输出
                    h_t_mlstm, mlstm_states_backward = self.cell_backward_mlstm(h_t_slstm, mlstm_states_backward)
                    # 将 mlstm 输出转换为 hidden_size
                    h_t_mlstm = self.mlstm_proj_backward(h_t_mlstm)
                    # 保存两个输出
                    outputs_backward_slstm.append(h_t_slstm)
                    outputs_backward_mlstm.append(h_t_mlstm)
                
                # 拼接slstm和mlstm的输出
                outputs_backward_slstm = torch.stack(outputs_backward_slstm[::-1], dim=0)  # [seq_len, batch_size, hidden_size]
                outputs_backward_mlstm = torch.stack(outputs_backward_mlstm[::-1], dim=0)  # [seq_len, batch_size, hidden_size]
                outputs_backward = torch.cat([outputs_backward_slstm, outputs_backward_mlstm], dim=2)  # [seq_len, batch_size, hidden_size*2]
                
                # Concatenate forward and backward
                output = torch.cat([outputs_forward, outputs_backward], dim=2)  # [seq_len, batch_size, hidden_size*4]
            else:
                output = outputs_forward
        else:
            # 原有逻辑 (slstm 或 mlstm)
            states_forward = self._init_states(batch_size, device)
            outputs_forward = []
            
            for t in range(seq_len):
                h_t, states_forward = self.cell_forward(x[t], states_forward)
                outputs_forward.append(h_t)
            
            outputs_forward = torch.stack(outputs_forward, dim=0)  # [seq_len, batch_size, mlstm_hidden_size or hidden_size]
            
            if self.bidirectional:
                # Backward direction
                states_backward = self._init_states(batch_size, device)
                outputs_backward = []
                
                for t in range(seq_len - 1, -1, -1):
                    h_t, states_backward = self.cell_backward(x[t], states_backward)
                    outputs_backward.append(h_t)
                
                outputs_backward = torch.stack(outputs_backward[::-1], dim=0)  # [seq_len, batch_size, mlstm_hidden_size or hidden_size]
                
                # Concatenate forward and backward
                output = torch.cat([outputs_forward, outputs_backward], dim=2)  # [seq_len, batch_size, (mlstm_hidden_size or hidden_size)*2]
            else:
                output = outputs_forward
            
            # 如果是 mLSTM，需要通过线性层转换维度
            if self.cell_type == 'mlstm':
                output = self.mlstm_proj(output)  # [seq_len, batch_size, hidden_size * num_directions]
        
        return output, None  # Return None for compatibility with RNN interface


class MultiLevelxLSTM(nn.Module):
    """
    Multi-level xLSTM for word, phrase, and sentence level processing
    """
    def __init__(self, input_size, hidden_size, cell_type='slstm', bidirectional=True, 
                 dropout=0.3, layer_norm=True, num_heads=4, mlstm_hidden_size=None):
        super(MultiLevelxLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_heads = num_heads
        
        # mLSTM 使用更小的 hidden_size 以节省内存
        if mlstm_hidden_size is None:
            # 默认使用 hidden_size 的 1/4，但至少为 128
            mlstm_hidden_size = max(128, hidden_size // 4)
        self.mlstm_hidden_size = mlstm_hidden_size
        
        # 确保 hidden_size 能被 num_heads 整除
        while hidden_size % num_heads != 0:
            num_heads -= 1
        self.num_heads = num_heads
        
        # Three separate xLSTM layers for different levels
        self.word_xlstm = xLSTMLayer(input_size, hidden_size, cell_type, bidirectional, num_heads, mlstm_hidden_size)
        self.phrase_xlstm = xLSTMLayer(input_size, hidden_size, cell_type, bidirectional, num_heads, mlstm_hidden_size)
        self.sentence_xlstm = xLSTMLayer(input_size, hidden_size, cell_type, bidirectional, num_heads, mlstm_hidden_size)
        
        # 计算输出维度
        # 'all' 模式: (slstm + mlstm) * num_directions = hidden_size * 2 * num_directions
        # 其他模式: hidden_size * num_directions
        if cell_type == 'all':
            output_dim = hidden_size * 2 * self.num_directions
        else:
            output_dim = hidden_size * self.num_directions
        
        # Layer normalization
        self.layer_norm = layer_norm
        if layer_norm:
            self.word_norm = nn.LayerNorm(output_dim)
            self.phrase_norm = nn.LayerNorm(output_dim)
            self.sentence_norm = nn.LayerNorm(output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fusion layer to combine three levels
        self.fusion = nn.Linear(output_dim * 3, output_dim)
        self.fusion_norm = nn.LayerNorm(output_dim)
    
    def forward(self, word_level, phrase_level, sentence_level):
        """
        Args:
            word_level: [seq_len, batch_size, input_size] or [batch_size, seq_len, input_size]
            phrase_level: [seq_len, batch_size, input_size] or [batch_size, seq_len, input_size]
            sentence_level: [seq_len, batch_size, input_size] or [batch_size, seq_len, input_size]
        Returns:
            output: [seq_len, batch_size, hidden_size * num_directions]
        """
        # Process each level with xLSTM
        word_out, _ = self.word_xlstm(word_level)
        phrase_out, _ = self.phrase_xlstm(phrase_level)
        sentence_out, _ = self.sentence_xlstm(sentence_level)
        
        # Apply layer normalization
        if self.layer_norm:
            word_out = self.word_norm(word_out)
            phrase_out = self.phrase_norm(phrase_out)
            sentence_out = self.sentence_norm(sentence_out)
        
        # Apply dropout
        word_out = self.dropout(word_out)
        phrase_out = self.dropout(phrase_out)
        sentence_out = self.dropout(sentence_out)
        
        # Concatenate all levels
        combined = torch.cat([word_out, phrase_out, sentence_out], dim=-1)
        
        # Fusion
        output = self.fusion(combined)
        output = self.fusion_norm(output)
        
        return word_out, phrase_out, sentence_out  # Return None for compatibility
