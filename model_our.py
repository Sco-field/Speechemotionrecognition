import torch
import torch.nn as nn
from transformers import AutoConfig, Wav2Vec2Model
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureEncoder, Wav2Vec2FeatureProjection

wav2vec2model = 'wav2vec2-large-xlsr-53-chinese-zh-cn/wav2vec2-large-xlsr-53-chinese-zh-cn'

# 创建一个测试用的配置对象
config = AutoConfig.from_pretrained(wav2vec2model)

# 手动添加自定义参数
config.pooling_strategy = "mean"
config.feature_fusion = 2048  # 更新为合适的特征融合大小
config.hidden_size = 1024
config.dense_hidden_size=512
# Attention 层
class AttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, tar):
        src = src.transpose(0, 1)
        tar = tar.transpose(0, 1)
        attn_output, _ = self.self_attn(tar, src, src)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        src = src.transpose(0, 1)
        return src

# 定义 AudioModel 类
class AudioModel(nn.Module):
    def __init__(self, config):
        super(AudioModel, self).__init__()

        # Wav2Vec2 和 Pitch 编码器
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2model, config=config)
        self.pooling_strategy = config.pooling_strategy
        self.pitch_encoder = Wav2Vec2FeatureEncoder(config)
        self.pitch_projection = Wav2Vec2FeatureProjection(config)
        self.lstm = nn.LSTM(config.mfccs_input_dim,
                            config.bilstm_hidden_dim,
                            num_layers=config.bilstm_num_layers,
                            batch_first=True,
                            bidirectional=True)

        # 交叉注意力层
        self.attn_a_p = AttentionLayer(config.hidden_size, 16)
        self.attn_p_a = AttentionLayer(config.hidden_size, 16)
        
        # 自注意力层
        self.self_attn = AttentionLayer(config.feature_fusion, 8)

        # Dropout、全连接层、Tanh 和分类层
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc = nn.Linear(config.feature_fusion, config.dense_hidden_size)
        self.tanh = nn.Tanh()
        self.dropout2 = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(config.dense_hidden_size, 1)

    def _apply_pooling(self, x, strategy="mean"):
        if strategy == "mean":
            return x.mean(dim=1)
        elif strategy == "max":
            return x.max(dim=1)[0]
        elif strategy == "min":
            return x.min(dim=1)[0]
        else:
            raise ValueError(f"未知的池化策略: {strategy}")

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self, audio_input, pitch_input,mfcc_input):
        # Wav2Vec2 输出
        audio_output = self.wav2vec2(audio_input).last_hidden_state
        lstm_out, _ = self.lstm(mfcc_input)
        pooled_lstm_out = self._apply_pooling(lstm_out, strategy=self.pooling_strategy)
        # Pitch 编码器输出
        pitch_input = pitch_input.squeeze(1)
        pitch = self.pitch_encoder(pitch_input)
        pitch = pitch.transpose(1, 2)
        pitch, _ = self.pitch_projection(pitch)
        # 交叉注意力
        a_p_attn = self.attn_a_p(audio_output, pitch)
        p_a_attn = self.attn_p_a(pitch, audio_output)
        # 拼接交叉注意力输出
        attention_output = torch.cat((a_p_attn, p_a_attn), -1)
        pooled_attention_output = self._apply_pooling(attention_output, strategy=self.pooling_strategy)
        combined_output = torch.cat((pooled_attention_output, pooled_lstm_out), dim=-1)
        combined_output = combined_output.unsqueeze(1)  # 增加一个维度以应用自注意力
        attn_final_output = self.self_attn(combined_output, combined_output)
        attn_final_output = attn_final_output.squeeze(1)
        # 添加自注意力
        # Dropout、全连接层、Tanh、第二个 Dropout 和分类器
        x = self.dropout1(attn_final_output)
        x = self.fc(x)
        x = self.tanh(x)
        x = self.dropout2(x)
        logits = self.classifier(x)

        return logits
