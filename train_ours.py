import librosa
import pandas as pd
import torch
import numpy as np
import random
import torchaudio
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Processor, AutoConfig, Wav2Vec2Config
from model_our import AudioModel
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="It is strongly recommended to pass the ``sampling_rate`` argument")
# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(43)
SAMPLE_RATE = 16000
# 自定义配置类
class CustomWav2Vec2Config(Wav2Vec2Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier_proj_size = kwargs.get("classifier_proj_size", 512)
        self.num_labels = kwargs.get("num_labels", 1)
        self.pooling_strategy = kwargs.get("pooling_strategy", "mean")
        self.mfccs_input_dim = kwargs.get("mfccs_input_dim", 39)  # 指定MFCC特征维度
        self.bilstm_hidden_dim = kwargs.get("bilstm_hidden_dim", 256)  # 双向LSTM的隐藏单元数
        self.bilstm_num_layers = kwargs.get("bilstm_num_layers", 1)
wav2vec2model = 'wav2vec2-large-xlsr-53-chinese-zh-cn/wav2vec2-large-xlsr-53-chinese-zh-cn'
config = CustomWav2Vec2Config.from_pretrained(wav2vec2model)
config.feature_fusion = 2560
config.dense_hidden_size = 512
# 加载音频文件及音调特征
def load_audio_and_compute_pitch_librosa(file_path):
    target_length = SAMPLE_RATE * 10  # 10秒对应的样本数
    frame_length = int(SAMPLE_RATE * 0.025)  # 25ms的帧长
    hop_length = int(SAMPLE_RATE * 0.01)
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    audio = torch.tensor(y).unsqueeze(0)
    pitch, _ = librosa.core.piptrack(y=y, sr=sr)
    pitch_input = np.max(pitch, axis=0)
    waveform_length = len(y)
    pitch_length = pitch_input.shape[0]
    interpolated_pitch = np.interp(
        np.linspace(0, pitch_length, waveform_length),
        np.linspace(0, pitch_length, pitch_length),
        pitch_input
    )
    if len(y) < target_length:
        padding = target_length - len(y)
        padded_y = np.pad(y, (0, padding), 'constant')  # 填充至10秒
    else:
        padded_y = y[:target_length]  # 截断至10秒
    mfcc_features = librosa.feature.mfcc(y=padded_y, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=frame_length)
    delta = librosa.feature.delta(mfcc_features)
    delta_delta = librosa.feature.delta(mfcc_features, order=2)

    # 合并MFCC特征、一阶差分和二阶差分
    mfccs_39 = np.concatenate([mfcc_features, delta, delta_delta], axis=0)
    return audio, interpolated_pitch, mfccs_39.T
# 自定义数据集类
class AudioDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=0, sep='\t')  # 不读取表头

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx, 1]  # 第二列是音频文件名
        label_str = self.data.iloc[idx, 2]  # 第三列是标签
        audio, interpolated_pitch,mfcc_features = load_audio_and_compute_pitch_librosa(file_name)
        # 将标签 "positive" 转换为 1，"negative" 转换为 0
        label = 1 if label_str == 'positive' else 0
        label = torch.tensor(label, dtype=torch.float32)
        return {'audio_input': audio, 'pitch_input': interpolated_pitch, 'label': label,'mfcc_input': mfcc_features,}
# 自定义数据 collator
class CustomDataCollator:
    def __init__(self, processor, audio_max_length):
        self.processor = processor
        self.audio_max_length = audio_max_length

    def __call__(self, examples):
        audio_inputs = [example['audio_input'].squeeze(0).numpy() for example in examples]
        audio_inputs = self.processor(audio_inputs, padding=True, return_tensors="pt", truncation=True,
                                      max_length=self.audio_max_length, sampling_rate=SAMPLE_RATE).input_values
        pitch_inputs = [example['pitch_input'] for example in examples]
        pitch_inputs = self.processor(pitch_inputs, padding=True, return_tensors="pt", truncation=True,
                                      max_length=self.audio_max_length, sampling_rate=SAMPLE_RATE).input_values
        labels = torch.tensor([example["label"] for example in examples])
        mfcc_inputs = [torch.tensor(example['mfcc_input'], dtype=torch.float32) for example in examples]
        mfcc_inputs = torch.stack(mfcc_inputs, dim=0)
        return {'audio_input': audio_inputs, 'pitch_input': pitch_inputs, 'label': labels,'mfcc_input': mfcc_inputs}
# 初始化处理器和数据集
processor = Wav2Vec2Processor.from_pretrained(wav2vec2model)
data_collator = CustomDataCollator(processor, 160000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载训练集、验证集和测试集
train_dataset = AudioDataset(csv_file="train_fold_3.csv")
val_dataset = AudioDataset(csv_file="val_fold_3.csv")
test_dataset = AudioDataset(csv_file="test.csv")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
# 训练参数
epochs = 30
best_f1 = 0
best_model_path = "./best_model_BiLSTM_co_attention_self_attention_mfcc39_wav_pitch_9_16_fold_3_3e_-5.pth"
# 初始化模型
model = AudioModel(config)
model.to(device)
model.freeze_feature_extractor()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
criterion = nn.BCEWithLogitsLoss()
for epoch in tqdm(range(epochs), desc="Training Epochs",leave=False):
    model.train()
    total_train_loss = 0.0
    for batch in tqdm(train_loader, desc="Training",leave=False):
        audio_input = batch['audio_input'].to(device)
        pitch_input = batch['pitch_input'].to(device)
        mfcc_input=batch['mfcc_input'].to(device)

        labels = batch['label'].to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(audio_input, pitch_input,mfcc_input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    # 在验证集上评估
    model.eval()
    total_val_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation",leave=False):
            audio_input = batch['audio_input'].to(device)
            pitch_input = batch['pitch_input'].to(device)
            mfcc_input = batch['mfcc_input'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            outputs = model(audio_input, pitch_input,mfcc_input)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(
        f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}, Val Recall: {recall:.4f}, Val F1: {f1:.4f}")

    # 保存最佳模型
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with F1: {f1}")

# 加载最佳模型
best_model = AudioModel(config).to(device)
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()
# 在测试集上进行评估
all_labels = []
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        audio_input = batch['audio_input'].to(device)
        pitch_input = batch['pitch_input'].to(device)
        mfcc_input = batch['mfcc_input'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        outputs = best_model(audio_input, pitch_input,mfcc_input)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        
        all_labels.extend(labels.squeeze(1).cpu().numpy())
        all_preds.extend(preds.squeeze(1).cpu().numpy())
# 计算测试集上的评估指标
test_accuracy = accuracy_score(all_labels, all_preds)
test_recall = recall_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds)
# 打印测试集上的评估结果
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1: {test_f1:.4f}")
