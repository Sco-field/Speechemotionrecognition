# Deep Learning-Based Feature Fusion for Emotion Analysis and Suicide Risk Differentiation in Chinese Psychological Support Hotlines

This repository contains material associated with the paper: "Deep Learning-Based Feature Fusion for Emotion Analysis and Suicide Risk Differentiation in Chinese Psychological Support Hotlines". 
It contains
- code and material for reproducing the experiments on Negative Emotion Recognition and Fine-grained emotion multi-label classification
## Citation
- Anonymous authors. *Deep Learning-Based Feature Fusion for Emotion Analysis and Suicide Risk Differentiation in Chinese Psychological Support Hotlines*
## Download and install
- Model checkpoints download: please download Wav2Vec 2.0,
   `transformers-cli download jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`,
  and put checkpoints to the models path.

## Code Explanations
- [model_our.py](https://github.com/Sco-field/Speechemotionrecognition/blob/main/model_our.py) contains the model architecture of our proposed model,
- [train_ours.py](https://github.com/Sco-field/Speechemotionrecognition/blob/main/train_ours.py) contains the data-preprocessing,trainning and evaluating progress.
- [calculate_emotion_boostrap.py](https://github.com/Sco-field/Speechemotionrecognition/blob/main/calculate_emotion_bootstrap.py) contains the calculating bootstrap and p-value progress of the emotion number.
- [calculate_change_rate_bootstrap.py](https://github.com/Sco-field/Speechemotionrecognition/blob/main/calculate_change_rate_bootstrap.py) contains the calculating bootstrap and p-value progress of the change of rate

## Related public dataset
- [Visec](https://drive.google.com/file/d/1wAK6XcQBZgusyB8sDxlmuC3GhWbNUqCM/view?usp=sharing) [1]

## Reference
[1] Thanh P V, Huyen N T T, Quan P N, et al. A Robust Pitch-Fusion Model for Speech Emotion Recognition in Tonal Languages[C]//ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024: 12386-12390.


