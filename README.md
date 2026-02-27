# WeChat Voice Saver

微信语音录制工具 - 自动录制微信语音消息并保存为 WAV 文件

## 环境要求

- Windows 10 2004+ 或 Windows 11
- Python 3.8+
- 微信 PC 版

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/Cairl/WeChatVoiceSaver.git
cd WeChatVoiceSaver
```

2. 安装依赖：
```bash
pip install process-audio-capture
```

## 使用方法

1. 确保微信 PC 版正在运行
2. 运行脚本：
```bash
python wechat_voice_recorder.py
```

3. 在微信中播放语音消息，程序将自动录制并保存

## 功能特点

- 自动检测微信语音播放
- 自动识别好友名称并分类保存
- 支持分段录制
- 实时音量显示
- Windows 中文环境适配

## 输出格式

录音文件保存在以好友名称命名的文件夹中：
```
./好友名称/20240215_143025 05s.wav
```

## 注意事项

- 打开与好友的聊天窗口并保持在前台，以便程序正确识别好友名称
