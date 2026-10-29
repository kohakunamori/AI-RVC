# AI-RVC Studio — 原生桌面 GUI 与实时变声

本地默认入口是 **PySide6 原生窗口**。无 WebView、无浏览器、无监听 HTTP 端口。原有 Gradio 页面仅作为兼容模式保留，可显式 `python run.py --web` 启动。Hugging Face Space 服务入口不变。

## 安装

Windows 首先根据自己的 NVIDIA 显卡和 CUDA 版本安装兼容的 PyTorch / torchaudio；之后在项目 Python 环境执行：

```powershell
python -m pip install -r requirements_desktop.txt
python run.py
```

或双击 `start-desktop.bat`。桌面依赖列表**不包含** Gradio / FastAPI / MCP。模型和已安装 Python 环境仍复用项目共享资源管理，不重复下载已有 RVC 权重。

如需使用 RoFormer 的 GPU/ONNX Runtime，按原项目 GPU 安装说明额外安装匹配的 `audio-separator[gpu]==0.47.0`。首次运行之前，请先准备原项目的 HuBERT、RMVPE 等基础模型；离线翻唱分离模型也按原项目流程准备。桌面程序**不会自动从互联网下载**模型，也不会启动本地 HTTP 服务器。

### 原生窗口

- **AI 翻唱**：RVC v1/v2 的 `.pth`、`.index` 或 Seed-VC / YingMusic / SoulX 的参考声音，支持完整歌曲/分离干声、音高、去混响、扩散步数。
- **实时变声**：选本地 RVC `.pth`，输入麦克风与输出声卡（可以是 VB-CABLE 等已安装的虚拟设备），调整 F0 算法、音高、缓冲、输入/输出增益。
- **模型管理**：和原项目 SVC 注册中心共用 `configs/svc_engines.json`，兼容引擎可使用共享 Python / 权重缓存。

### 实时模式运行边界

模型：仅限原有兼容的 **RVC v1/v2** 实时转换模型。**不能**把 Seed-VC / YingMusic / SoulX 直接当作低延迟麦克风模型调用。

路径：PortAudio `sounddevice.Stream` 捕获 32/40/44.1/48 kHz PCM，轻量回调写入最大 3 块队列；单独工作线程复用项目已有 `infer/lib/rtrvc.py` 的 HuBERT、RVC、RMVPE/FCPE；实时输出采用小范围 SOLA（相关对齐 + 交叉淡入淡出）。音频回调**不执行模型推理**，排队满时丢弃旧音频块，而不是无限堆积引起持续延迟。停止后关闭声卡流和释放模型资源。

默认块长 **250ms**、上下文 **1200ms**、交叉淡入淡出 **40ms**；实际端到端延迟还包含声卡缓冲、模型时间及队列等待，不可宣称低于 250ms 或“零延迟”。建议先使用耳机、48kHz 支持的同主机 API 声卡组合，并查看推理耗时是否小于块长。设备不支持双工时会显式报错。

使用虚拟声卡：把实时输出设备设为安装好的虚拟声卡播放端，在 Discord/录制软件中选择其录音端。请自行选择合规的已授权目标声线模型；避免外放回录造成啸叫。

**测试状态：** 提供 CPU 单元测试验证音频帧长度、SOLA 拼接、队列与桌面入口；真正的 GPU 推理、驱动 ASIO/WASAPI 兼容性、Windows 50 系显卡端到端实时延迟**必须在目标机器上实测**，不能靠 CI 证实。

### 依赖和开发方向

桌面 UI 不重复安装一个浏览器内核；PySide6 的许可证、分发规范与动态 Qt 库条款应在打包时复核。当前版本运行于已安装 Python 的本地环境，后续可加入 PyInstaller/Nuitka 打包以及统一自动安装程序，不在本次变更中承诺已完成 Windows EXE 发布。
