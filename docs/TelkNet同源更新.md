# TelkNet AI 翻唱同源更新

本轮将 TelkNet 的音频更新移植到 AI-RVC，运行时不连接 TelkNet。AI-RVC 基线为 `efb047e4d3d74bf1bfb6bd9455a467538b2b5a1a`，在原 `master` 工作区修改。

来源为私有仓库 `mason369/telknet` 的 `dev` 提交 `a1116773b55ededc81ab37d088eb81f7e1176e7f`（2026-10-23），路径 `external/ai-rvc`；以 Git 对象提取的该提交为准。项目原有 VC 固定提交 `81eed5e8f68b6bed1789f682fe78cdd324495afc` 和 UVR5 固定提交 `7ef19867780cf703841ebafb565a4e47d1ea86ff` 保持不变。

## 同步范围与适配

| 来源 | 本项目位置与处理 |
|---|---|
| `infer/audio_splice.py`、`rmvpe_boundaries.py` | 同步原算法，增加本项目新版官方入口的绝对坐标拼接适配 `lib/rvc_chunking.py`；短尾段限制上下文 |
| `infer/rvc_checkpoint.py`、`infer/lib/applio_algorithm` | 同步实际架构、原生精度和全层检查；保留第三方许可证及来源说明 |
| `infer/lib/rmvpe.py`、`infer_pack/models.py`、`attentions.py` | 同步 RMVPE 原生精度、向量解码与连续 F0 激励精度；本地兼容入口共用边界校验 |
| `infer/cover_pipeline.py` 的参考后处理 | 移植参考幅度/目标相位、有界缓存、候选选择和单次间隙抑制，保留本项目固定新版官方推理入口 |
| 纯人声输入行为 | 适配为 `CoverPipeline.process(input_mode="vocals")`；Web 两语言和播放器只展示实际两轨；原 MCP `convert_voice` 共用新版推理核心 |
| `infer/separator.py` | 同步固定配置校验、严格输出识别、静音主唱与短输入精度；保留本项目原有设备合同 |
| `lib/audio.py`、`lib/mixer.py`、`lib/vocoder_fix.py` | 保留模型 FLOAT64 输出、混音与 JUCE 混响计算；保留原参数/文件错误校验 |
| 本地角色资产兼容修正 | 标准 `trained_` / `added_` 双索引包仅在前者实际为零向量训练模板时选择后者；保留原文件，继续严格验证权重、维度和完整向量 |

未整目录覆盖 TelkNet 的旧官方入口、Web 服务适配和部署代码。AI-RVC 原有完整向量精确检索、固定资源哈希、跨平台设备选择继续有效。两项目固定 HuBERT/官方封装版本不同，本次一致性指同步算法与行为合同，不承诺随机推理结果逐字节相同。

## 输入与输出

`song` 是默认值，沿用整曲分离、去混响、转换、后处理和混音。`vocals` 直接解码输入进入转换，返回 `vocals`、`converted_vocals` 与真实 `all_files_dir`。不会把人声路径伪装成成品或伴奏。模型音高、索引、说话人、保护、响度参数继续校验；歌曲专属设置在纯人声界面隐藏且保留原值。

MCP 的 `convert_voice` 原本就是纯人声音色转换工具，本轮不扩展它的 API；它使用同一安全权重加载、RMVPE、精度与拼接实现。

## 验证记录

本机 Windows、RTX 4070 Ti SUPER 16 GB、PyTorch 2.11.0+cu126，独立 `venv310`；audio-separator 0.47.0、Gradio 5.49.1。模型使用项目固定下载器和 SHA-256 校验，歌曲来自用户本地音频。

| 验证 | 实际结果 |
|---|---|
| `venv310/Scripts/python.exe -m unittest discover -s tests` | 首轮 396 项通过；花丸双索引修复后完整重跑 399 项通过（46.943 秒）。包含参数化子用例：架构完整性、原生精度、F0 边带、RMVPE 转音、参考相位、短尾拼接、分离完整性、纯人声参数与两语言合同 |
| `npm ci --no-audit --no-fund`、`npm run build` | 多轨播放器依赖安装和构建通过，生成文件随本次更新 |
| `python -m pip check`、`git diff --check` | 无依赖冲突，无差异格式错误 |
| 默认整曲 / CUDA / Rin / 索引 0.5 / RMVPE | 本地 249.67 秒歌曲完成全部六模型链及混音；340.13 秒；七类输出均有限、非零、FLOAT WAV。人声与伴奏 44.1 kHz、转换人声原生 40 kHz；跨采样率时长差小于 0.1 ms |
| Far far away / CUDA / 国木田花丸 Zurakichi v2 | 原曲、权重和索引 SHA-256 与 TelkNet 既有任务记录完全匹配；音调 0、索引 0.5、RMVPE、保护 0.33、响度包络 0、主唱与伴奏音量 1。重新执行完整六模型链，耗时 325.89 秒；240.9836 秒全曲七轨均有限、非零、无满幅采样。混音 44.1 kHz 双声道 FLOAT，单独转换人声原生 40 kHz 单声道 FLOAT；跨采样率时间差小于 0.03 ms |
| 官方整曲 / CUDA / Rin / 索引 0.5 / RMVPE | 20 秒实际歌曲片段完成 UVR5、去混响、VC、混音；23.21 秒；四轨 FLOAT。UVR5 自身 STFT 边缘输出为 19.9946 秒，VC 保持该分离结果时间线 |
| 本地兼容入口 / 纯人声 / CUDA | 45 秒真实干声经 fairseq HuBERT、本地分块 RVC 完成，14.48 秒；两轨有效。原有 HuBERT 特征边缘使输出为 44.98 秒，拼接不再累计移位；后处理结果为 DOUBLE |
| MCP `convert_voice` / CUDA | 5 秒真实干声完成同一固定官方入口；12.06 秒；200000 帧、40 kHz、FLOAT |
| 浏览器纯人声 / CUDA | 真实上传 45 秒干声并点击转换，生成且仅显示输入/转换两轨；转换结果 1800000 帧、40 kHz、FLOAT，输入 1984500 帧、44.1 kHz |
| 浏览器播放与下载 | 多轨播放实际走时至 1.4 秒；键盘跳转到 45 秒、暂停、独奏、复位正常；真实点击下载所得 WAV 与工作区输出 SHA-256 相同 |
| `zh_CN` / `en_US` | 2/2 支持语言通过解析、键与占位符检查；真实切换后纯人声模式、已生成音轨及播放位置保留，播放器轨名与说明随语言更新 |
| 390 px 窄屏 | 英文页面宽度 390/390，播放器视口/内容宽度 334/334，无横向溢出；桌面中文、英文窄屏截图已目视核对；浏览器错误控制台 0 条 |

实测记录保存在本机 `C:\Temp\telknet-ai-rvc-sync-20261023`：`regression-final.log`、`real-current.json`、`real-variants.json`、`ui-final2.log`、`player-zh.png`、`player-en-mobile.png`。界面生成的两轨位于项目 `outputs/`；测试歌曲与模型未加入 Git。

后续《Far far away》试听位于 `outputs/farfar-hanamaru-20261023/`，`verification.json` 记录全部参数、源音频/模型/索引与输出哈希、采样数和信号检查；完整混音与单独转换人声直接复制原 FLOAT 结果，交付哈希一致。对应日志为 `farfar-conversion.log`、`farfar-character-tests.log`（28 项资产相关回归）及 `regression-farfar-final.log`（399 项完整回归）。首次实际下载发现同一压缩包中的空训练索引被误判为歧义，修正后重新通过正式下载器安装及整曲转换；没有删除训练模板或关闭索引绕过检查。

### 后续四首完整试听（2026-10-23）

用户指定的四份原曲均来自 `D:\Download`。沿用上次《Far far away》的花丸权重、索引与全部参数，逐首重新执行完整分离、去混响、RVC、后处理及混音。每首独立进程运行，使用本机 CUDA；批次于 17:28:41 开始、17:51:27 完成（UTC+8）。

| 原曲 | 完整成品时长 | 处理耗时 |
|---|---:|---:|
| 王心凌 - 梦的光点 | 231.853379 秒 | 321.68 秒 |
| 高槻かなこ - あこがれランララン | 249.173379 秒 | 340.19 秒 |
| Aqours - SILENT PAIN | 249.672132 秒 | 335.55 秒 |
| ヨルシカ - だから僕は音楽を辞めた | 242.973628 秒 | 330.63 秒 |

28 条实际输出音轨均完成全量解码，全部有限、非零且无满幅采样；与原曲时长的最大差为 0.046 ms，属于采样率转换的整帧取整。四首混音均为 44.1 kHz 双声道 FLOAT WAV，单独转换人声保留模型原生 40 kHz。8 份交付 WAV 与对应处理产物 SHA-256 一致，原曲处理前后的 SHA-256 一致；四份收据中的模型、索引及参数与上次试听记录完全匹配。

结果保存在 `outputs/hanamaru-four-songs-20261023/`，`batch.json` 汇总批次状态，各歌曲子目录含完整混音、单独转换人声、七类音轨及 `verification.json`。过程日志保存在原实测目录的 `four-songs-*.log`，执行脚本为 `four_song_listening.py`。

长歌曲首次测试发现极短尾段超过拼接上下文，已修复并加入极短首尾段回归，随后整曲重新从分离到混音完整通过。单元测试也检出了纯人声入口的导入遗漏，修复后完成上述浏览器实测。记录保留真实失败与重跑结果。

准备 1.6.0 提交时再次在最终源码执行全部 399 项回归，51.872 秒通过；重新构建播放器，`pip check` 与 495 个固定 VC/UVR5 源码文件校验均通过。日志为同一实测目录的 `release-1.6.0-regression.log`，后续成品验收单独记录于[发布准备](发布准备.md)。

本轮验证对象为上述本机源码和真实模型；Applio/FP64 等额外架构有完整权重与数值回归，不代表全部公开角色都完成试听。尚未复验的其他硬件、便携包、Docker 与线上部署不计入本轮通过范围。技术验收不等同于所有歌曲主观听感保证。
