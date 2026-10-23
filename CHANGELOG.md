# Changelog

## 1.6.0 候选版 / Release candidate — 2026-10-23

### 中文

- 从 TelkNet `dev` 固定提交 `a1116773b55ededc81ab37d088eb81f7e1176e7f` 移植同源音频更新，保持 AI-RVC 独立运行及当前固定官方源码，不增加 TelkNet 服务依赖。
- 新增整曲 / 纯人声输入选择，配置和处理入口支持 `song` / `vocals`。纯人声跳过分离、去混响、源约束与混音，只导出实际输入及转换两轨；界面隐藏不适用参数，中英文及多轨播放器同步更新。
- 统一安全检查点加载、真实架构识别与全层校验，支持标准 v1/v2、非 F0、多说话人及 Applio HiFi-GAN / MRF / RefineGAN。RVC、RMVPE 独立遵守原生权重精度，HuBERT 保持固定 FP32；索引保持完整 FP32 向量精确检索。
- 修复连续 F0 和激励相位的低精度帧率边带；向量化 RMVPE 峰值邻域解码，并按原始波形核验提前转音及短缺口。保留真实无声帧，让清辅音保护生效。
- 长音频按绝对采样坐标裁剪上下文，使用短且互补的交叉淡化；极短尾段按实际长度限制上下文，不移动后续歌词时间位置。
- 后处理使用参考幅度和目标相位，加入有界参考缓存；质量候选选择后只做一次间隙抑制。FLOAT64 音频、混音和混响不经过 FLOAT32/PCM16 中转。
- 分离配置读取统一验证固定资源，运行时配置写入会话目录；检查输出文件唯一性、主唱模型语义、采样率/通道/帧数和非有限值。纯伴奏区间允许静音主唱，不按相对能量交换音轨。短输入补长和裁切保留浮点精度。
- 安装器兼容新版 `CUDA UMD Version` 驱动信息；本地 Gradio 回环自检绕过系统代理，并保留已有代理排除项。
- 修复花丸等角色的标准双索引压缩包安装失败：验证 `trained_` 为零向量训练模板后，选择同名 `added_` 推理索引；保留原文件及完整向量校验，损坏或真正冲突的索引仍拒绝。
- 修复干净环境准备默认模型时误报缺失：持久资源清单只要求六组固定权重与原始配置，派生运行配置在会话目录生成。真实模型在干净目录中通过 SHA-256 校验及下载器准备；移除会话配置后仍可正常检查就绪，原始配置缺失仍报错。
- 补齐 Docker CPU/CUDA 运行镜像的 `libatomic1`。在相同基础镜像实际复现 Pedalboard 因 `libatomic.so.1` 缺失无法加载的问题，新增成品依赖检查中的直接导入；保留 Float32 / Float64 混响数值回归。
- 增加已构建便携包的草稿暂存流程：严格核对成功运行、源码提交、版本、四种包及分卷 SHA-256，将原件附到对应 Release 草稿；不重打包、不覆盖不同文件、不公开发布，成品运行验收另行记录。
- 草稿工作流改用固定提交的官方 artifact 下载步骤，强制下载摘要检查并设置 20 分钟超时；暂存继续核对既有构建的准确源码、四种包目录和每个原始分卷。该发布工具调整不改变候选应用产物的源码提交。
- 修复冻结程序官方 UVR5 路线因旧模块路径缓存崩溃：替换应用包时同步失效这些包在 `inspect` 中的双向文件索引，保留无关依赖缓存及冻结依赖搜索路径。新增真实缓存回归，覆盖命名空间包、普通包和旧代码定位；路径检查解析 Windows 短路径和 macOS 临时目录链接。Windows CPU/CUDA 诊断副本的官方路线各完成 4 条 Float WAV，正式新包另行重建验收。
- 增补数值、参数、分离完整性、原生精度、纯人声、长音频与干净安装回归，连同草稿暂存和模块缓存验证，本地全量 414 项通过。花丸同一模型与参数完成《Far far away》《梦的光点》《あこがれランララン》《SILENT PAIN》《だから僕は音楽を辞めた》五首全曲；另完成 Rin 整曲、官方 UVR5、本地兼容入口、MCP 与浏览器纯人声实测。两种界面语言、窄屏、播放和原始 WAV 下载均已实际检查。
- 统一源码、Compose 与 OCI 构建版本为 1.6.0，更新 README、架构图、兼容性、播放器模式及发布准备文档。模型、索引、原曲和试听音频不进入发行源码。来源和量化结果见 `docs/TelkNet同源更新.md`；当前为候选版，四种便携包及两种 Docker 镜像须分别完成本版成品验收后才正式发布。

### English

- Port the shared audio updates from TelkNet `dev` commit `a1116773b55ededc81ab37d088eb81f7e1176e7f`. AI-RVC remains standalone and retains its pinned upstream sources.
- Add `song` and `vocals` inputs across configuration, processing and the bilingual UI. Vocal input skips separation, dereverb, source constraints and mixing, and exports only input and converted vocals. The player and applicable controls follow the selected mode.
- Share safe checkpoint loading, architecture detection and complete weight validation across standard v1/v2, non-F0, multi-speaker and Applio HiFi-GAN / MRF / RefineGAN models. RVC and RMVPE keep independent native precision; the pinned HuBERT stays FP32, and index retrieval retains exact full FP32 vectors.
- Fix low-precision continuous-pitch and oscillator sidebands. Vectorize RMVPE decoding and verify premature note transitions and short gaps against the waveform while preserving actual unvoiced frames.
- Crop long-form inference by absolute sample positions and join aligned context with short complementary fades. Bound context for very short terminal segments without shifting the source timeline.
- Use reference magnitude with target phase, bounded reference caches and a single gap-suppression pass after quality selection. Preserve FLOAT64 through output, mixing and reverb.
- Verify pinned separation configurations, unique output files, model stem semantics and audio frames/rate/channels/finiteness. Keep legitimately silent vocal stems in instrumental sections. Retain precision during short-input padding and cropping.
- Support the new `CUDA UMD Version` driver header and bypass proxies for Gradio loopback checks while preserving existing exclusions.
- Accept standard paired `trained_` / `added_` character archives, including Hanamaru. Verify the former is an empty trained template before selecting the latter; preserve both files and full vector validation, and reject corrupt or ambiguous indices.
- Fix false missing-model errors during clean installation: persistent assets include the six pinned weight/configuration pairs, while derived runtime configurations belong to the session directory. Verify preparation with real model hashes in a clean directory; readiness survives removing session configurations and still rejects missing original configurations.
- Include `libatomic1` in both Docker runtime variants. Reproduce Pedalboard's missing `libatomic.so.1` failure with the same base image and add a direct component import to image checks, retaining the Float32/Float64 reverb regression.
- Stage existing portable build artifacts in a matching release draft after verifying the successful run, exact source commit, version, all four packages and volume SHA-256 checksums. Preserve original archives, refuse differing replacements, and keep the release unpublished; runtime acceptance is recorded separately.
- Use the commit-pinned official artifact download action with mandatory digest verification and a 20-minute download timeout. Staging still validates the original build commit, all four package directories and every archive volume. This release-tool change does not change the candidate application's source commit.
- Fix the frozen official UVR5 route crashing on stale module paths. Invalidate both `inspect` file mappings only for replaced application packages, preserving unrelated caches and bundled dependency paths. Regressions exercise actual caches with namespace packages, regular packages and old-code lookup, resolving Windows short paths and macOS temporary-directory links. Windows CPU/CUDA diagnostic copies each produce four valid Float WAV tracks; rebuilt release candidates require separate acceptance.
- Pass all 414 local regressions for numerical behavior, parameters, separation integrity, native precision, vocal input, long-form conversion, clean installation, draft staging and module caches. Complete five full Hanamaru covers with identical settings: “Far far away”, “梦的光点”, “あこがれランララン”, “SILENT PAIN” and “だから僕は音楽を辞めた”. Also exercise a full Rin cover, official UVR5, the local compatibility path, MCP and browser vocal conversion. Verify both UI languages, mobile layout, playback and original WAV downloads in a real browser.
- Set source, Compose and OCI build versions to 1.6.0 and update architecture, compatibility, player-mode and release documentation. Exclude model weights, indices, source songs and listening outputs from the source release. See `docs/TelkNet同源更新.md` for provenance and measurements. This is a release candidate; all four portable packages and both Docker images require separate acceptance before public release.

## 1.5.1 — 2026-09-06

### 中文

- 修复 UVR5 重建频谱时读取未初始化频点、偶发产生 NaN 的上游缺陷：内置实现与独立固定官方运行时均先清零缓冲区，保留复数精度、模型及重采样参数。含 NaN 旧内存的确定性回归和 Windows/Linux 真实权重分离均通过，全量回归为 343 项。
- 修复 PyTorch 2.11 经由 cuda-toolkit 间接引入的 NVIDIA 动态库漏装：激活官方收集钩子并补齐带版本号的动态库和插件；本地 spec 与 Actions 同步，GPU 成品必须通过 NVRTC 实际编译检查。1.5.1 草稿因 Linux GPU 真实翻唱发现缺库而未发布，不关闭源约束或降低质量参数规避错误。
- 同时准备固定版 VC 与 UVR5 源码，修复干净构建环境只内置 VC、切换 UVR5 时缺少官方源码的问题。Actions 与本地 spec 在打包前后逐文件校验两套源码。1.5.0 草稿在成品验收中被拦截，未公开发布；本版包含以下全部更新。
- 修正 Windows 英文系统读取中文依赖清单时的编码错误；相关 CI 命令采用遇错即停的 shell，避免前面的安装或测试失败被后续成功命令覆盖。
- 修正 Apple Silicon 安装器和 CI 引用了不存在的 TorchAudio 2.13：采用官方稳定 ABI 兼容组合 PyTorch 2.13.0 / torchvision 0.28.0 / TorchAudio 2.11.0，依赖清单与版本检查同步约束；增加真实重采样回归。分离模型、推理参数及 Windows/Linux 运行栈保持不变。
- Windows 全量回归增至 343 项并通过；主分支 Windows、Ubuntu、Apple Silicon 三平台依赖安装及合同测试均通过。CI、便携包和 Docker 工作流的 GitHub/Docker Actions 统一使用已发布的 Node.js 24 版本，消除 Node.js 20 弃用项；应用运行时版本不变。
- 四种正式便携包均完成默认与官方路线的真实翻唱，共校验 44 个 Float32 输出；两种正式 GHCR 镜像均匿名拉取并断网运行两条路线，另校验 22 个输出。发行分卷、逐文件清单、镜像摘要与上一版全量对照随 Release 提供。
- 增加 Linux x86-64 CPU/CUDA Docker 镜像和 Compose：固定完整 PyTorch 栈，多阶段构建前端与依赖，非 root 运行，单数据卷、可选密码文件登录、健康检查和显式设备失败；默认六模型、精度和音频输出策略不变。
- 配置原子保存保留容器文件软链接，升级镜像继续使用原有配置和模型；本地浏览器启动行为保持，容器可无浏览器启动并设置反向代理子路径。
- 兼容新版 FastAPI 的懒路由，保留下载鉴权、Range 和中文文件名；容器仅开放实际输出目录供下载。启动配置固定设备时，界面只读且保存接口明确拒绝无效修改。
- 六个默认分离模型固定到已实测的提交与 SHA-256，已有有效缓存可断网使用；CPU 环境不预加载 CUDA 库，实际依赖或下载错误继续完整报告。
- 发布工作流默认生成审核产物，上传草稿必须绑定一致的版本标签；完整回归、前端构建、包内文件清单和压缩包校验码进入流程，公开文档和截图随便携包分发。源码版本、Compose 默认镜像与 OCI 构建版本统一为 1.5.1。

- 完全移除 PolarFormer 运行封装、混合预设、专用窗口参数及下载入口，新包不再包含遗留权重；旧配置显式报错。纯伴奏统一使用已设为默认的 Leap Instrumental，保留历史对比测量。

- 跨平台安装改为一次解析完整项目依赖，并保留所选 PyTorch 构建；补齐 Apple Silicon 的 PyTorch 2.13 与 ARM 音频库要求、Blackwell CUDA 12.8 选择，明确拒绝不兼容组合。MCP 1.x 与 Gradio 5.49.1 保持兼容。
- 修复同长度、同修改时间的权重或索引覆盖可能复用旧校验结果的问题；模型校验依据内容哈希。便携包对固定官方源码逐文件校验，不再依赖用户机器安装 Git；本地 spec 与发行流程统一使用目录包；补齐 safehttpx/groovy 资源，VC/UVR5 隔离导入保留冻结程序的依赖搜索路径。
- 前期候选 Windows 源码、Docker CPU 与 CUDA 运行时各 335 项回归通过；两个镜像均完成断网默认六模型、RVC 与七类 Float WAV 输出。登录、子路径代理、上传下载、数据持久化和备份恢复均已实测。Colab 安装成功后被平台使用限制断开，HF 新建 CPU Space 返回 402；不把受阻云端及未接入硬件计为验收通过。

- README 更新真实多轨波形、翻唱设置与可下载角色目录截图；区分伴奏受控实测、公开榜单与去混响权重名称指标，补充逐平台验收范围。
- 本地、Colab、Space 与打包共用固定版 pip/setuptools 安装工具；补齐 CPU Space 功能依赖与完整 PyTorch 栈，正常解析依赖并执行检查。
- 原生 PyTorch XPU 检测不再强制依赖 IPEX；固定官方源码校验兼容 Windows/WSL 共享工作树的换行符，仍拒绝真实内容修改。

- 分离引擎统一升级到 audio-separator 0.47.0，同步本地 CPU/CUDA/DirectML、HF、Colab、安装检查与打包依赖。打包收集完整分离器模块、模型配置及版本元数据。
- RoFormer 使用模型 YAML 的重叠参数，保持 FP32 常规推理；接入上游分块与输出修正，移除按轨道类型猜测缺失输出文件的兼容处理。输出、耗时可能随重叠计算修正而变化。
- 明确拒绝上游 0.47.0 不支持的 MDXC/RoFormer + DirectML 组合，避免选择 DirectML 后自动在 CPU 推理。
- Windows CUDA 源码的约 241 秒整曲和 5 秒短音频均完成默认翻唱，六模型参数生效、七类 Float WAV 输出完整。此前 Windows CUDA 实际包完成默认短音频与 UVR5 验收；正式便携包采用本版最终源码构建，逐包验收记录随 Release 提供。技术验收不作为主观听感提升证明。

- 修复参数实际生效问题：UVR5 分离不再被音色转换引擎开关改成 Demucs；默认说话人 ID 按配置初始化并校验。补齐 FCPE 静音门限提取器，保留 PM 无声帧的零值。
- 固定版官方 RVC 保留音高提取器的无声标记，修复插值使清辅音保护失效的问题。实际权重、固定随机种子验证修正前保护参数输出相同、修正后生效；不修改官方源码目录或模型权重。
- 保留现有页面与操作流程，压缩顶部留白并对齐标题和语言栏；状态文本按内容展开、模型路径自动换行。修正 Gradio 样式重写导致的窄屏规则失效，优化手机页签、按钮与多轨触控尺寸。
- 简化界面操作说明和中英文文案；处理流程展示选定步骤，模型面板分别显示文件、组件与校验状态。运行代码标记移至设置详情。
- 修正默认角色详情、中文无障碍名称和播放器错误提示；数值校验使用界面语言并保留失败原因。
- 更新 README、HF README、Colab 和兼容性文档，移除编辑过程残留并修正过时参数说明。

- 角色模型下载/导入先在暂存目录校验完整权重、真实架构与索引，再安装；失败显式报告，替换前的文件保留备份。列表不再把发布名 v2 当作架构，拒绝按文件名相似度猜索引。
- 修复稀疏 IVF 索引返回无效近邻与精确重复向量除零：当前/官方两路共用完整 FP32 向量的 FlatL2 精确检索，不关闭索引、不降维、不改写资产；支持中文索引路径。
- Google Drive 对齐 gdown 6.0.0 的大文件确认接口；Mega 增加哈希锁定的 Megatools 客户端。181 个条目中 178 个 HF 来源文件存在、2 个 Drive 来源实际下载；樱坂雫原 Mega 来源 ENOENT，明确保留失效状态。8 个本地角色共 16 次带索引双路线真实推理通过，其余权重尚未完成推理验证。

- 翻唱结果增加与 TelkNet 对齐的多轨时间轴，支持同步播放、静音、独奏、逐轨增益、时间偏移、缩放和拖入对照轨；默认只启用转换后人声与伴奏，七类输出保留独立试听和原始 WAV 下载。
- 播放器默认整曲适配窗口、高度自适应，中英文及离线资源与 Gradio 共用；试听不改写音频，加载或解码失败显式停止。独立试听与多轨播放互斥，防止成品与分轨意外叠加。
- 按 Gradio 上游修复上传进度 ID 的初始化，补齐流连接取消；Windows 网页服务器显式使用 Selector 套接字事件循环，避免媒体断连触发 Python 3.10 Proactor 清理异常。日志保留完整 traceback，错误不会被过滤。

- 默认纯伴奏采用标准 Leap Instrumental 62；人声保留 Leap XE 90，主唱/带和声伴奏使用 MVSep 9205，去混响更新为 Stereo De-Reverb 22.5050。
- 移除默认入口中未消费的滤波半径、HuBERT 层数、旧 F0 调优配置与重复去混响开关；配置和 MCP 拒绝未知参数，UI 按模型及路由禁用不适用控件。
- 按真实权重支持 RVC v1/256、v2/768、F0/无 F0、32/40/48 kHz 和多说话人；校验完整推理权重及同维度 FAISS 索引，不填充/截断特征，不加载缺层的随机参数。
- 音频中间结果与混音使用 Float32 WAV；固定版官方 VC/UVR5 采用编码层适配，避免先量化 PCM16 再写入浮点容器。
- MCP 包改为 `rvc_mcp`，解决与 SDK 同名冲突；转换使用隔离进程，避免 Windows 原生数值库初始化阻塞 stdio；模型列表返回唯一 ID。
- 离线官方转换固定常规推理，修正变长整曲反复预热和保留 CUDA Graph 显存的问题；模型及 FP32/FP16 策略不变。
- Demucs 改用统一浮点解码并汇总全部非人声轨；补齐 FCPE 依赖，移除本地未实现的 DIO 配置；MCP 失败同步设置协议错误标志。
- 补正冻结程序工作进程调度与打包资源；GPU 构建配置对齐 PyTorch 2.11/CUDA 12.8。Windows CUDA 包已完成默认短音频和 UVR5 实跑，云端部署仍未验收。
- README、HF README、Colab 和兼容性说明同步更新；云端部署与未接入硬件的验收范围单独记录。

- 界面语言即时切换，合并标签与候选项更新，保留手动参数和多轨会话；筛选后清理失效角色选择。
- 基础模型固定提交并校验大小与 SHA-256，下载采用可验证续传及原子发布；失败不覆盖现有文件，脚本返回真实失败退出码。
- 配置原子保存，预设工具支持任意工作目录；拒绝非有限数值、小数整数参数和不存在的设备编号。修正 Gradio 带 API 前缀的下载路由，保留授权及 Range 请求。

### English

- Initialize unused UVR5 spectrum bins in both the bundled implementation and pinned upstream runtime, fixing intermittent NaN audio caused by uninitialized memory. Preserve complex precision, models and resampling parameters. Deterministic dirty-memory regression, real Windows/Linux weight inference and all 343 tests passed.
- Prepare both pinned VC and UVR5 source trees. Fix clean portable builds that included only VC and failed when selecting UVR5. Actions and the local spec validate every pinned source file before and after bundling. The 1.5.0 draft was rejected during artifact acceptance and was never published; this release includes all changes below.
- Declare UTF-8 for dependency files on non-UTF-8 Windows systems. CI shell steps now stop at the first failed command so later commands cannot hide installation or test failures.
- Fix the nonexistent TorchAudio 2.13 pin in the Apple Silicon installer and CI. Use the officially compatible stable-ABI stack PyTorch 2.13.0 / torchvision 0.28.0 / TorchAudio 2.11.0, align dependency checks and add a real resampling regression. Preserve separation models, inference settings and the Windows/Linux runtime stack.
- Pass all 343 Windows regression tests and the Windows, Ubuntu and Apple Silicon dependency/contract CI jobs. Update GitHub/Docker Actions in CI and package/image workflows to released Node.js 24 versions, removing deprecated Node.js 20 references without changing application runtimes.
- Validate both real cover routes in all four distributed portable apps, producing 44 Float32 outputs. Anonymously pull both public GHCR images and run both routes without network access, validating another 22 outputs. Publish volume checksums, file inventories, image digests and the complete previous-release comparison with the Release.
- Add Linux x86-64 CPU/CUDA Docker images and Compose with a pinned PyTorch stack, multistage frontend/dependency builds, non-root execution, one persistent data volume, optional secret-file login, health checks and explicit device errors. Preserve the complete model chain, precision and output policy.
- Keep configuration symlinks intact during atomic saves. Recreated containers retain settings and models; headless startup and reverse-proxy paths are optional without changing desktop startup defaults.
- Support lazy routers in newer FastAPI while preserving download authentication, ranges and Unicode filenames. Allow only the actual output directory for container downloads. Make startup-pinned device settings read-only and reject ineffective save requests.
- Pin all six default separator models to tested revisions and SHA-256 hashes for offline cache reuse. Avoid CUDA library preloading on CPU; retain complete dependency and download errors.
- Collect NVIDIA libraries introduced indirectly through PyTorch 2.11's cuda-toolkit dependencies, using upstream hooks plus versioned library/plugin collection. Require a real NVRTC compilation check in both local and Actions GPU builds. The v1.5.1 draft was rejected after its Linux GPU cover exposed the omission; source constraints and quality settings remain enabled.
- Make manual release builds review-only by default. Draft uploads require matching source/version tags; run the full suite, build the player, include public documentation and generate package inventories/checksums. Align the source, Compose image defaults and OCI build version at 1.5.1.

- Remove the PolarFormer runtime, hybrid preset, dedicated window controls, and download entry points; new builds exclude any remaining cached weights. Retired model IDs fail explicitly. Pure accompaniment uses the existing Leap Instrumental default; historical comparison measurements remain available.

- Resolve the full project dependency set in one transaction while preserving the selected PyTorch build. Add Apple Silicon runtime/library requirements and Blackwell cu128 selection; keep MCP 1.x compatible with Gradio 5.49.1.
- Validate model and index contents despite unchanged file size and timestamps. Portable builds verify all pinned upstream source files without requiring system Git and use directory bundles consistently.
- Earlier candidate Windows source and both Docker runtimes passed 335 tests each. Both images completed network-disabled default six-model/RVC covers with seven Float WAV outputs. Authentication, subpath proxying, uploads/downloads, persistence and backup/restore were verified. Colab was terminated by platform restrictions after installation; new HF CPU Space creation returned HTTP 402. These cloud routes and unavailable hardware remain unverified.

- Refresh README with actual multitrack waveforms, cover controls and the downloadable character catalog. Separate controlled accompaniment scores from public leaderboard and checkpoint-name metrics; document platform-specific evidence.
- Share pinned pip/setuptools bootstrap across local, Colab, Space and package builds. Include all CPU Space feature dependencies, install the complete PyTorch stack and validate dependency resolution instead of bypassing it.
- Detect native PyTorch XPU without requiring IPEX. Normalize Git checkout line endings when validating pinned upstream sources shared between Windows and WSL, while still rejecting content changes.

- Upgrade audio-separator to 0.47.0 across local CPU/CUDA/DirectML, HF, Colab, installer checks and packaging. Bundle separator modules, model configuration data and distribution metadata.
- Honor model-configured RoFormer overlap with FP32 eager inference. Adopt upstream chunk/output fixes and reject missing output paths instead of substituting existing stems. Corrected overlap can change output and runtime.
- Reject MDXC/RoFormer on DirectML because upstream 0.47.0 substitutes CPU for that unsupported allocator; do not report GPU execution while running on CPU.
- Windows CUDA source completed default covers of a roughly 241-second song and a 5-second clip, with six effective model configurations and seven Float WAV outputs. The earlier Windows CUDA portable app passed a default short cover and real UVR5 separation; final portable builds use this version's tagged source, with package validation recorded in the Release. Listening improvement is not inferred from technical checks.

- Make UVR5 selection independent of the VC engine and initialize the speaker ID from validated configuration. Add the missing FCPE extractor for silence gating and preserve PM unvoiced zeros.
- Preserve native unvoiced F0 decisions in the pinned RVC adapter so consonant protection works. Fixed-seed, real-weight runs confirm that protection previously produced identical outputs and now changes them. Keep the upstream checkout and model weights intact.
- Preserve the existing pages and workflows while aligning the header and language controls, reducing excess spacing, expanding status text and wrapping model paths. Fix responsive rules affected by Gradio CSS rewriting and improve mobile tabs, buttons and multitrack touch targets.
- Switch interface language immediately while preserving manual controls and multitrack state; merge component updates and clear invalid filtered selections.
- Pin base-model revisions and verify size/SHA-256. Use validated resume and atomic publication, preserve existing files on failure, and propagate script exit codes.
- Save configuration atomically, resolve preset paths independently of the working directory, reject invalid numeric/device inputs, and preserve authorization and ranges on Gradio file routes.


- Simplify interface copy in both languages. Show selected workflow steps separately from model-file, component-import and validation status; move the runtime build marker to settings details.
- Correct initial character details, Chinese accessible names and player errors. Localize numeric validation and retain failure details.
- Update README, HF README, Colab and compatibility documentation, removing editorial residue and outdated parameter instructions.

- Validate character weights, native architecture and indices in staging before installation/import. Preserve replaced assets, surface failures, distinguish distribution labels from RVC versions and reject ambiguous index pairing.
- Use shared exhaustive FlatL2 retrieval over original FP32 vectors in both VC routes, fixing sparse-IVF invalid neighbors and zero-distance weighting. Preserve dimensions and index files; support Unicode index paths. Exact search can cost more memory/time.
- Use gdown 6.0.0 for public Drive confirmation forms and pinned, SHA-256-verified Megatools builds for MEGA. Check remote file presence for 178 HF entries, download both Drive entries and complete 16 indexed runs across 8 local characters. Keep the Shizuku Osaka entry visibly source-unavailable (MEGA ENOENT); Other registered weights remain untested.

- Add the TelkNet multitrack timeline with synchronized playback, mute/solo, gain, offsets, zoom and draggable comparison tracks. Start with converted vocals and accompaniment; retain standalone playback and original WAV downloads for all seven output roles.
- Fit the complete timeline to the available width and resize the embedded player to its content. Bundle all frontend assets and both locales. Preview edits do not rewrite audio; loading/decoding errors stop the mixer. Standalone and multitrack playback are mutually exclusive.
- Backport Gradio's upload progress ID initialization fix and implement actual stream cancellation. Use an explicit Selector socket loop for the Windows web server to avoid Python 3.10 Proactor cleanup failures on media disconnects. Preserve exception tracebacks without filtering errors.

- Use standard Leap Instrumental 62 for pure accompaniment, retaining Leap XE 90 vocals, MVSep 9205 lead/backing+instrumental and Stereo De-Reverb 22.5050.
- Remove unused default-route parameters and duplicate preprocessing controls. Reject unknown config/MCP fields and disable controls that do not apply to the selected model or route.
- Validate native RVC v1/256 and v2/768, F0/non-F0, 32/40/48 kHz, speaker IDs and matching FAISS indices. No feature padding/truncation, missing-layer random initialization or low-bit quantization.
- Preserve Float32 WAV intermediates and exports with checked encoding-only adapters for pinned upstream VC/UVR5.
- Rename the MCP package to `rvc_mcp`, isolate numerical conversion from stdio threads and expose unambiguous model IDs.
- Use eager execution for offline conversion without changing model precision. Avoid repeated CUDA Graph warmup and retained memory pools for variable-length clips.
- Use float decoding and all non-vocal stems in Demucs, include FCPE dependencies, reject unsupported local DIO settings and propagate MCP protocol errors.
- Correct frozen worker dispatch and bundled assets; align GPU build definitions with PyTorch 2.11/CUDA 12.8. The Windows CUDA portable app passed a default short cover and real UVR5 separation; cloud E2E remains unverified.
- Cloud deployment and hardware-specific acceptance are recorded separately from source and portable-package validation.

## v1.4.0 - 2026-07-12

### 中文更新说明

- TelKNet 默认分离链路已对齐：非 WAV 输入统一解码为 44.1 kHz 双声道 PCM16，Leap XE 与 PolarFormer 使用同一份输入；PolarFormer 增加孤立声道饱和抑制；MVSep 9205 三模型 `avg_wave` 从原始整曲分离主唱与带和声伴奏。
- 输出语义与生产环境保持一致：`backing_vocals.wav` 为 Leap 人声减去 MVSep 主唱得到的纯和声，`accompaniment.wav` 为 MVSep `Back+Instrumental`，`accompaniment_without_harmony.wav` 为 PolarFormer 纯伴奏。
- 移除未调用的旧 Karaoke 和声回混链路，以及会把 `bs_polarformer_124bands_fp16` 错当成 62-band ONNX 模型的兼容别名；错误模型 ID 现在会显式失败。
- README、Hugging Face README、Colab、依赖注释、诊断脚本和 Web UI 文案已同步说明 Leap XE + BS PolarFormer + MVSep 9205 + RoFormer De-Reverb 默认链路及三类输出语义。
- `tools/download_models.py`、`run.py`、HF `app.py`、MCP 模型状态和 GitHub Release 打包流程现在都会显式准备/检查 Leap XE vocals、BS PolarFormer pure accompaniment、MVSep 9205 子模型与 RoFormer De-Reverb。
- UI 路由状态现在明确展示统一 PCM、Leap/PolarFormer、MVSep 9205、纯和声推导、RoFormer De-Reverb、RVC 和混音的真实顺序。
- MVSep 9205 现在按榜单口径直接处理原始整曲，短音频会补到三模型所需的最大窗口，再把输出裁回原时长。
- 单卡流程会在 MVSep 前卸载 Leap XE 与 PolarFormer；最终混音直接使用 MVSep 的带和声伴奏轨，PolarFormer 纯伴奏仍单独导出，不会重复叠加乐器。
- 运行日志不再把 MVSep `Back+Instrumental` 写成 `backing_vocals.wav`：默认 Karaoke 临时输出改为 `karaoke/accompaniment.wav`，纯和声仍为会话根目录的 `backing_vocals.wav`；进度、模型加载和混音日志统一使用“纯伴奏 / 带和声伴奏 / 纯和声”。
- 显式选择 CUDA、XPU、DirectML 或 MPS 后，设备不可用会直接报错；只有 `device=auto` 才会自动选设备。CPU 安装和 CPU 便携包会显式写入 `device=cpu`。
- Karaoke 输出无法唯一识别主唱与第二路、或检测到疑似反转时会停止，不再按文件顺序猜测或自动交换。
- Karaoke 开启时，`accompaniment.wav` 始终导出 MVSep `Back+Instrumental`；关闭成品和声时只改变混音输入，不改变公开输出语义。
- ONNX Runtime 改由 `audio-separator` 的 CPU/GPU/DML extra 安装，避免 CPU 与 GPU 发行包同时写入同一模块；GPU 便携包不再宣称缺少 CUDA 时自动回退 CPU。
- Windows 控制台和重定向日志统一使用 UTF-8，避免中文/English 混合输出在 CI、Codex 或发布诊断日志中乱码。
- 控制台消息现在严格跟随 `zh_CN` / `en_US`：英文模式使用随包发布的离线目录，漏译会显式报错，不再混入中文。
- `docs` 目录中的 Markdown 已从仓库移除并加入忽略规则；发布要求保留在 README 与 Changelog 中。
- PyTorch、torchvision 与 torchaudio 现在从同一 PyTorch 索引一次性安装；`install.py --check` 会执行 `pip check`，依赖版本不一致时直接失败。
- 2026-07-12 本地验证：完整 170 项测试全部通过；RTX 4070 Ti SUPER 真实 CUDA 完整翻唱已覆盖 Leap XE、PolarFormer、MVSep 9205 三模型、RoFormer De-Reverb、RVC 与最终混音，英文控制台全程无汉字。

### English Notes

- Default vocals/instrumental separation now uses `hybrid:leap_xe90_vocals+polarformer62_instrumental`: Leap XE 90 bands extracts vocals from the original full mix and BS PolarFormer public ONNX 62 bands extracts pure accompaniment; default lead / backing+instrumental separation uses `ensemble:mvsep_9205_avg` on the original full mix.
- README, Hugging Face README, Colab, dependency comments, diagnostics, and Web UI copy now describe the Leap XE + BS PolarFormer + MVSep 9205 + RoFormer De-Reverb default route and its three output stems.
- `tools/download_models.py`, `run.py`, HF `app.py`, MCP model status, and GitHub release packaging now explicitly prepare/check Leap XE vocals, BS PolarFormer pure accompaniment, the MVSep 9205 submodels, and RoFormer De-Reverb.
- The TelKNet production separation route is now aligned end to end: non-WAV input is decoded once to 44.1 kHz stereo PCM16, Leap XE and PolarFormer share that input, PolarFormer applies isolated-channel saturation suppression, and MVSep 9205 runs its three-model `avg_wave` ensemble on the original mix.
- Export semantics now match production: `backing_vocals.wav` is pure harmony derived as Leap vocals minus MVSep lead, `accompaniment.wav` is the MVSep `Back+Instrumental` stem, and `accompaniment_without_harmony.wav` is the pure PolarFormer accompaniment.
- Removed the unused legacy Karaoke backing-remix path and the alias that treated `bs_polarformer_124bands_fp16` as the 62-band ONNX model; invalid model IDs now fail explicitly.
- MVSep 9205 now processes the original full mix, pads short inputs to the longest model window, and trims both outputs back to the source duration.
- Single-GPU runs unload Leap XE and PolarFormer before MVSep. The final mix uses MVSep backing+instrumental directly while keeping the pure PolarFormer accompaniment as a separate export.
- Runtime logs no longer label the MVSep `Back+Instrumental` stem as `backing_vocals.wav`: the default Karaoke temporary output is now `karaoke/accompaniment.wav`, while pure harmony remains `backing_vocals.wav` at the session root.
- Explicit CUDA, XPU, DirectML, and MPS selections now fail when unavailable; only `device=auto` performs automatic selection. CPU installs and portable CPU builds write `device=cpu` explicitly.
- Karaoke output classification now stops on unknown, ambiguous, or apparently reversed stems instead of guessing by file order or swapping tracks.
- With Karaoke enabled, `accompaniment.wav` always exports MVSep `Back+Instrumental`; disabling harmony in the final mix changes only the mix input, not the public output contract.
- ONNX Runtime now comes from the matching `audio-separator` CPU/GPU/DML extra, avoiding simultaneous CPU and GPU runtime distributions.
- Windows console and redirected project logs now use UTF-8 so mixed Chinese/English output remains readable in CI and release diagnostics.
- Console messages now strictly follow `zh_CN` / `en_US`. English mode uses a bundled offline catalog and fails explicitly on missing translations instead of mixing in Chinese.
- Markdown files under `docs` were removed from version control and are now ignored; release requirements remain in the README and changelog.
- PyTorch, torchvision, and torchaudio are now installed together from one PyTorch index, and `install.py --check` fails when `pip check` finds an inconsistent environment.
- Verified locally on July 12, 2026: all 170 tests passed; a real end-to-end CUDA cover on an RTX 4070 Ti SUPER covered Leap XE, PolarFormer, all three MVSep 9205 models, RoFormer De-Reverb, RVC, and the final mix with no Han characters in the English console log.

## v1.3.0 - 2026-07-04

AI-RVC v1.3.0 重点整理默认翻唱质量路线、模型兼容性和发布打包流程。

### 中文更新说明

#### 亮点

- 默认翻唱路线改为严格 RMVPE，并关闭 `f0_hybrid_mode`；混合 F0 和 fallback F0 不再静默改结果，出问题会直接报错。
- 默认启用更贴近官方 RVC 的推理路径，同时保留本项目的翻唱预处理、清理、源约束和混音流程。
- RVC 权重会根据模型形状识别 v1/v2，不再只依赖可能缺失或写错的 `version` 元数据。
- 转换前会校验 FAISS 索引维度是否匹配当前 RVC 模型。
- 支持导入自定义角色模型：单个 `.pth`、`.pth + .index`，或只包含一个模型的 `.zip`。
- 可下载角色模型扩展到 181 个，并优化了系列和分类展示。
- 清理 Gradio 临时前缀生成的下载文件名，并给每个输出结果单独提供下载按钮。

#### 质量路线

- RoFormer De-Reverb 现在是默认翻唱的严格 VC 预处理路线。
- 源音清理和过渡平滑会尽量保留有效人声，同时压掉尾音回声、低电平杂音和切换尖刺。
- 新增默认质量审计工具，用来阻止隐藏参数覆盖，并要求最终质量结论必须经过听感确认。
- 移除了默认流程里的 UI singing-repair 开关，因为它依赖 F0 fallback 行为。

#### 安装和运行

- `run.py` 会检查必需基础模型，并在默认路线需要时准备 `_official_rvc/` 官方源码树。
- `tools/download_models.py` 支持准备 `_official_rvc/`，源码树不完整会直接报错。
- 依赖检查会严格校验 Gradio、fairseq、audio-separator 等关键版本。
- Hugging Face Hub 限制在 1.0 以下，避免和 Space 运行环境冲突。
- GitHub Release 打包已对齐 Gradio 5.49.1，按 CPU/GPU 版本固定 audio-separator 0.44.1，预下载当时的 RoFormer 默认模型，并把 `_official_rvc/` 打进便携包。

#### UI 和文档

- 翻唱参数现在会校验输入值，不再自动夹取或静默降级。
- 混音预设会同步到音量和混响滑块，同时仍可手动微调。
- README 和 Hugging Face README 已更新当前模型定位、SOTA 边界、严格默认参数和官方 RVC 源码准备方式。
- 本地 agent 指令文件和生成的审计产物已从 Git 跟踪中移除；被 README 引用的文档和界面演示图继续保留。

#### 测试

- 新增或补齐了音频清理保护、RVC 版本识别、官方 adapter 导出、索引维度校验、自定义模型导入、下载文件名清理、UI 下载按钮、默认质量审计策略、安装依赖检查和严格翻唱配置的测试。
- 2026-07-04 本地已验证：`python -m unittest discover -s tests`，116 个测试通过。

#### 下载说明

- Windows 包是 `.7z` 分卷，下载同一版本的所有 `.001`、`.002` 等文件后，用 7-Zip 从 `.001` 解压。
- Linux 包是 `.tar.gz.part*` 分卷，下载同一版本的全部分卷后再合并解压。
- CPU 版不需要显卡；GPU 版面向 NVIDIA CUDA 12.1 环境。

### English Release Notes

AI-RVC v1.3.0 focuses on the default cover quality route, model compatibility, and release hygiene.

#### Highlights

- Changed the default cover route to strict RMVPE with `f0_hybrid_mode=off`; hybrid/fallback F0 paths now fail loudly instead of silently changing behavior.
- Enabled the official-compatible RVC inference path by default while keeping the project cover preprocessing, cleanup, source constraint, and mixing chain.
- Added RVC checkpoint version detection from weight shape, so v1/v2 models with missing or wrong `version` metadata are handled explicitly.
- Validates FAISS index dimensions against the loaded RVC model before conversion.
- Added custom character model import from `.pth`, `.pth + .index`, or a single-model `.zip`.
- Expanded the downloadable character registry to 181 entries and improved series/category grouping.
- Cleaned generated cover download filenames by removing Gradio temp prefixes and adding per-output download buttons.

#### Quality Route

- RoFormer De-Reverb is now the strict VC preprocessing path for default covers.
- Source cleanup and transition smoothing now preserve more active vocal body while still suppressing echo tails, quiet artifacts, and transition spikes.
- Added a default-quality audit tool that blocks hidden parameter overrides and requires listening review for final quality verdicts.
- Removed the UI singing-repair toggle from the default flow because it depended on F0 fallback behavior.

#### Installation And Runtime

- `run.py` now verifies required base models and prepares the vendored official RVC source tree when the default route needs it.
- `tools/download_models.py` can prepare `_official_rvc/` and reports incomplete trees as hard errors.
- Dependency checks now enforce exact versions for Gradio, fairseq, and audio-separator where the project depends on pinned behavior.
- Hugging Face Hub is constrained below 1.0 to stay aligned with the Space runtime.
- GitHub release packaging now builds against Gradio 5.49.1, pins audio-separator 0.44.1 per CPU/GPU variant, preloads the then-current separator defaults, and bundles `_official_rvc/` into portable artifacts.

#### UI And Documentation

- Cover controls now validate values instead of clamping or silently falling back.
- Mix presets update actual mix sliders and still allow manual adjustment.
- README and Hugging Face README now document the current model positioning, SOTA boundaries, strict defaults, and official RVC source preparation.
- Local agent instruction files and generated audit artifacts are removed from Git tracking, while referenced docs and demo images remain tracked.

#### Tests

- Added coverage for audio cleanup guards, RVC version detection, official adapter export, index dimension validation, custom model import, clean output filenames, UI download buttons, default quality audit policy, install requirement checks, and strict cover configuration.
- Verified locally with `python -m unittest discover -s tests` on July 4, 2026: 116 tests passed.

#### Download Notes

- Windows packages are split `.7z` archives. Download every volume for the same edition, then extract from `.001` with 7-Zip.
- Linux packages are split `.tar.gz.part*` archives. Download every part for the same edition, then concatenate and extract.
- CPU builds do not require a GPU. GPU builds target NVIDIA CUDA 12.1 environments.
