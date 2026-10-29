"""Native PySide6 desktop frontend. No QWebEngine, Gradio, or HTTP server."""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QThread, QTimer, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSlider, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from desktop.cover import CoverRequest, run_cover
from desktop.realtime import RealtimeRVC, RealtimeSettings, audio_devices
from tools.svc_model_registry import (
    KNOWN_ENGINES, engine_health, load_registry, save_engine_config,
    save_shared_config,
)

ROOT = Path(__file__).resolve().parents[1]
STYLES = """
QWidget { background: #11141c; color: #edf0f6; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; font-size: 13px; }
QMainWindow, QTabWidget::pane { background: #11141c; }
QLabel#hero { font-size: 25px; font-weight: 700; color: #f8f9fc; }
QLabel#subtitle { color: #9ba9c2; font-size: 12px; }
QGroupBox { border: 1px solid #303a4c; border-radius: 11px; margin-top: 18px; padding: 14px 12px 10px; background: #191e29; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 13px; padding: 0 6px; color: #cfd9ed; }
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit { border: 1px solid #3b4558; border-radius: 7px; padding: 7px; background: #222938; color: #f3f5fa; selection-background-color: #4a6cc4; }
QPushButton { background: #26344d; border: 1px solid #425575; border-radius: 8px; padding: 8px 14px; font-weight: 600; }
QPushButton:hover { background: #334768; }
QPushButton:disabled { color: #788396; background: #202533; }
QPushButton#primary { background: #466bd3; color: white; border: 1px solid #678cf2; }
QPushButton#primary:hover { background: #557dea; }
QPushButton#stop { background: #60303b; border: 1px solid #a65b70; }
QTabBar::tab { background: #202635; padding: 12px 24px; border-radius: 6px; margin-right: 5px; }
QTabBar::tab:selected { background: #3456a3; }
QCheckBox { spacing: 10px; }
QProgressBar { border: 1px solid #3a485f; border-radius: 7px; background: #222938; min-height: 8px; }
QProgressBar::chunk { background: #4e7ae5; border-radius: 7px; }
"""


class CoverWorker(QObject):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, request: CoverRequest):
        super().__init__()
        self.request = request

    @Slot()
    def run(self):
        try:
            result = run_cover(self.request, progress=self.progress.emit)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()


def _path_entry(label: str, filter_spec: str = "All files (*)", directory: bool = False):
    """Return a form-ready field with native file/directory chooser."""
    field = QLineEdit()
    field.setPlaceholderText(label)
    button = QPushButton("浏览…")
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(field, 1)
    layout.addWidget(button)

    def select():
        if directory:
            result = QFileDialog.getExistingDirectory(container, label, field.text() or str(ROOT))
        else:
            result, _ = QFileDialog.getOpenFileName(container, label, field.text() or str(ROOT), filter_spec)
        if result:
            field.setText(result)

    button.clicked.connect(select)
    return field, container


class DesktopWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-RVC Studio · Desktop")
        self.resize(1060, 780)
        self.setMinimumSize(830, 620)
        self.settings = QSettings("AI-RVC", "AI-RVC-Studio")
        self.live: RealtimeRVC | None = None
        self.cover_thread: QThread | None = None
        self.cover_worker: CoverWorker | None = None
        self._last_live_state = ""
        self._build()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll_live)
        self.timer.start(300)

    def _build(self):
        central = QWidget()
        content = QVBoxLayout(central)
        content.setContentsMargins(22, 18, 22, 18)
        content.setSpacing(12)
        hero = QLabel("AI-RVC  STUDIO")
        hero.setObjectName("hero")
        content.addWidget(hero)
        subtitle = QLabel("原生桌面 AI 翻唱 · 麦克风实时变声 · 本地模型统一管理")
        subtitle.setObjectName("subtitle")
        content.addWidget(subtitle)
        tabs = QTabWidget()
        content.addWidget(tabs, 1)
        tabs.addTab(self._scroll(self._cover_tab()), "🎵 AI 翻唱")
        tabs.addTab(self._scroll(self._live_tab()), "🎙 实时变声")
        tabs.addTab(self._scroll(self._models_tab()), "🧩 模型管理")
        self.setCentralWidget(central)
        self.statusBar().showMessage("本地桌面模式 · 不启动 Web 服务")

    def _scroll(self, child: QWidget) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(child)
        return scroll

    def _cover_tab(self):
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(12, 12, 12, 16)
        layout.setSpacing(13)
        group = QGroupBox("音频与目标声线")
        form = QFormLayout(group)
        self.cover_input, widget = _path_entry("歌曲或干声音频", "Audio (*.wav *.mp3 *.flac *.ogg);;All (*)")
        form.addRow("输入音频", widget)
        self.cover_engine = QComboBox()
        for key, spec in KNOWN_ENGINES.items():
            if spec["supported"]:
                self.cover_engine.addItem(spec["name"], key)
        form.addRow("转换引擎", self.cover_engine)
        self.cover_mode = QComboBox()
        self.cover_mode.addItem("完整歌曲（自动分离）", "song")
        self.cover_mode.addItem("已分离主唱 / 干声", "vocals")
        form.addRow("输入模式", self.cover_mode)
        self.cover_model, widget = _path_entry("RVC .pth 声线模型", "RVC checkpoints (*.pth);;All (*)")
        self.cover_model.setText(self.settings.value("rvc_model", "", str))
        form.addRow("RVC .pth", widget)
        self.cover_index, widget = _path_entry("可选 .index（留空禁用检索）", "FAISS index (*.index);;All (*)")
        form.addRow("RVC .index", widget)
        self.cover_reference, widget = _path_entry("目标音色参考（零样本 SVC）", "Audio (*.wav *.mp3 *.flac);;All (*)")
        form.addRow("参考声音", widget)
        layout.addWidget(group)

        controls = QGroupBox("转换设置")
        opts = QFormLayout(controls)
        self.cover_pitch = QSpinBox()
        self.cover_pitch.setRange(-12, 12)
        self.cover_pitch.setValue(0)
        self.cover_pitch.setSuffix(" 半音")
        opts.addRow("音高偏移", self.cover_pitch)
        self.cover_steps = QSpinBox()
        self.cover_steps.setRange(1, 150)
        self.cover_steps.setValue(35)
        opts.addRow("扩散步数", self.cover_steps)
        self.cover_karaoke = QCheckBox("保留原曲和声（Karaoke 分离）")
        self.cover_karaoke.setChecked(True)
        opts.addRow("", self.cover_karaoke)
        self.cover_dereverb = QCheckBox("主唱去混响（干声优先时建议关闭）")
        opts.addRow("", self.cover_dereverb)
        self.cover_output, widget = _path_entry("输出目录", directory=True)
        self.cover_output.setText(str(ROOT / "outputs"))
        opts.addRow("输出路径", widget)
        layout.addWidget(controls)

        self.cover_start = QPushButton("▶  开始生成翻唱")
        self.cover_start.setObjectName("primary")
        self.cover_start.clicked.connect(self._start_cover)
        layout.addWidget(self.cover_start)
        self.cover_log = QPlainTextEdit()
        self.cover_log.setReadOnly(True)
        self.cover_log.setPlaceholderText("进度与输出文件会显示在这里…")
        self.cover_log.setMinimumHeight(130)
        layout.addWidget(self.cover_log)
        self.cover_engine.currentIndexChanged.connect(self._update_cover_engine)
        self._update_cover_engine()
        layout.addStretch(1)
        return pane

    def _update_cover_engine(self):
        rvc = self.cover_engine.currentData() == "rvc"
        self.cover_model.setEnabled(rvc)
        self.cover_index.setEnabled(rvc)
        self.cover_reference.setEnabled(not rvc)
        self.cover_steps.setEnabled(not rvc)

    def _start_cover(self):
        if self.cover_thread and self.cover_thread.isRunning():
            return
        request = CoverRequest(
            audio_file=self.cover_input.text().strip(),
            engine=self.cover_engine.currentData(),
            input_mode=self.cover_mode.currentData(),
            rvc_model=self.cover_model.text().strip(),
            rvc_index=self.cover_index.text().strip(),
            reference_audio=self.cover_reference.text().strip(),
            pitch_shift=self.cover_pitch.value(),
            karaoke=self.cover_karaoke.isChecked(),
            dereverb=self.cover_dereverb.isChecked(),
            diffusion_steps=self.cover_steps.value(),
            output_dir=self.cover_output.text().strip(),
        )
        if request.engine == "rvc" and request.rvc_model:
            self.settings.setValue("rvc_model", request.rvc_model)
        self.cover_start.setEnabled(False)
        self.cover_log.clear()
        self.cover_log.appendPlainText(f"启动：{request.engine} · {request.input_mode}")
        thread = QThread(self)
        worker = CoverWorker(request)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.cover_log.appendPlainText)
        worker.failed.connect(lambda message: self.cover_log.appendPlainText("❌ " + message))
        worker.completed.connect(self._cover_done)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._cover_thread_done)
        self.cover_thread, self.cover_worker = thread, worker
        thread.start()

    def _cover_done(self, result):
        self.cover_log.appendPlainText("✅ 完成")
        for key in ("cover", "converted_vocals", "vocals", "accompaniment", "all_files_dir"):
            if result.get(key):
                self.cover_log.appendPlainText(f"{key}: {result[key]}")

    def _cover_thread_done(self):
        self.cover_start.setEnabled(True)
        if self.cover_thread:
            self.cover_thread.deleteLater()
        self.cover_thread = None
        self.cover_worker = None

    def _live_tab(self):
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(12, 12, 12, 16)
        layout.setSpacing(14)
        devices = QGroupBox("实时音频输入 / 输出")
        form = QFormLayout(devices)
        self.live_model, field = _path_entry("选择本地 .pth RVC 模型", "RVC checkpoints (*.pth);;All (*)")
        self.live_model.setText(self.settings.value("rvc_model", "", str))
        form.addRow("声线模型", field)
        self.live_index, field = _path_entry(".index 可选，默认不使用检索", "FAISS index (*.index);;All (*)")
        form.addRow("索引文件", field)
        self.input_device = QComboBox()
        self.output_device = QComboBox()
        form.addRow("麦克风设备", self.input_device)
        form.addRow("输出设备 / 虚拟声卡", self.output_device)
        refresh = QPushButton("↻  刷新声卡")
        refresh.clicked.connect(self._refresh_devices)
        form.addRow("", refresh)
        self._refresh_devices()
        layout.addWidget(devices)

        params = QGroupBox("实时参数")
        fields = QFormLayout(params)
        self.live_f0 = QComboBox()
        for name in ("rmvpe", "fcpe", "pm"):
            self.live_f0.addItem(name.upper(), name)
        fields.addRow("F0 算法", self.live_f0)
        self.live_rate = QComboBox()
        for value in (32000, 40000, 44100, 48000):
            self.live_rate.addItem(f"{value:,} Hz", value)
        self.live_rate.setCurrentIndex(3)
        fields.addRow("声卡采样率", self.live_rate)
        self.live_index_rate = QSpinBox()
        self.live_index_rate.setRange(0, 100)
        self.live_index_rate.setValue(0)
        self.live_index_rate.setSuffix(" %")
        fields.addRow("索引检索比例", self.live_index_rate)
        self.live_pitch = QSpinBox()
        self.live_pitch.setRange(-12, 12)
        self.live_pitch.setSuffix(" 半音")
        fields.addRow("音高偏移", self.live_pitch)
        self.live_block = QComboBox()
        for ms in (200, 250, 300, 400):
            self.live_block.addItem(f"{ms} ms", ms)
        self.live_block.setCurrentIndex(1)
        fields.addRow("推理块时长", self.live_block)
        self.live_context = QSpinBox()
        self.live_context.setRange(500, 3000)
        self.live_context.setSingleStep(100)
        self.live_context.setValue(1200)
        self.live_context.setSuffix(" ms")
        fields.addRow("声音上下文", self.live_context)
        self.live_input_gain = QSlider(Qt.Orientation.Horizontal)
        self.live_input_gain.setRange(0, 200)
        self.live_input_gain.setValue(100)
        fields.addRow("麦克风增益", self.live_input_gain)
        self.live_output_gain = QSlider(Qt.Orientation.Horizontal)
        self.live_output_gain.setRange(0, 150)
        self.live_output_gain.setValue(80)
        fields.addRow("输出增益", self.live_output_gain)
        layout.addWidget(params)
        buttons = QHBoxLayout()
        self.live_start = QPushButton("🎙 开始变声")
        self.live_start.setObjectName("primary")
        self.live_stop = QPushButton("■ 停止")
        self.live_stop.setObjectName("stop")
        self.live_stop.setEnabled(False)
        self.live_start.clicked.connect(self._start_live)
        self.live_stop.clicked.connect(self._stop_live)
        buttons.addWidget(self.live_start, 1)
        buttons.addWidget(self.live_stop)
        layout.addLayout(buttons)
        self.live_status = QLabel("未启动 · 麦克风不会在此状态下被采集")
        self.live_status.setWordWrap(True)
        layout.addWidget(self.live_status)
        self.live_meter = QProgressBar()
        self.live_meter.setRange(0, 400)
        self.live_meter.setValue(0)
        self.live_meter.setFormat("最近一次推理 %v ms")
        layout.addWidget(self.live_meter)
        layout.addWidget(QLabel("提示：请使用耳机监听防止啸叫；输出选虚拟声卡即可提供给其他应用。\\n"
                               "此模式为有缓冲的实时变声，延迟取决于设备、音高模型与 GPU。"))
        layout.addStretch(1)
        return pane

    def _refresh_devices(self):
        self.input_device.clear()
        self.output_device.clear()
        try:
            sources, sinks = audio_devices()
            for title, index in sources:
                self.input_device.addItem(title, index)
            for title, index in sinks:
                self.output_device.addItem(title, index)
            if not sources or not sinks:
                self.live_status.setText("找不到可用麦克风或输出设备") if hasattr(self, "live_status") else None
        except Exception as exc:
            if hasattr(self, "live_status"):
                self.live_status.setText(f"音频设备枚举失败：{exc}")

    def _start_live(self):
        if self.live and self.live.running:
            return
        if self.input_device.currentData() is None or self.output_device.currentData() is None:
            QMessageBox.warning(self, "音频设备", "请先选择有效的麦克风及输出设备")
            return
        settings = RealtimeSettings(
            model_path=self.live_model.text().strip(),
            index_path=self.live_index.text().strip(),
            input_device=self.input_device.currentData(),
            output_device=self.output_device.currentData(),
            block_ms=self.live_block.currentData(),
            sample_rate=self.live_rate.currentData(),
            index_rate=self.live_index_rate.value() / 100,
            context_ms=self.live_context.value(),
            pitch_shift=self.live_pitch.value(),
            f0_method=self.live_f0.currentData(),
            input_gain=self.live_input_gain.value() / 100,
            output_gain=self.live_output_gain.value() / 100,
            device=(os.environ.get("AI_RVC_DEVICE") or json.loads((ROOT / "configs" / "config.json").read_text(encoding="utf-8")).get("device", "cuda")),
        )
        try:
            self.live = RealtimeRVC(settings)
            self.live.start()
        except Exception as exc:
            QMessageBox.critical(self, "无法启动实时变声", str(exc))
            return
        self.live_start.setEnabled(False)
        self.live_stop.setEnabled(True)
        self.live_status.setText("正在准备 HuBERT / RVC / F0 模型…")
        self.settings.setValue("rvc_model", settings.model_path)

    def _stop_live(self):
        if self.live:
            self.live.stop()
        self.live_status.setText("正在停止实时引擎并释放模型…")

    def _poll_live(self):
        if not self.live:
            return
        state = self.live.state
        if state == "active":
            self.live_meter.setValue(min(400, int(self.live.last_runtime_ms)))
            self.live_status.setText(
                f"变声运行中 · 推理 {self.live.last_runtime_ms:.0f}ms / "
                f"块 {self.live.settings.block_ms}ms · "
                f"缓冲不足 {self.live.underruns} 次 · 溢出 {self.live.overruns} 次"
            )
        elif not self.live.running and state in ("error", "stopped"):
            self.live_start.setEnabled(True)
            self.live_stop.setEnabled(False)
            if self._last_live_state != state:
                self.live_status.setText(
                    f"❌ {self.live.last_error}" if state == "error" else "已停止并释放实时资源"
                )
        self._last_live_state = state

    def _models_tab(self):
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(12, 12, 12, 16)
        layout.setSpacing(13)
        shared = QGroupBox("共享环境：兼容依赖复用")
        form = QFormLayout(shared)
        self.shared_python, widget = _path_entry("兼容的通用 Python 可执行文件")
        form.addRow("共享 Python", widget)
        self.shared_cache, widget = _path_entry("共享权重缓存", directory=True)
        form.addRow("模型缓存", widget)
        shared_button = QPushButton("保存共享环境")
        shared_button.clicked.connect(self._save_shared)
        form.addRow("", shared_button)
        layout.addWidget(shared)

        model = QGroupBox("SVC 引擎安装配置")
        form = QFormLayout(model)
        self.manage_engine = QComboBox()
        for key, meta in KNOWN_ENGINES.items():
            if key != "rvc" and meta["supported"]:
                self.manage_engine.addItem(meta["name"], key)
        form.addRow("模型类型", self.manage_engine)
        self.manage_repo, widget = _path_entry("本地项目目录", directory=True)
        form.addRow("仓库目录", widget)
        self.manage_python, widget = _path_entry("留空使用共享 Python")
        form.addRow("专用 Python", widget)
        self.manage_checkpoint, widget = _path_entry("模型权重（Seed 可留空）")
        form.addRow("权重文件", widget)
        self.manage_config, widget = _path_entry("YAML 配置")
        form.addRow("模型配置", widget)
        save = QPushButton("保存当前模型设置")
        save.setObjectName("primary")
        save.clicked.connect(self._save_engine)
        form.addRow("", save)
        self.manage_engine.currentIndexChanged.connect(self._show_engine)
        layout.addWidget(model)
        rvc_library_group = QGroupBox("已下载 RVC 声线库")
        library_form = QFormLayout(rvc_library_group)
        self.rvc_library = QComboBox()
        library_form.addRow("本地 .pth 声线", self.rvc_library)
        library_actions = QHBoxLayout()
        library_reload = QPushButton("↻ 扫描模型")
        library_to_cover = QPushButton("用于 AI 翻唱")
        library_to_live = QPushButton("用于实时变声")
        library_reload.clicked.connect(self._refresh_rvc_library)
        library_to_cover.clicked.connect(lambda: self._select_rvc_library_model(self.cover_model))
        library_to_live.clicked.connect(lambda: self._select_rvc_library_model(self.live_model))
        library_actions.addWidget(library_reload)
        library_actions.addWidget(library_to_cover)
        library_actions.addWidget(library_to_live)
        library_form.addRow("", library_actions)
        layout.addWidget(rvc_library_group)

        status = QGroupBox("本地模型检测")
        status_layout = QVBoxLayout(status)
        self.models_log = QPlainTextEdit()
        self.models_log.setReadOnly(True)
        self.models_log.setMinimumHeight(130)
        status_layout.addWidget(self.models_log)
        refresh = QPushButton("↻ 更新模型检测")
        refresh.clicked.connect(self._refresh_models)
        status_layout.addWidget(refresh)
        layout.addWidget(status)
        layout.addStretch(1)
        data = load_registry()
        self.shared_python.setText(data.get("_shared", {}).get("engine_python", ""))
        self.shared_cache.setText(data.get("_shared", {}).get("cache_directory", ""))
        self._show_engine()
        self._refresh_models()
        self._refresh_rvc_library()
        return pane

    def _refresh_rvc_library(self):
        self.rvc_library.clear()
        root = ROOT / "assets" / "weights"
        if not root.is_dir():
            return
        for model in sorted(root.rglob("*.pth"), key=lambda path: str(path).lower()):
            # List existing local model files; never load big checkpoints in the GUI.
            self.rvc_library.addItem(str(model.relative_to(root)), str(model.resolve()))

    def _select_rvc_library_model(self, field):
        value = self.rvc_library.currentData()
        if value:
            field.setText(value)
            self.settings.setValue("rvc_model", value)
            self.statusBar().showMessage("已选本地 RVC 声线：" + Path(value).stem, 5000)

    def _show_engine(self):
        config = load_registry().get(self.manage_engine.currentData(), {})
        for field, key in (
            (self.manage_repo, "engine_repository"), (self.manage_python, "engine_python"),
            (self.manage_checkpoint, "checkpoint"), (self.manage_config, "model_config"),
        ):
            field.setText(config.get(key, ""))

    def _refresh_models(self):
        lines = []
        for key, meta in KNOWN_ENGINES.items():
            available, reason = engine_health(key)
            lines.append(f'{"●" if available else "○"}  {meta["name"]}: {reason}')
        self.models_log.setPlainText("\n".join(lines))

    def _save_shared(self):
        try:
            python = self.shared_python.text().strip()
            if python and not Path(python).is_file():
                raise ValueError("共享 Python 文件不存在")
            save_shared_config({
                "engine_python": python,
                "cache_directory": self.shared_cache.text().strip(),
            })
            self._refresh_models()
            QMessageBox.information(self, "共享环境", "已保存共享路径；依赖是否兼容仍需要各模型实测。")
        except Exception as exc:
            QMessageBox.critical(self, "配置错误", str(exc))

    def _save_engine(self):
        try:
            save_engine_config(self.manage_engine.currentData(), {
                "engine_repository": self.manage_repo.text().strip(),
                "engine_python": self.manage_python.text().strip(),
                "checkpoint": self.manage_checkpoint.text().strip(),
                "model_config": self.manage_config.text().strip(),
            })
            self._refresh_models()
            QMessageBox.information(self, "模型管理", "本地配置保存成功")
        except Exception as exc:
            QMessageBox.critical(self, "配置错误", str(exc))

    def closeEvent(self, event):
        if self.cover_thread and self.cover_thread.isRunning():
            QMessageBox.warning(self, "任务运行中", "请在离线翻唱任务结束后关闭程序，防止生成中断。")
            event.ignore()
            return
        if self.live:
            self.live.stop()
            if not self.live.join(1.5):
                QMessageBox.warning(self, "实时引擎运行中", "实时推理尚未退出，请稍后再次关闭。")
                event.ignore()
                return
        event.accept()


def launch() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AI-RVC Studio")
    app.setStyleSheet(STYLES)
    window = DesktopWindow()
    window.show()
    return app.exec()
