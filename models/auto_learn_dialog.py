"""
Auto-Learning Dialog
GUI for automatic AI learning
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QProgressBar, QTextEdit,
    QSpinBox, QCheckBox, QGroupBox, QFrame,
    QTabWidget, QWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor

from models.auto_learner import AutoLearner, LearningProgress, ContinuousLearner
from utils.logger import log


class AutoLearnDialog(QDialog):
    """
    Dialog for auto-learning AI model
    
    Features:
    - One-click learning
    - Progress visualization
    - Learning history
    - Settings customization
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤖 AI自动学习系统")
        self.setMinimumSize(800, 600)
        
        self.learner = AutoLearner()
        self.learner.add_callback(self._on_progress)
        
        self.continuous = ContinuousLearner(self.learner)
        
        self._setup_ui()
        self._setup_timer()
    
    def _setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("🤖 AI自动学习系统")
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #00E5FF; padding: 10px;")
        layout.addWidget(header)
        
        # Tabs
        tabs = QTabWidget()
        
        # Tab 1: Quick Learn
        quick_tab = self._create_quick_tab()
        tabs.addTab(quick_tab, "🚀 快速学习")
        
        # Tab 2: Settings
        settings_tab = self._create_settings_tab()
        tabs.addTab(settings_tab, "⚙️ 设置")
        
        # Tab 3: History
        history_tab = self._create_history_tab()
        tabs.addTab(history_tab, "📊 历史")
        
        layout.addWidget(tabs)
        
        # Progress section
        progress_group = QGroupBox("学习进度")
        progress_layout = QVBoxLayout(progress_group)
        
        self.stage_label = QLabel("状态: 就绪")
        self.stage_label.setFont(QFont("Arial", 12))
        progress_layout.addWidget(self.stage_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        progress_layout.addWidget(self.message_label)
        
        # Stats during learning
        stats_layout = QHBoxLayout()
        
        self.stocks_label = QLabel("发现股票: 0")
        stats_layout.addWidget(self.stocks_label)
        
        self.processed_label = QLabel("已处理: 0")
        stats_layout.addWidget(self.processed_label)
        
        self.accuracy_label = QLabel("准确率: --")
        stats_layout.addWidget(self.accuracy_label)
        
        progress_layout.addLayout(stats_layout)
        
        layout.addWidget(progress_group)
        
        # Log
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("🚀 开始学习")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00E5FF, stop:1 #00BCD4);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00BCD4, stop:1 #0097A7);
            }
            QPushButton:disabled {
                background: #333;
                color: #666;
            }
        """)
        self.start_btn.clicked.connect(self._start_learning)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹️ 停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_learning)
        btn_layout.addWidget(self.stop_btn)
        
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)
        
        layout.addLayout(btn_layout)
        
        self._apply_style()
    
    def _create_quick_tab(self) -> QWidget:
        """Create quick learn tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Quick description
        desc = QLabel("""
        <h3>🚀 一键自动学习</h3>
        <p>点击下方按钮，AI将自动执行以下操作：</p>
        <ol>
            <li>🔍 <b>搜索互联网</b> - 寻找热门股票、涨跌榜、机构推荐</li>
            <li>📥 <b>下载数据</b> - 获取最新股票历史数据</li>
            <li>🧮 <b>计算特征</b> - 生成80+技术指标</li>
            <li>🧠 <b>训练模型</b> - 训练6个神经网络</li>
            <li>✅ <b>保存模型</b> - 保存最佳模型供使用</li>
        </ol>
        <p><i>整个过程约需30-60分钟</i></p>
        """)
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Quick settings
        quick_settings = QGroupBox("快速设置")
        qs_layout = QGridLayout(quick_settings)
        
        qs_layout.addWidget(QLabel("训练轮数:"), 0, 0)
        self.quick_epochs = QSpinBox()
        self.quick_epochs.setRange(50, 300)
        self.quick_epochs.setValue(100)
        qs_layout.addWidget(self.quick_epochs, 0, 1)
        
        qs_layout.addWidget(QLabel("最大股票数:"), 0, 2)
        self.quick_stocks = QSpinBox()
        self.quick_stocks.setRange(20, 200)
        self.quick_stocks.setValue(80)
        qs_layout.addWidget(self.quick_stocks, 0, 3)
        
        self.quick_search = QCheckBox("自动搜索互联网")
        self.quick_search.setChecked(True)
        qs_layout.addWidget(self.quick_search, 1, 0, 1, 2)
        
        self.quick_incremental = QCheckBox("增量学习（保留旧知识）")
        self.quick_incremental.setChecked(True)
        qs_layout.addWidget(self.quick_incremental, 1, 2, 1, 2)
        
        layout.addWidget(quick_settings)
        
        layout.addStretch()
        
        return widget
    
    def _create_settings_tab(self) -> QWidget:
        """Create settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Training settings
        train_group = QGroupBox("训练参数")
        train_layout = QGridLayout(train_group)
        
        train_layout.addWidget(QLabel("训练轮数 (Epochs):"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 500)
        self.epochs_spin.setValue(100)
        train_layout.addWidget(self.epochs_spin, 0, 1)
        
        train_layout.addWidget(QLabel("批次大小 (Batch):"), 0, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(16, 256)
        self.batch_spin.setValue(64)
        train_layout.addWidget(self.batch_spin, 0, 3)
        
        train_layout.addWidget(QLabel("学习率:"), 1, 0)
        self.lr_label = QLabel("0.0005")
        train_layout.addWidget(self.lr_label, 1, 1)
        
        layout.addWidget(train_group)
        
        # Search settings
        search_group = QGroupBox("搜索设置")
        search_layout = QVBoxLayout(search_group)
        
        self.search_gainers = QCheckBox("搜索涨幅榜")
        self.search_gainers.setChecked(True)
        search_layout.addWidget(self.search_gainers)
        
        self.search_losers = QCheckBox("搜索跌幅榜")
        self.search_losers.setChecked(True)
        search_layout.addWidget(self.search_losers)
        
        self.search_volume = QCheckBox("搜索成交额榜")
        self.search_volume.setChecked(True)
        search_layout.addWidget(self.search_volume)
        
        self.search_hot = QCheckBox("搜索热门股票")
        self.search_hot.setChecked(True)
        search_layout.addWidget(self.search_hot)
        
        self.search_analyst = QCheckBox("搜索机构推荐")
        self.search_analyst.setChecked(True)
        search_layout.addWidget(self.search_analyst)
        
        layout.addWidget(search_group)
        
        # Continuous learning
        continuous_group = QGroupBox("持续学习")
        cont_layout = QVBoxLayout(continuous_group)
        
        self.cont_daily = QCheckBox("每日自动更新数据")
        self.cont_daily.setChecked(True)
        cont_layout.addWidget(self.cont_daily)
        
        self.cont_weekly = QCheckBox("每周自动重新训练")
        self.cont_weekly.setChecked(True)
        cont_layout.addWidget(self.cont_weekly)
        
        self.cont_trades = QCheckBox("从交易结果中学习")
        self.cont_trades.setChecked(True)
        cont_layout.addWidget(self.cont_trades)
        
        layout.addWidget(continuous_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_history_tab(self) -> QWidget:
        """Create history tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Stats summary
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background: #1a1a3e;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        stats_layout = QHBoxLayout(stats_frame)
        
        stats = self.learner.get_learning_stats()
        
        for title, value in [
            ("学习次数", f"{stats['sessions_count']}"),
            ("最佳准确率", f"{stats['best_accuracy']*100:.1f}%"),
            ("学习股票数", f"{stats['total_stocks']}"),
        ]:
            container = QWidget()
            cont_layout = QVBoxLayout(container)
            
            title_label = QLabel(title)
            title_label.setStyleSheet("color: #888; font-size: 12px;")
            
            value_label = QLabel(value)
            value_label.setStyleSheet("color: #00E5FF; font-size: 20px; font-weight: bold;")
            
            cont_layout.addWidget(title_label)
            cont_layout.addWidget(value_label)
            
            stats_layout.addWidget(container)
        
        layout.addWidget(stats_frame)
        
        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            "时间", "股票数", "样本数", "轮数", "准确率", "用时"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        
        self._load_history_table()
        
        layout.addWidget(self.history_table)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.clicked.connect(self._load_history_table)
        layout.addWidget(refresh_btn)
        
        return widget
    
    def _load_history_table(self):
        """Load history into table"""
        stats = self.learner.get_learning_stats()
        sessions = self.learner.history.get('sessions', [])
        
        self.history_table.setRowCount(len(sessions))
        
        for i, session in enumerate(reversed(sessions)):
            self.history_table.setItem(i, 0, QTableWidgetItem(
                session.get('timestamp', '')[:16]
            ))
            self.history_table.setItem(i, 1, QTableWidgetItem(
                str(session.get('stocks_used', 0))
            ))
            self.history_table.setItem(i, 2, QTableWidgetItem(
                str(session.get('samples', 0))
            ))
            self.history_table.setItem(i, 3, QTableWidgetItem(
                str(session.get('epochs', 0))
            ))
            
            acc = session.get('test_accuracy', 0) * 100
            acc_item = QTableWidgetItem(f"{acc:.1f}%")
            acc_item.setForeground(QColor("#4CAF50" if acc > 50 else "#FF5252"))
            self.history_table.setItem(i, 4, acc_item)
            
            self.history_table.setItem(i, 5, QTableWidgetItem(
                f"{session.get('duration_minutes', 0):.1f}分钟"
            ))
    
    def _setup_timer(self):
        """Setup update timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_ui)
        self.timer.start(500)
    
    def _apply_style(self):
        """Apply dialog style"""
        self.setStyleSheet("""
            QDialog {
                background: #0a0a1a;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2a2a5a;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 12px;
                color: #00E5FF;
                background: #0f0f2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
            QLabel {
                color: #ddd;
            }
            QSpinBox, QCheckBox {
                color: #fff;
            }
            QProgressBar {
                border: none;
                background: #1a1a3e;
                border-radius: 5px;
                text-align: center;
                color: #fff;
                height: 25px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00E5FF, stop:1 #00BCD4);
                border-radius: 5px;
            }
            QTextEdit {
                background: #0a0a1a;
                color: #0f0;
                border: 1px solid #2a2a5a;
                border-radius: 5px;
            }
            QTableWidget {
                background: #1a1a3e;
                color: #fff;
                border: none;
                gridline-color: #2a2a5a;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background: #2a2a5a;
                color: #00E5FF;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
            QPushButton {
                background: #3a3a7a;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #4a4a9a;
            }
            QPushButton:disabled {
                background: #222;
                color: #555;
            }
            QTabWidget::pane {
                border: 2px solid #2a2a5a;
                background: #0a0a1a;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #1a1a3e;
                color: #888;
                padding: 10px 20px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #2a2a5a;
                color: #00E5FF;
            }
        """)
    
    def _start_learning(self):
        """Start auto learning"""
        reply = QMessageBox.question(
            self,
            "开始学习",
            "AI将自动搜索互联网并训练模型。\n\n"
            "这可能需要30-60分钟。\n\n"
            "是否开始？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.log_text.clear()
        self._log("🚀 开始自动学习...")
        
        self.learner.start_learning(
            auto_search=self.quick_search.isChecked(),
            max_stocks=self.quick_stocks.value(),
            epochs=self.quick_epochs.value(),
            incremental=self.quick_incremental.isChecked()
        )
    
    def _stop_learning(self):
        """Stop learning"""
        self.learner.stop_learning()
        self._log("⏹️ 学习已停止")
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def _on_progress(self, progress: LearningProgress):
        """Handle progress update (called from thread)"""
        pass  # Update happens via timer
    
    def _update_ui(self):
        """Update UI from progress"""
        p = self.learner.progress
        
        # Update stage
        stage_text = {
            'idle': '就绪',
            'searching': '🔍 搜索中',
            'downloading': '📥 下载中',
            'preparing': '🧮 准备数据',
            'training': '🧠 训练中',
            'evaluating': '📊 评估中',
            'complete': '✅ 完成',
            'error': '❌ 错误'
        }
        self.stage_label.setText(f"状态: {stage_text.get(p.stage, p.stage)}")
        
        # Update progress bar
        self.progress_bar.setValue(int(p.progress))
        
        # Update message
        self.message_label.setText(p.message)
        
        # Update stats
        self.stocks_label.setText(f"发现股票: {p.stocks_found}")
        self.processed_label.setText(f"已处理: {p.stocks_processed}")
        
        if p.training_accuracy > 0:
            self.accuracy_label.setText(f"准确率: {p.training_accuracy:.1%}")
        
        # Update buttons
        if not p.is_running:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
        
        # Log errors
        for error in p.errors:
            if error not in self.log_text.toPlainText():
                self._log(f"⚠️ {error}")
        
        # Completion
        if p.stage == 'complete':
            self._log(f"✅ 训练完成！最终准确率: {p.training_accuracy:.1%}")
            self._load_history_table()
    
    def _log(self, message: str):
        """Add log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")


def show_auto_learn_dialog(parent=None):
    """Show auto-learn dialog"""
    dialog = AutoLearnDialog(parent)
    dialog.exec()