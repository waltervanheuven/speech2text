""" settings.py """

# Copyright © 2023-2025 Walter van Heuven

import platform
import logging
import validators
from PyQt6.QtWidgets import QMessageBox, QWidget, QDialog, QPushButton, QLabel, QCheckBox
from PyQt6.QtWidgets import QComboBox, QDialogButtonBox, QGroupBox, QRadioButton
from PyQt6.QtWidgets import QLineEdit, QHBoxLayout, QGridLayout, QVBoxLayout
from PyQt6.QtCore import Qt
import utils as app_utils

class SettingsDialog(QDialog):
    def __init__(self, mainWindow) -> None:
        super().__init__()

        self.setWindowTitle("Settings")
        self.mainWindow = mainWindow

        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        gb = QGroupBox()
        gb.setTitle("Select Automatic Speech Recognition (ASR) Engine:")

        r:int = 0
        grid = QGridLayout()
        grid.setSpacing(0)

        # ---------------------------------------------------------------------------------------------------------
        # openai-whisper
        self.radio_openai_whisper = QRadioButton("Whisper")
        self.radio_openai_whisper.setToolTip("Use OpenAI's original Whisper implementation. Requires FFmpeg")
        self.radio_openai_whisper.clicked.connect(self.apply_whisper)
        self.radio_openai_whisper.setMaximumWidth(200)
        grid.addWidget(self.radio_openai_whisper, r, 0, 1, 2)

        # openai-whisper model
        w1a = QWidget()
        box_layout_openai = QHBoxLayout()
        self.label_openai_model = QLabel("Model:")
        self.combobox_openai_model = QComboBox()
        self.combobox_openai_model.setFixedWidth(150)
        self.combobox_openai_model.setToolTip("Select model. Please note that the medium model requires ~3 GB of VRAM and the large model requires ~5 GB")
        box_layout_openai.addWidget(self.label_openai_model, Qt.AlignmentFlag.AlignLeft)
        box_layout_openai.addWidget(self.combobox_openai_model, Qt.AlignmentFlag.AlignLeft)

        w1a.setLayout(box_layout_openai)
        grid.addWidget(w1a, r, 1, 1, 1, Qt.AlignmentFlag.AlignLeft)
        r += 1

        if mainWindow.whisper_engine is None:
            self.radio_openai_whisper.setEnabled(False)
            self.combobox_openai_model.setEnabled(False)

        # mlx-whisper
        # only show on Darwin arm
        if mainWindow.mlx_whisper_engine is not None:
            # whisper MLX
            w1b = QWidget()
            self.radio_mlx_whisper = QRadioButton("MLX Whisper")
            self.radio_mlx_whisper.setToolTip("Use MLX Whisper implementation (only available on Apple silicon Macs). Requires FFmpeg")
            self.radio_mlx_whisper.clicked.connect(self.apply_mlx_whisper)
            self.radio_mlx_whisper.setMaximumWidth(200)
            grid.addWidget(self.radio_mlx_whisper, r, 0, 1, 2)

            # whisper MLX model
            w1b = QWidget()
            box_layout_mlx_whisper = QHBoxLayout()
            self.label_mlx_model = QLabel("Model:")
            self.combobox_mlx_model = QComboBox()
            self.combobox_mlx_model.setFixedWidth(150)
            self.combobox_mlx_model.setToolTip("Select model. Please note that the medium model requires ~3 GB of VRAM and the large model requires ~5 GB")
            box_layout_mlx_whisper.addWidget(self.label_mlx_model, Qt.AlignmentFlag.AlignLeft)
            box_layout_mlx_whisper.addWidget(self.combobox_mlx_model, Qt.AlignmentFlag.AlignLeft)

            w1b.setLayout(box_layout_mlx_whisper)
            grid.addWidget(w1b, r, 1, 1, 1, Qt.AlignmentFlag.AlignLeft)
            r += 1

        # ---------------------------------------------------------------------------------------------------------
        # whisper.cpp
        self.radio_whisper_cpp = QRadioButton("Whisper.cpp")
        self.radio_whisper_cpp.setToolTip("Use whisper.cpp")
        self.radio_whisper_cpp.clicked.connect(self.apply_whisper_cpp)
        grid.addWidget(self.radio_whisper_cpp, r, 0, 1, 2)

        # whisper.cpp model
        w2 = QWidget()
        box_layout_whisper_cpp = QHBoxLayout()
        self.label_whisper_cpp_model = QLabel("Model:")
        self.combobox_whisper_cpp = QComboBox()
        self.combobox_whisper_cpp.setFixedWidth(150)
        box_layout_whisper_cpp.addWidget(self.label_whisper_cpp_model, Qt.AlignmentFlag.AlignLeft)
        box_layout_whisper_cpp.addWidget(self.combobox_whisper_cpp, Qt.AlignmentFlag.AlignLeft)

        if platform.system() == "Darwin":
            w2.setLayout(box_layout_whisper_cpp)
            grid.addWidget(w2, r, 1, 1, 1, Qt.AlignmentFlag.AlignLeft)
            r += 1

            w2b = QWidget()
            box_layout2b = QHBoxLayout()

            self.checkbox_cpp_metal = QCheckBox("Use Metal")
            self.checkbox_cpp_metal.setToolTip("Use Metal on MacOS")
            if platform.system() == "Darwin":
                self.checkbox_cpp_metal.setEnabled(False)
            else:
                self.checkbox_cpp_metal.setEnabled(False)

            #self.checkbox_cpp_coreml = QCheckBox("Use CoreML")
            #self.checkbox_cpp_coreml.setToolTip("Use CoreML on MacOS")
            #if platform.processor() == "arm":
            #    self.checkbox_cpp_coreml.setEnabled(True)
            #else:
            #    self.checkbox_cpp_coreml.setEnabled(False)

            self.checkbox_cpp_metal.stateChanged.connect(self.on_checkbox_changed_whispercpp)
            #self.checkbox_cpp_coreml.stateChanged.connect(self.on_checkbox_changed_whispercpp)

            box_layout2b.addWidget(self.checkbox_cpp_metal, Qt.AlignmentFlag.AlignLeft)
            #box_layout2b.addWidget(self.checkbox_cpp_coreml, Qt.AlignmentFlag.AlignLeft)

            w2b.setLayout(box_layout2b)
            grid.addWidget(w2b, r, 1, 1, 1, Qt.AlignmentFlag.AlignLeft)
            r += 1
        else:
            self.checkbox_cpp_cuda = QCheckBox("Use CUDA")
            self.checkbox_cpp_cuda.setToolTip("Use CUDA on supported GPUs")
            self.checkbox_cpp_cuda.setEnabled(True)
            box_layout_whisper_cpp.addWidget(self.checkbox_cpp_cuda, Qt.AlignmentFlag.AlignLeft)

            w2.setLayout(box_layout_whisper_cpp)
            grid.addWidget(w2, r, 1, 1, 1, Qt.AlignmentFlag.AlignLeft)
            r += 1

        w3 = QWidget()
        box_layout3 = QHBoxLayout()
        self.threads_label = QLabel("Threads:")
        self.threads = QLineEdit()
        self.threads.setFixedWidth(50)
        self.threads.setToolTip("Enter the number of threads used by whisper.cpp. Default is 4")
        box_layout3.addWidget(self.threads_label, Qt.AlignmentFlag.AlignLeft)
        box_layout3.addWidget(self.threads, Qt.AlignmentFlag.AlignLeft)

        w3.setLayout(box_layout3)
        grid.addWidget(w3, r, 1, 1, 1, Qt.AlignmentFlag.AlignLeft)
        r += 1

        w4 = QWidget()
        box_layout4 = QHBoxLayout()
        self.whispercpp_options_label = QLabel("Options:")
        self.whispercpp_options_label.setFixedWidth(50)
        self.whispercpp_options_line_edit = QLineEdit("")
        self.whispercpp_options_line_edit.setFixedWidth(250)
        self.whispercpp_options_line_edit.setToolTip("Enter optional whisper.cpp options")
        box_layout4.addWidget(self.whispercpp_options_label, Qt.AlignmentFlag.AlignLeft)
        box_layout4.addWidget(self.whispercpp_options_line_edit, Qt.AlignmentFlag.AlignLeft)
        w4.setLayout(box_layout4)
        grid.addWidget(w4, r, 1, 1, 1, Qt.AlignmentFlag.AlignLeft)
        r += 1

        if mainWindow.whispercpp_engine is None:
            self.label_whisper_cpp_model.setEnabled(False)
            self.combobox_whisper_cpp.setEnabled(False)
            #self.checkbox1_feedback.setEnabled(False)
            self.threads_label.setEnabled(False)
            self.threads.setEnabled(False)
            self.whispercpp_options_label.setEnabled(False)
            self.whispercpp_options_line_edit.setEnabled(False)

        # ---------------------------------------------------------------------------------------------------------
        # faster-whisper
        if mainWindow.faster_whisper_engine is not None:
            self.radio_faster_whisper = QRadioButton("Faster Whisper")
            self.radio_faster_whisper.setToolTip("Use Faster Whisper")
            self.radio_faster_whisper.clicked.connect(self.apply_faster_whisper)
            grid.addWidget(self.radio_faster_whisper, r, 0, 1, 2)

            # faster-whisper model
            w7 = QWidget()
            box_layout_faster_whisper = QHBoxLayout()
            self.model_label_fw = QLabel("Model:")
            self.comboModel_fw = QComboBox()
            self.comboModel_fw.setFixedWidth(150)
            box_layout_faster_whisper.addWidget(self.model_label_fw, Qt.AlignmentFlag.AlignLeft)
            box_layout_faster_whisper.addWidget(self.comboModel_fw, Qt.AlignmentFlag.AlignLeft)

            #if platform.system() != "Darwin":
            #    self.checkbox_fw_cuda = QCheckBox("Use CUDA")
            #    self.checkbox_fw_cuda.setToolTip("Use CUDA on compatible GPUs")
            #    self.checkbox_fw_cuda.setEnabled(app_utils.cuda_available())
            #    box_layout_faster_whisper.addWidget(self.checkbox_fw_cuda, Qt.AlignmentFlag.AlignLeft)
            #else:
            self.checkbox_fw_cuda = None

            w7.setLayout(box_layout_faster_whisper)
            grid.addWidget(w7, r, 1, 1, 1, Qt.AlignmentFlag.AlignLeft)
            r += 1

        # ---------------------------------------------------------------------------------------------------------
        # whisper asr webservice
        whisper_webservice_engine = mainWindow.whisper_webservice_engine
        self.radio_whisper_asr_webservice = QRadioButton("Whisper ASR Webservice")
        self.radio_whisper_asr_webservice.setToolTip("Use a Whisper ASR webservice")
        self.radio_whisper_asr_webservice.clicked.connect(self.apply_whisper_asr)
        grid.addWidget(self.radio_whisper_asr_webservice, r, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
        r += 1

        w5 = QWidget()
        box_layout_whisper_asr = QHBoxLayout()
        self.asr_url_label = QLabel("URL:")
        self.asr_url_label.setFixedWidth(50)
        self.asr_url_line_edit = QLineEdit("")
        self.asr_url_line_edit.setPlaceholderText("Enter URL")
        self.asr_url_line_edit.setToolTip("Enter the URL of the whisper ASR webservice")
        self.asr_url_line_edit.setFixedWidth(250)
        box_layout_whisper_asr.addWidget(self.asr_url_label, Qt.AlignmentFlag.AlignLeft)
        box_layout_whisper_asr.addWidget(self.asr_url_line_edit, Qt.AlignmentFlag.AlignLeft)
        w5.setLayout(box_layout_whisper_asr)
        grid.addWidget(w5, r, 1, 1, 1, Qt.AlignmentFlag.AlignLeft)
        r += 1

        if whisper_webservice_engine is None:
            self.radio_whisper_asr_webservice.setEnabled(False)
            self.asr_url_label.setEnabled(False)
            self.asr_url_line_edit.setEnabled(False)

        # ---------------------------------------------------------------------------------------------------------
        # whisper api
        whisper_api_engine = mainWindow.whisper_api_engine
        self.radio_whisper_openai_api = QRadioButton("Whisper OpenAI API")
        self.radio_whisper_openai_api.setToolTip("Use OpenAI's whisper API to run ASR on OpenAI servers")
        self.radio_whisper_openai_api.clicked.connect(self.apply_whisper_api)
        self.radio_whisper_openai_api.setEnabled(True)
        grid.addWidget(self.radio_whisper_openai_api, r, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
        r += 1

        w6 = QWidget()
        box_layout_whisper_api = QHBoxLayout()
        self.api_label = QLabel("API:")
        self.api_label.setFixedWidth(50)
        self.api_line_edit = QLineEdit("")
        self.api_line_edit.setFixedWidth(250)
        self.api_line_edit.setPlaceholderText("Enter API key")
        box_layout_whisper_api.addWidget(self.api_label, Qt.AlignmentFlag.AlignLeft)
        box_layout_whisper_api.addWidget(self.api_line_edit, Qt.AlignmentFlag.AlignLeft)
        w6.setLayout(box_layout_whisper_api)
        grid.addWidget(w6, r, 1, 1, 1, Qt.AlignmentFlag.AlignLeft)
        r += 1

        if whisper_api_engine is None:
            self.radio_whisper_openai_api.setEnabled(False)
            self.api_label.setEnabled(False)
            self.api_line_edit.setEnabled(False)

        # ---------------------------------------------------------------------------------------------------------

        # set main layout
        gb.setLayout(grid)

        ## Second group
        gb2 = QGroupBox()
        gb2.setFixedHeight(50)
        box_layout = QHBoxLayout()

        text = "Reset Settings"
        self.reset_button = QPushButton(text)
        self.reset_button.setToolTip("Reset settings and window size/location")
        self.reset_button.setFixedWidth(app_utils.str_width(self.reset_button, text) + self.mainWindow._TEXT_SPACE)
        self.reset_button.clicked.connect(self.reset_settings)
        box_layout.addWidget(self.reset_button)

        text = "Delete Downloaded Models"
        self.delete_models_button = QPushButton(text)
        self.delete_models_button.setToolTip("Delete downloaded models (whisper and whisper.cpp)")
        self.delete_models_button.setFixedWidth(app_utils.str_width(self.reset_button, text) + self.mainWindow._TEXT_SPACE)
        self.delete_models_button.clicked.connect(self.delete_downloaded_models)
        box_layout.addWidget(self.delete_models_button)

        gb2.setLayout(box_layout)

        self.layout.addWidget(gb)
        self.layout.addWidget(gb2)

        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

        # default size and prevent resizing
        self.setMaximumWidth(525)
        self.setMinimumWidth(525)
        self.resizeMe()

        # default
        self.set_content()

    def resizeMe(self) -> None:
        self.resize(self.minimumSizeHint())

    def on_checkbox_changed_whispercpp(self, state):
        sender = self.sender()
        if state:
            if sender == self.checkbox_cpp_metal:
                #self.checkbox_cpp_coreml.setChecked(False)
                ...
            else:
                self.checkbox_cpp_metal.setChecked(False)

    def set_content(self) -> None:
        settings = self.mainWindow.settings

        # openai-whisper
        whisper_engine = self.mainWindow.whisper_engine
        if whisper_engine is not None:
            self.combobox_openai_model.addItems(whisper_engine.whisper_models())
            self.combobox_openai_model.setCurrentText(settings.value("Settings/OpenAI_model"))

        # mlx-whisper
        mlx_whisper_engine = self.mainWindow.mlx_whisper_engine
        if mlx_whisper_engine is not None:
            self.combobox_mlx_model.addItems(mlx_whisper_engine.whisper_models())
            self.combobox_mlx_model.setCurrentText(settings.value("Settings/MLX_model"))

        # whisper.cpp
        whispercpp_engine = self.mainWindow.whispercpp_engine
        if whispercpp_engine is not None:
            self.combobox_whisper_cpp.addItems(whispercpp_engine.whisper_cpp_models())
            self.combobox_whisper_cpp.setCurrentText(settings.value("Settings/CPP_model"))
            self.threads.setText(settings.value("Settings/CPP_threads"))
            if platform.system() == "Darwin":
                self.checkbox_cpp_metal.setChecked(app_utils.bool_value(settings.value("Settings/CPP_Metal")))
                #self.checkbox_cpp_coreml.setChecked(app_utils.bool_value(settings.value("Settings/CPP_CoreML")))
            if platform.system() != "Darwin":
                self.checkbox_cpp_cuda.setChecked(app_utils.bool_value(settings.value("Settings/CPP_CUDA")))
            self.whispercpp_options_line_edit.setText(settings.value("Settings/CPP_options"))

        # faster-whisper
        faster_whisper_engine = self.mainWindow.faster_whisper_engine
        if faster_whisper_engine is not None:
            self.comboModel_fw.addItems(faster_whisper_engine.faster_whisper_models())
            self.comboModel_fw.setCurrentText(settings.value("Settings/FW_model"))
            if platform.system() != "Darwin" and self.checkbox_fw_cuda is not None:
                self.checkbox_fw_cuda.setChecked(app_utils.bool_value(settings.value("Settings/FW_CUDA")))

        # whisper asr webservice
        self.asr_url_line_edit.setText(settings.value("Whisper/WhisperASRwebservice_URL"))

        # whisper api
        self.api_line_edit.setText(settings.value("Whisper/WhisperOpenAI_API"))

        # If FFmpeg is not installed disable engine options that require FFmpeg
        if self.mainWindow.settings.value("FFmpeg/path") == "":
            # OpenAI's whisper
            if whisper_engine is not None:
                self.radio_openai_whisper.setEnabled(False)
                self.combobox_openai_model.setEnabled(False)
            # MLX
            if mlx_whisper_engine is not None:
                self.radio_mlx_whisper.setEnabled(False)
                self.combobox_mlx_model.setEnabled(False)

        if platform.system() == "Darwin" and platform.processor() == "i386":
            # disable faster-whisper on Intel Macs
            if faster_whisper_engine is not None:
                self.radio_faster_whisper.setEnabled(False)
                self.comboModel_fw.setEnabled(False)

        # enable/disable based on selected whisper version
        if self.mainWindow.settings.value("Whisper/Engine") == "whisper" and whisper_engine is not None:
            self.radio_openai_whisper.setChecked(True)
            self.apply_whisper()
        elif self.mainWindow.settings.value("Whisper/Engine") == "mlx-whisper" and mlx_whisper_engine is not None:
            self.radio_mlx_whisper.setChecked(True)
            self.apply_mlx_whisper()            
        elif self.mainWindow.settings.value("Whisper/Engine") == "whisper.cpp":
            self.radio_whisper_cpp.setChecked(True)
            self.apply_whisper_cpp()
        elif self.mainWindow.settings.value("Whisper/Engine") == "faster-whisper" and faster_whisper_engine is not None:
            self.radio_faster_whisper.setChecked(True)
            self.apply_faster_whisper()
        elif self.mainWindow.settings.value("Whisper/Engine") == "whisper_asr_webservice":
            self.radio_whisper_asr_webservice.setChecked(True)
            self.apply_whisper_asr()
        elif self.mainWindow.settings.value("Whisper/Engine") == "whisper.api":
            self.radio_whisper_openai_api.setChecked(True)
            self.apply_whisper_api()

    def apply_whisper(self) -> None:
        self.set_status_settings("whisper", True)
        self.set_status_settings("mlx-whisper", False)
        self.set_status_settings("whisper.cpp", False)
        self.set_status_settings("whisper.asr", False)
        self.set_status_settings("whisper.api", False)
        self.set_status_settings("faster-whisper", False)

    def apply_mlx_whisper(self) -> None:
        self.set_status_settings("whisper", False)
        self.set_status_settings("mlx-whisper", True)
        self.set_status_settings("whisper.cpp", False)
        self.set_status_settings("whisper.asr", False)
        self.set_status_settings("whisper.api", False)
        self.set_status_settings("faster-whisper", False)

    def apply_whisper_cpp(self) -> None:
        self.set_status_settings("whisper", False)
        self.set_status_settings("mlx-whisper", False)
        self.set_status_settings("whisper.cpp", True)
        self.set_status_settings("whisper.asr", False)
        self.set_status_settings("whisper.api", False)
        self.set_status_settings("faster-whisper", False)

    def apply_whisper_asr(self) -> None:
        self.set_status_settings("whisper", False)
        self.set_status_settings("mlx-whisper", False)
        self.set_status_settings("whisper.cpp", False)
        self.set_status_settings("whisper.asr", True)
        self.set_status_settings("whisper.api", False)
        self.set_status_settings("faster-whisper", False)

    def apply_whisper_api(self) -> None:
        self.set_status_settings("whisper", False)
        self.set_status_settings("mlx-whisper", False)
        self.set_status_settings("whisper.cpp", False)
        self.set_status_settings("whisper.asr", False)
        self.set_status_settings("whisper.api", True)
        self.set_status_settings("faster-whisper", False)

    def apply_faster_whisper(self) -> None:
        self.set_status_settings("whisper", False)
        self.set_status_settings("mlx-whisper", False)
        self.set_status_settings("whisper.cpp", False)
        self.set_status_settings("whisper.asr", False)
        self.set_status_settings("whisper.api", False)
        self.set_status_settings("faster-whisper", True)

    def update_mainwindow_output_options(self) -> None:
        # update form_widget on mainWindow to reflect file output options compatible with
        # selected ASR engine
        settings = self.mainWindow.settings
        engine:str = settings.value("Whisper/Engine")
        selected:str = settings.value("Settings/Output")
        logging.debug("Engine: %s, selected: %s", engine, selected)

        current_items = [self.mainWindow.form_widget.comboOutput.itemText(i) for i in range(self.mainWindow.form_widget.comboOutput.count())]
        logging.debug("update_mainwindow_output_options, Current: %s", current_items)

        output_items = self.mainWindow.output_items()
        logging.debug("update_mainwindow_output_options, New: %s", output_items)

        self.mainWindow.form_widget.comboOutput.setVisible(False)

        # show selected files
        self.mainWindow.show_selected_files()

        # ugly fix around clear(), addItem bug in Qt6
        i = 0
        while self.mainWindow.form_widget.comboOutput.count() > i:
            if self.mainWindow.form_widget.comboOutput.itemText(i) == 'TEXT':
                i += 1
            self.mainWindow.form_widget.comboOutput.removeItem(i)

        output_items.remove('TEXT')
        self.mainWindow.form_widget.comboOutput.addItems(output_items)

        index = 0
        for i in range(0, self.mainWindow.form_widget.comboOutput.count()):
            if self.mainWindow.form_widget.comboOutput.itemText(i) == selected:
                index = i
        self.mainWindow.form_widget.comboOutput.setCurrentIndex(index)

        #self.mainWindow.form_widget.comboOutput.setEnabled(True)
        self.mainWindow.form_widget.comboOutput.setVisible(True)

    def set_mainwindow_output_option(self, option_str) -> None:
        self.mainWindow.form_widget.comboOutput.setCurrentText(option_str)

    def update_mainwindow_language_options(self) -> None:
        # update language options based on ASR engine
        self.mainWindow.form_widget.comboLanguage.clear()
        self.mainWindow.form_widget.comboLanguage.addItems(self.mainWindow.form_widget.languages())

    def set_mainwindow_language_option(self, option_str) -> None:
        self.mainWindow.form_widget.comboLanguage.setCurrentText(option_str)

    def set_status_settings(self, radioStr:str, status: bool) -> None:
        if radioStr == "whisper" and self.mainWindow.whisper_engine is not None:
            self.label_openai_model.setEnabled(status)
            self.combobox_openai_model.setEnabled(status)
        elif radioStr == "mlx-whisper" and self.mainWindow.mlx_whisper_engine is not None:
            self.label_mlx_model.setEnabled(status)
            self.combobox_mlx_model.setEnabled(status)
        elif radioStr == "whisper.cpp":
            self.label_whisper_cpp_model.setEnabled(status)
            self.combobox_whisper_cpp.setEnabled(status)
            self.threads_label.setEnabled(status)
            self.threads.setEnabled(status)

            if platform.system() == "Darwin":
                #self.checkbox_cpp_metal.setEnabled(status)
                #self.checkbox_cpp_coreml.setEnabled(status)
                ...
            if app_utils.cuda_available():
                self.checkbox_cpp_cuda.setEnabled(status)

            self.whispercpp_options_label.setEnabled(status)
            self.whispercpp_options_line_edit.setEnabled(status)
        elif radioStr == "faster-whisper" and self.mainWindow.faster_whisper_engine is not None:
            self.model_label_fw.setEnabled(status)
            self.comboModel_fw.setEnabled(status)
            if app_utils.cuda_available() and self.checkbox_fw_cuda is not None:
                self.checkbox_fw_cuda.setEnabled(status)     
        elif radioStr == "whisper.asr":
            self.asr_url_label.setEnabled(status)
            self.asr_url_line_edit.setEnabled(status)
            if status:
                self.asr_url_line_edit.setFocus()
        elif radioStr == "whisper.api":
            self.api_label.setEnabled(status)
            self.api_line_edit.setEnabled(status)
            if status:
                self.api_line_edit.setFocus()

    def accept(self) -> None:
        ok:bool = False
        settings = self.mainWindow.settings
        form_widget = self.mainWindow.form_widget

        previous_engine:str = settings.value("Whisper/Engine")

        if self.mainWindow.whisper_engine is not None and self.radio_openai_whisper.isChecked():
            settings.setValue("Whisper/Engine", "whisper")
            settings.setValue("Settings/OpenAI_model", self.combobox_openai_model.currentText())
            ok = True

        elif self.mainWindow.mlx_whisper_engine is not None and self.radio_mlx_whisper.isChecked():
            settings.setValue("Whisper/Engine", "mlx-whisper")
            settings.setValue("Settings/MLX_model", self.combobox_mlx_model.currentText())
            ok = True

        elif self.radio_whisper_cpp.isChecked():
            settings.setValue("Whisper/Engine", "whisper.cpp")
            settings.setValue("Settings/CPP_model", self.combobox_whisper_cpp.currentText())
            settings.setValue("Settings/CPP_threads", self.threads.text())
            if platform.system() == "Darwin":
                settings.setValue("Settings/CPP_Metal", app_utils.bool_to_string(self.checkbox_cpp_metal.isChecked()))
                #settings.setValue("Settings/CPP_CoreML", app_utils.bool_to_string(self.checkbox_cpp_coreml.isChecked()))                     
            if platform.system() != "Darwin":
                settings.setValue("Settings/CPP_CUDA", app_utils.bool_to_string(self.checkbox_cpp_cuda.isChecked()))
            settings.setValue("Settings/CPP_options", self.whispercpp_options_line_edit.text())
            ok = True

        elif self.mainWindow.faster_whisper_engine is not None and self.radio_faster_whisper.isChecked():
            settings.setValue("Whisper/Engine", "faster-whisper")
            settings.setValue("Settings/FW_model", self.comboModel_fw.currentText())
            if platform.system() != "Darwin" and self.checkbox_fw_cuda is not None:
                settings.setValue("Settings/FW_CUDA", app_utils.bool_to_string(self.checkbox_fw_cuda.isChecked()))
            ok = True

        elif self.radio_whisper_asr_webservice.isChecked():
            settings.setValue("Whisper/Engine", "whisper_asr_webservice")

            new_url = self.asr_url_line_edit.text()
            if new_url == "" or new_url != settings.value("Whisper/WhisperASRwebservice_URL"):

                if validators.url(new_url):
                    if settings.contains("Whisper/WhisperASRwebservice_URL"):
                        settings.setValue("Whisper/WhisperASRwebservice_URL", new_url)
                    else:
                        settings.beginGroup("Whisper")
                        settings.setValue("WhisperASRwebservice_URL", new_url)
                        settings.endGroup()

                    form_widget.button1.setEnabled(True)
                    form_widget.button1.setDefault(True)
                    ok = True

                else:
                    dlg = QMessageBox(self)
                    dlg.setIcon(QMessageBox.Icon.Warning)
                    dlg.setWindowModality(Qt.WindowModality.WindowModal)
                    dlg.setText(f"URL '{new_url}' not valid. Please enter a valid URL.")
                    dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
                    _ = dlg.exec()

                    form_widget.button1.setEnabled(False)
            else:
                # no change
                ok = True

        elif self.radio_whisper_openai_api.isChecked():
            settings.setValue("Whisper/Engine", "whisper.api")

            new_api = self.api_line_edit.text()
            if new_api != settings.value("Whisper/WhisperOpenAI_API"):

                if new_api != "":
                    if settings.contains("Whisper/WhisperOpenAI_API"):
                        settings.setValue("Whisper/WhisperOpenAI_API", new_api)
                    else:
                        settings.beginGroup("Whisper")
                        settings.setValue("WhisperOpenAI_API", new_api)
                        settings.endGroup()

                    form_widget.button1.setEnabled(True)
                    form_widget.button1.setDefault(True)

                    ok = True

                else:
                    dlg = QMessageBox(self)
                    dlg.setIcon(QMessageBox.Icon.Warning)
                    dlg.setWindowModality(Qt.WindowModality.WindowModal)
                    dlg.setText("API key not provided.")
                    dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
                    _ = dlg.exec()

                    form_widget.button1.setEnabled(False)
            else:
                # no change
                ok = True

        if ok:
            form_widget.delete_feedback()

            # set task options
            self.mainWindow.update_task_options()

            engine = settings.value("Whisper/Engine")
            if engine == "faster-whisper" and settings.value("Settings/Output") not in ['VTT', 'SRT', 'TEXT']:
                # reset to VTT because other engines do not support this output
                settings.setValue("Settings/Output", "VTT")
                self.set_mainwindow_output_option("VTT")

            if previous_engine == "whisper.cpp" and engine != "whisper.cpp" and \
                settings.value("Settings/Output") in ['TSV', 'LRC']:
                # reset to VTT because other engines do not support this output
                settings.setValue("Settings/Output", "VTT")
                self.set_mainwindow_output_option("VTT")

            if engine == "whisper_asr_webservice":
                # check if server is available
                self.mainWindow.check_if_server_is_running()

            self.update_mainwindow_output_options()

            self.close()

    def reset_settings(self) -> None:

        dlg = QMessageBox(self)
        dlg.setIcon(QMessageBox.Icon.Warning)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setText("Reset settings?")
        dlg.setStandardButtons(QMessageBox.StandardButton.No | QMessageBox.StandardButton.Yes)
        r = dlg.exec()
        if r == QMessageBox.StandardButton.Yes:
            # reset settings
            self.mainWindow.reset_ini_file()

            # change window size
            self.mainWindow.restore_window()

            # Check if FFmpeg installed
            self.mainWindow.check_ffmpeg_installed()

            # update content settings window
            self.set_content()

            # update main window content
            self.mainWindow.form_widget.set_content()

    def delete_downloaded_models(self) -> None:
        dlg = QMessageBox(self)
        dlg.setIcon(QMessageBox.Icon.Warning)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setText("Delete the downloaded models?")
        dlg.setStandardButtons(QMessageBox.StandardButton.No | QMessageBox.StandardButton.Yes)
        r = dlg.exec()
        if r == QMessageBox.StandardButton.No:
            return

        self.mainWindow.delete_downloaded_models()