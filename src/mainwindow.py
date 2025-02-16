""" Module defines the MainWindow class """

# Copyright © 2023-2025 Walter van Heuven

import os
import sys
import shutil
import subprocess
import threading
import logging
import platform
import time
import socket
import urllib.parse
from enum import Enum
import requests
import torch
from packaging import version
from PyQt6.QtGui import QDesktopServices, QScreen, QAction, QGuiApplication
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PyQt6.QtCore import Qt, QCoreApplication, QStandardPaths, QSettings
from PyQt6.QtCore import QUrl, QPoint, QRect
import utils as app_utils
from mainwindow_content import FormWidget
from settings import SettingsDialog
from convert_worker import ConvertWorker
from whispercpp_engine import WhisperCPPEngine
from whisper_webservice_engine import WhisperWebserviceEngine
from whisper_api_engine import WhisperAPIEngine
from whisper_engine import WhisperEngine
from faster_whisper_engine import FasterWhisperEngine

_APP_NAME: str = "Speech2Text"

class Status(Enum):
    IDLE: str = "Idle"
    PROCESSING: str = "Processing"
    CONVERTING: str = "Converting"
    CANCELLING: str = "Cancelling"

class MainWindow(QMainWindow):
    """ Create Main Window """

    # space around text buttons, labels, etc. 20px on each side
    _TEXT_SPACE: int = 40

    def __init__(self, app_name, app_version, version_url, website_url, author, debug) -> None:
        super().__init__()

        self.APP_NAME:str = app_name
        self.VERSION:str = app_version
        self.VERSION_URL:str = version_url
        self.WEBSITE_URL:str = website_url
        self.AUTHOR:str = author
        self.DEBUG:bool = debug
        self.STOP: bool = False

        self.insecure_server_ok_all: bool = False
        self.overwrite_all_existing_files: bool = False
        self.not_overwrite_all_existing_files: bool = False

        self.worker = None

        self.whisper_engine = None
        self.mlx_whisper_engine = None
        self.faster_whisper_engine = None
        self.whispercpp_engine = None
        self.whisper_webservice_engine = None
        self.whisper_api_engine = None

        self.savedBg = None
        self.tic = None

        # current folder
        self.active_folder:str = os.path.expanduser("~")

        # files to process
        self.filenames: list[str] = []
        self.status:Status = Status.IDLE
        self.current_subprocess: subprocess.Popen = None

        self.settings:QSettings = None
        self.ready_to_accept_files:bool = True

        self.selected_filter:str = ""

        # create ini file / read ini file
        self.create_ini_file()

        # ask for URL if not available in settings file or when new version of the app is installed
        app_version_ini:str = self.settings.value("Application/version")
        if app_version_ini is None or version.parse(app_version_ini) != version.parse(self.VERSION):
            self.reset_ini_file()
            self.delete_downloaded_models()

        # check if ffmpeg is installed on the computer
        self.check_ffmpeg_installed()

        # init ASR egines
        self.init_asr_engines()

        # init gui
        self.init_gui()

        # menuBar
        self.create_menu()

        # set task options
        self.update_task_options()

        # close window means Quit
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        # check for new version of the app
        _ = self.check_new_version_available()

        # check if Whisper ASR webservice is available
        self.check_if_server_is_running()

        self.mySize = [self.width(), self.height()]

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape.value:
            if self.DEBUG:
                print("Pressed Escape...")
                print(self.get_worker_name())

            self.stop_processing()
    
    def init_asr_engines(self) -> None:
        """ Init ASR engines """
        self.whispercpp_engine = WhisperCPPEngine(self)
        self.whisper_webservice_engine = WhisperWebserviceEngine(self)
        self.whisper_api_engine = WhisperAPIEngine(self)

        logging.debug("System: %s, %s", platform.system(), platform.processor())

        self.whisper_engine = WhisperEngine(self)

        if (platform.system() == "Darwin" and platform.processor() == "arm"):
            # only import on Apple Silicon Macs
            from whisper_mlx_engine import WhisperMLXEngine
            self.mlx_whisper_engine = WhisperMLXEngine(self)

            self.faster_whisper_engine = FasterWhisperEngine(self)

            if self.settings.value("FFmpeg/path") == "":
                if self.settings.value("Whisper/Engine") == "whisper" or self.settings.value("Whisper/Engine") == "mlx-whisper":
                    self.settings.setValue("Whisper/Engine", "whisper.cpp")

        elif (platform.system() == "Darwin" and platform.processor() == "i386"):
            # Note issue with Faster-Whisper on Intel Macs
            # Multiple Intel OpenMP runtime libraries, fix with: KMP_DUPLICATE_LIB_OK=TRUE
            # torch version too old?

            if self.settings.value("FFmpeg/path") == "":
                if self.settings.value("Whisper/Engine") == "whisper" or self.settings.value("Whisper/Engine") == "faster-whisper":
                    self.settings.setValue("Whisper/Engine", "whisper.cpp")

        elif platform.system() == "Windows":

            self.faster_whisper_engine = FasterWhisperEngine(self)

            if self.settings.value("FFmpeg/path") == "":              
                if self.settings.value("Whisper/Engine") == "whisper":
                    self.settings.setValue("Whisper/Engine", "whisper.cpp")

            # issue with faster_whisper on windows
            # 'could not load library cudnn_ops_infer64_8.dll'
            # requires toch with cuda libraries
        else:
            self.faster_whisper_engine = FasterWhisperEngine(self)
            
            if self.settings.value("FFmpeg/path") == "":
                if self.settings.value("Whisper/Engine") == "whisper":
                    self.settings.setValue("Whisper/Engine", "whisper.cpp")

    def init_gui(self) -> None:
        self.setWindowTitle("Speech2Text")

        self.form_widget:FormWidget = FormWidget(self)
        self.setCentralWidget(self.form_widget)

        # set content based on in
        self.form_widget.set_content()

        # accept file drops
        self.setAcceptDrops(True)

        # show app info
        self.show_app_info(True)

        # update main dailog values of comboboxes in form based on ini file
        if self.settings.value("Settings/Output") != "":
            self.form_widget.comboOutput.setCurrentText(self.settings.value("Settings/Output"))
        if self.settings.value("Settings/Language") != "":
            self.form_widget.comboLanguage.setCurrentText(self.settings.value("Settings/Language"))
        if self.settings.value("Settings/Task") != "":
            self.form_widget.comboTask.setCurrentText(self.settings.value("Settings/Task"))

        # set window location and size
        self.restore_window()

        # show window
        self.show()

    def create_ini_file(self) -> None:
        ini_file_fpath:str = os.path.join(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation), f"{self.APP_NAME}.ini")
        self.settings:QSettings = QSettings(ini_file_fpath, QSettings.Format.IniFormat, self)

        # load settigs
        logging.info("Location ini file: %s", ini_file_fpath)
        self.settings.sync()

    def reset_ini_file(self) -> None:
        logging.info("Reset settings")

        # reset
        self.settings.clear()

        # default
        self.settings.beginGroup("Application")
        self.settings.setValue("version", self.VERSION)
        self.settings.endGroup()

        self.settings.beginGroup("FFmpeg")
        self.settings.setValue("path", "")
        self.settings.endGroup()

        self.settings.beginGroup("Whisper")
        if platform.system() == "Darwin" and platform.processor() == "arm":
            self.settings.setValue("Engine", "mlx-whisper")
        else:
            self.settings.setValue("Engine", "whisper.cpp")
        self.settings.setValue("WhisperASRwebservice_URL", "")
        self.settings.setValue("WhisperOpenAI_API", "")
        self.settings.endGroup()

        self.settings.beginGroup("Settings")
        # general
        self.settings.setValue("Output", "VTT")
        self.settings.setValue("Language", "Auto Detect")
        self.settings.setValue("Task", "transcribe")
        # whisper
        self.settings.setValue("OpenAI_model", "base")
        # mlx-whisper
        self.settings.setValue("MLX_model", "base")
        # whisper.cpp
        self.settings.setValue("CPP_model", "base")
        self.settings.setValue("CPP_threads", "4")

        if platform.system() == "Darwin":
            self.settings.setValue("CPP_Metal", "True")
            #self.settings.setValue("CPP_CoreML", "False")
        else:
            self.settings.setValue("CPP_Metal", "False")
            #self.settings.setValue("CPP_CoreML", "False")
        
        if app_utils.cuda_available():
            self.settings.setValue("CPP_CUDA", "True")
        else:
            self.settings.setValue("CPP_CUDA", "False")
        self.settings.setValue("CPP_options", "")
        # faster-whisper
        self.settings.setValue("FW_model", "base")
        if app_utils.cuda_available():
            self.settings.setValue("FW_CUDA", "True")
        else:
            self.settings.setValue("FW_CUDA", "False")
        self.settings.endGroup()

        # Window settings
        self.settings.beginGroup("Window")
        self.settings.setValue("x", "")
        self.settings.setValue("y", "")
        self.settings.setValue("width", "")
        self.settings.setValue("height", "")
        self.settings.endGroup()

        # reset window size and position
        self.reset_window()

    def reset_window(self) -> None:
        # default
        # move to 1/3 from top

        primary_screen:QScreen = QApplication.primaryScreen()
        screen_geometry:QRect = primary_screen.geometry()
        mh:int = self.minimumHeight()
        mw:int = self.minimumWidth()
        h:int = screen_geometry.height() - mh
        w:int = screen_geometry.width() - mw
        x:int = w // 2
        y:int = h // 2 // 2
        self.move(QPoint(x, y))
        self.resize(mh, mw)

        # store window dimensions and location
        self.settings.setValue("Window/x", x)
        self.settings.setValue("Window/y", y)
        self.settings.setValue("Window/width", mw)
        self.settings.setValue("Window/height", mh)

    def restore_window(self) -> None:
        stored_x = self.settings.value("Window/x")
        stored_y = self.settings.value("Window/y")
        stored_w = self.settings.value("Window/width")
        stored_h = self.settings.value("Window/height")

        if stored_x is not None and stored_y is not None and stored_w is not None and stored_h is not None:
            try:
                stored_x = int(stored_x)
                stored_y = int(stored_y)
                stored_w = int(stored_w)
                stored_h = int(stored_h)
            except ValueError as e:
                logging.exception("Converting stored location to int failure: %s", e)
                self.reset_window()
            else:
                self.move(QPoint(stored_x, stored_y))
                self.resize(stored_w, stored_h)
        else:
            self.reset_window()

    def check_ffmpeg_installed(self):
        """ check if ffmpeg is installed """
        err:bool = False
        ffmpeg_not_found:bool = False
        msg:str

        logging.debug("Determine ffmpeg path...")
        ffmpeg_fpath:str = self.settings.value("FFmpeg/path")
        logging.debug("Current ffmpeg path: %s", ffmpeg_fpath)

        env:str = self.set_env()
        logging.debug("PATH: %s", env['PATH'])

        cmd:str
        try:
            if platform.system() == "Windows":
                env, si = app_utils.get_windows_env()

                cmd = ['powershell', '-WindowStyle', 'Hidden', '-ExecutionPolicy', "bypass", \
                       '-noninteractive', '-Command', "(gcm ffmpeg | Select-Object 'Source')"]
                p = subprocess.run(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=False, shell=False, startupinfo=si, env=env, check=True
                    )

                # hack to get path to FFmpeg
                lines = p.stdout.decode('utf-8').split(os.linesep)

                if len(lines) > 3:
                    # it's on the 3rd line
                    ffmpeg_fpath = lines[3]
                else:
                    logging.debug("Output subprocess.run: %d\n%s", len(lines), lines)
                    ffmpeg_not_found = True
            else:
                # Use which to find path to FFmpeg on Darwin/Unix
                cmd = ['which', 'ffmpeg']
                p = subprocess.run(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=False, shell=False, env=env, check=True
                    )

                ffmpeg_fpath:str = p.stdout.decode('utf-8').strip()

                logging.debug("ffmpeg_fpath: %s", ffmpeg_fpath)
                if ffmpeg_fpath is None or ffmpeg_fpath == "":
                    ffmpeg_not_found = True

        except subprocess.CalledProcessError as e:
            logging.exception("CalledProcessError (FFmpeg): %s", e)
            msg = f"which FFmpeg call error, {str(e)}"
            err = True
        finally:
            logging.debug("Check if ffmpeg is installed: %s", ffmpeg_fpath)

            if err:
                if hasattr(self, 'form_widget'):
                    #if self.form_widget is not None:
                    self.form_widget.feedback(msg)
                ffmpeg_fpath = ""
            elif ffmpeg_fpath is None or ffmpeg_fpath == "" or ffmpeg_fpath == "ffmpeg not found" or ffmpeg_fpath.startswith('gsm') or ffmpeg_not_found:
                logging.error("FFmpeg not installed")
                ffmpeg_fpath = ""
                err = True

        logging.debug("Saving ffmpeg in ini file: %s", ffmpeg_fpath)

        if self.settings.contains("FFmpeg/path"):
            self.settings.setValue("FFmpeg/path", ffmpeg_fpath)
        else:
            self.settings.beginGroup("FFmpeg")
            self.settings.setValue("path", ffmpeg_fpath)
            self.settings.endGroup()

    def set_env(self) -> str:
        env = os.environ
        if getattr(sys, 'frozen', False):
            logging.debug('Running in a PyInstaller bundle')

            # add paths where FFmpeg could be found
            # make sure path exists and is not already in env['PATH']

            if platform.system() == "Windows":
                # add Scoop path
                scoop_path = os.path.expanduser('~') + r"\scoop\shims"
                if not scoop_path in env['PATH'] and os.path.isdir(scoop_path):
                    env['PATH'] = f"{scoop_path}{os.pathsep}{env['PATH']}"

                # add Chocolatey path
                chocolatey_path = r"C:\ProgramData\chocolatey\bin"
                if not chocolatey_path in env['PATH'] and os.path.isdir(chocolatey_path):
                    env['PATH'] = f"{chocolatey_path}{os.pathsep}{env['PATH']}"

                # add C:\ffmpeg\bin path; installed by running the FFmpeg installer
                ffmpeg_path = r"C:\ffmpeg\bin"
                if not ffmpeg_path in env['PATH'] and os.path.isdir(ffmpeg_path):
                    env['PATH'] = f"{ffmpeg_path}{os.pathsep}{env['PATH']}"

            elif platform.system() == "Darwin":
                # add FFmpeg path for Apple Silicon Mac
                brew_path = "/opt/homebrew/bin"
                if not brew_path in env['PATH'] and os.path.isdir(brew_path):
                    env['PATH'] = f"{brew_path}{os.pathsep}{env['PATH']}"

                # add FFmpeg path for Intel Mac
                local_bin_path = "/usr/local/bin"
                if not local_bin_path in env['PATH'] and os.path.isdir(local_bin_path):
                    env['PATH'] = f"{local_bin_path}{os.pathsep}{env['PATH']}"

        return env

    def check_new_version_available(self) -> bool:
        err: bool = False
        current_version:str = self.settings.value("Application/version")
        latest_version:str = ""
        response:requests.Request = None
        new_version_available:bool = False
        msg:str
        try:
            response = requests.get(self.VERSION_URL, timeout=5)
        except requests.exceptions.RequestException as e:
            # exception that captures all possible exceptions
            logging.exception("RequestException (check version): %s", e)
            err = True

        if response is not None and response.status_code == 200:
            latest_version = response.content.decode("utf-8")
            logging.info("Latest version: %s, current version: %s", latest_version, current_version)

            if version.parse(current_version) < version.parse(latest_version):
                new_version_available = True
                dlg = QMessageBox(self)
                dlg.setWindowModality(Qt.WindowModality.WindowModal)
                dlg.setIcon(QMessageBox.Icon.Information)
                dlg.setTextFormat(Qt.TextFormat.RichText)
                dlg.setTextInteractionFlags(
                    Qt.TextInteractionFlag.LinksAccessibleByMouse |
                    Qt.TextInteractionFlag.LinksAccessibleByKeyboard
                )

                msg = f"New version available: v{latest_version}<br><br>Press the OK button to visit the website where you can download the new version."
                dlg.addButton(QMessageBox.StandardButton.Cancel)
                dlg.addButton(QMessageBox.StandardButton.Ok)
                dlg.setDefaultButton(QMessageBox.StandardButton.Ok)
                dlg.setText(msg)
                b = dlg.exec()

                if b == QMessageBox.StandardButton.Ok:
                    QDesktopServices.openUrl(QUrl(self.WEBSITE_URL))
        else:
            if response is not None:
                msg = f"Unable to check for latest version online: {str(response.status_code)}"
                err = True
            else:
                msg = f"Unable to access '{self.VERSION_URL}'"
                err = True

            if err:
                logging.warning(msg)
                self.feedback("\nUnable to check online for new version of this app.")

        return new_version_available

    # override closeEvent
    def closeEvent(self, event) -> None:
        #self.stop_processing()
        logging.info("Closed window")
        event.accept()

    # override resizeEvent
    def resizeEvent(self, event) -> None:
        self.settings.setValue("Window/height", self.height())
        self.settings.setValue("Window/width", self.width())
        event.accept()

    # override moveEvent
    def moveEvent(self, event) -> None:
        self.settings.setValue("Window/x", self.x())
        self.settings.setValue("Window/y", self.y())
        event.accept()

    def set_status(self, status: Status) -> Status:
        self.status = status

    def get_status(self) -> Status:
        return self.status

    def stop_processing(self) -> None:
        # only execute once
        if not self.STOP and self.get_status() != Status.CANCELLING:
            self.STOP = True

            self.set_status(Status.CANCELLING)
            self.form_widget.cancel_button.setText(self.get_status().value)
            self.form_widget.cancel_button.update()
            self.form_widget.cancel_button.setEnabled(False)

            if self.whispercpp_engine:
                if self.whispercpp_engine.worker is not None:
                    self.whispercpp_engine.worker.requestInterruption()

            if self.whisper_webservice_engine is not None:
                if self.whisper_webservice_engine.worker is not None:
                    self.feedback("Cancellation in process. Server processing cannot be interupted.")

            if self.whisper_api_engine is not None:
                if self.whisper_api_engine.worker is not None:
                    self.feedback("Cancellation in process. Server processing cannot be interupted.")

            if self.faster_whisper_engine:
                if self.faster_whisper_engine.worker is not None:
                    self.faster_whisper_engine.worker.requestInterruption()

            if self.whisper_engine:
                if self.whisper_engine.worker is not None:
                    self.whisper_engine.worker.requestInterruption()
            
            if self.mlx_whisper_engine:
                if self.mlx_whisper_engine.worker is not None:
                    logging.debug("Interrupting mlx_whisper worker")
                    self.mlx_whisper_engine.worker.interrupt.emit()

    def get_worker_name(self) -> list[str]:
        """ return name of the current whisper worker """
        mythreads: list[threading.Thread] = threading.enumerate()
        name:str = ""
        wlist = []

        #print(f"Active: {threading.active_count()}")

        for t in mythreads:
            wlist.append(t.name)
            logging.debug("thread: %s, number of threads: %d", t.name, len(mythreads))
            if t.name != "MainThread":
                name = t.name
                logging.debug("Worker name: %s", name)
                break
        return wlist

    def dragEnterEvent(self, event) -> None:
        # only accept if processing is completed or no files are selected and whisper url is provided
        self.savedBg = self.form_widget.terminal.styleSheet()
        if self.get_status() == Status.IDLE and self.ready_to_accept_files:
            self.form_widget.terminal.setStyleSheet("QTextBrowser { background-color: gray; }")
            event.accept()

    def dragLeaveEvent(self, event) -> None:
        self.form_widget.terminal.setStyleSheet(self.savedBg)
        event.accept()

    def dropEvent(self, event) -> None:
        self.form_widget.terminal.setStyleSheet(self.savedBg)
        files: list[str] = [u.toLocalFile() for u in event.mimeData().urls()]
        msg:str = f"Drop event files: {list(files)}"
        logging.debug(msg)

        acceptable_extensions: list[str] = self.acceptable_extensions()
        msg = f"acceptable_extensions: {acceptable_extensions}"
        logging.debug(msg)
        files_to_accept = []
        for fpath in files:
            if app_utils.check_acceptable_file(self, fpath, acceptable_extensions):
                files_to_accept.append(fpath)

            elif os.path.isdir(fpath):
                for filename in os.listdir(fpath):
                    fp = os.path.join(fpath, filename)
                    if app_utils.check_acceptable_file(self, fp, acceptable_extensions):
                        files_to_accept.append(fp)

        msg = f"Drop event accepted files: {files_to_accept}"
        logging.debug(msg)

        self.clear_queue()
        self.add_to_queue(files_to_accept)
        if self.queue_length() > 0:
            self.files_received()
            event.accept()
        else:
            dlg = QMessageBox(self)
            dlg.setIcon(QMessageBox.Icon.Warning)
            dlg.setWindowModality(Qt.WindowModality.WindowModal)
            dlg.setText("No compatible audio/video files found.")
            dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
            _ = dlg.exec()

    def acceptable_extensions(self) -> list[str]:
        extensions:list[str] = [".wav", ".mp3", ".mp4", ".m4a", ".aiff", ".mpeg", ".mov", ".avi", ".wmv", ".webm"]
        ffmpeg_fpath:str = self.settings.value("FFmpeg/path")
        engine:str = self.settings.value("Whisper/Engine")

        if ffmpeg_fpath == "":
            # ffmpeg not installed
            if engine == "whisper.cpp":
                extensions = ['.wav']
            elif engine == "whisper":
                # requires FFmpeg
                extensions = []

        if engine == "whisper_asr_webservice":
            # exclude '.avi' because it does not seem to work
            # all others formats are converted on the server to wav
            extensions = [".mp3", ".wav", ".mp4", ".aiff", ".mpeg", ".mov", ".wmv", ".webm"]
        elif engine == "whisper.api":
            extensions = [".mp3", ".wav", ".mp4", ".mpeg", ".mpega", ".m4a", ".flac", ".ogg", ".webm"]

        return extensions

    def create_menu(self) -> None:
        """ Set Menu """
        self.menuBar().setNativeMenuBar(True)

        in_windows = platform.system() == "Windows"
        winStyle = ""
        if in_windows:
            # check colour mode - not yet used
            #style = QGuiApplication.styleHints()
            #if style.colorScheme() == Qt.ColorScheme.Light:
            #else
            winStyle = "QMenu::item {padding: 2px 25px;} QMenu::separator {height: 1px; width: 100%; margin-left: 0px; margin-right: 0px; margin-top: 5px; margin-bottom: 5px;} QMenu::item::selected {background-color: rgb(211,211,211)}"

        self.fileMenu = self.menuBar().addMenu("File")
        if in_windows:
            self.fileMenu.setStyleSheet(winStyle)
        self.menu_open_action = QAction("Select Audio/Video file(s)", self)
        self.menu_open_action.setShortcut('Ctrl+O')
        self.menu_open_action.triggered.connect(self.select_files)

        #self.fileMenu.addAction("&Open", self.select_files)
        self.fileMenu.addAction(self.menu_open_action)
        #self.fileMenu.addSeparator()
        #self.fileMenu.addAction("Save")

        # Edit
        self.editMenu = self.menuBar().addMenu("Edit")
        if in_windows:
            self.editMenu.setStyleSheet(winStyle)
        cut_action = QAction("Cut", self)
        cut_action.setShortcut('Ctrl+X')
        self.editMenu.addAction(cut_action)

        copy_action = QAction("Copy", self)
        copy_action.setShortcut('Ctrl+C')
        copy_action.triggered.connect(self.copy)
        self.editMenu.addAction(copy_action)

        paste_action = QAction("Paste", self)
        paste_action.setShortcut('Ctrl+V')
        self.editMenu.addAction(paste_action)

        if not platform.system() == "Darwin":
            self.editMenu.addSeparator()
            settings_action = QAction("Settings", self)
            settings_action.triggered.connect(self.show_settings_dlg)
            self.editMenu.addAction(settings_action)

        # ASR
        self.asrMenu = self.menuBar().addMenu("ASR")
        if in_windows:
            self.asrMenu.setStyleSheet(winStyle)

        self.start_action = QAction("Start Speech Recognition", self)
        self.start_action.setShortcut('Ctrl+G')
        self.start_action.triggered.connect(self.do_process_files)
        self.asrMenu.addAction(self.start_action)
        # at startup disable
        self.start_action.setEnabled(False)

        self.asrMenu.addSeparator()

        show_files_action = QAction("Show Selected Files", self)
        show_files_action.setShortcut('Ctrl+I')
        show_files_action.triggered.connect(self.menu_list_selected_files)
        self.asrMenu.addAction(show_files_action)

        self.asrMenu.addSeparator()

        output_folder_action = QAction("Output Folder", self)
        output_folder_action.setShortcut('Ctrl+J')
        output_folder_action.triggered.connect(self.open_output_folder)
        self.asrMenu.addAction(output_folder_action)

        # Window
        self.windowMenu = self.menuBar().addMenu("Window")
        if in_windows:
            self.windowMenu.setStyleSheet(winStyle)
        minimize_action = QAction("Minimize", self)
        minimize_action.setShortcut('Ctrl+M')
        minimize_action.triggered.connect(self.menu_mainwindow_minimize)
        self.windowMenu.addAction(minimize_action)

        # Help
        self.helpMenu = self.menuBar().addMenu("Help")
        if in_windows:
            self.helpMenu.setStyleSheet(winStyle)

        # Settings / Preferences
        if platform.system() == "Darwin":
            settings_action = QAction("Settings", self)
            settings_action.triggered.connect(self.show_settings_dlg)
            settings_action.setMenuRole(QAction.MenuRole.PreferencesRole)
            self.helpMenu.addAction(settings_action)

        info_action = QAction("Info", self)
        info_action.triggered.connect(self.menu_info)
        self.helpMenu.addAction(info_action)

        website_action = QAction("Visit Website", self)
        website_action.triggered.connect(self.menu_visit_website)
        self.helpMenu.addAction(website_action)

        self.helpMenu.addSeparator()

        logfile_action = QAction("Open App Logfile", self)
        logfile_action.triggered.connect(self.menu_open_logfile)
        self.helpMenu.addAction(logfile_action)

        self.helpMenu.addSeparator()

        check_update_action = QAction("Check for Update...", self)
        check_update_action.triggered.connect(self.menu_check_update)
        self.helpMenu.addAction(check_update_action)

        self.helpMenu.addSeparator()

        # About
        about_action = QAction("About", self)
        about_action.triggered.connect(self.about_menu)
        about_action.setMenuRole(QAction.MenuRole.AboutRole)
        self.helpMenu.addAction(about_action)

    def about_menu(self):
        self.showNormal()
        self.show_app_info(False)

    def copy(self):
        clipboard = QGuiApplication.clipboard()
        cursor = self.form_widget.terminal.textCursor()
        clipboard.setText(cursor.selectedText())

    def paste(self):
        """ not used """
        clipboard = QGuiApplication.clipboard()
        mimeData = clipboard.mimeData()

        if mimeData.hasText():
            logging.debug("Paste: %s", mimeData.text())
            #setText(mimeData.text())
            #setTextFormat(Qt.PlainText)

    def menu_list_selected_files(self):
        self.show_selected_files()

    def menu_mainwindow_minimize(self):
        self.showMinimized()

    def menu_visit_website(self):
        QDesktopServices.openUrl(QUrl(self.WEBSITE_URL))

    def menu_info(self):
        self.showNormal()
        self.show_app_info(True)

    def menu_check_update(self):
        new_version = self.check_new_version_available()
        if not new_version:
            dlg = QMessageBox(self)
            dlg.setWindowModality(Qt.WindowModality.WindowModal)
            dlg.setIcon(QMessageBox.Icon.Information)
            dlg.setTextFormat(Qt.TextFormat.RichText)

            msg = f"No update available.\n\nVersion: {self.VERSION}"
            dlg.addButton(QMessageBox.StandardButton.Ok)
            dlg.setDefaultButton(QMessageBox.StandardButton.Ok)
            dlg.setText(msg)
            _ = dlg.exec()

    def menu_open_logfile(self):
        app_data_location:str = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
        log_path:str = os.path.join(app_data_location, f"{_APP_NAME}.log")

        QDesktopServices.openUrl(QUrl(f"file:///{log_path}"))

    def clear_queue(self) -> None:
        self.filenames:list[str] = []

    def queue(self) -> list[str]:
        return self.filenames

    def first_in_queue(self) -> str:
        if self.queue_length() > 0:
            return self.filenames[0]
        else:
            logging.error("Odd, no more items in queue.")
            return ""

    def queue_length(self) -> int:
        return len(self.filenames)

    def add_to_queue(self, filename:str | list) -> None:
        logging.debug("Queue Adding: %s", filename)

        if filename in self.filenames:
            msg = f"{filename} already in queue: {self.filenames}, but still added"
            logging.error(msg)

        if isinstance(filename, str):
            self.filenames = [filename] + self.filenames
        elif isinstance(filename, list):
            self.filenames = filename + self.filenames
        else:
            logging.error("Queue Add, Unknown type: %s", filename)

    def remove_from_queue(self, filename:str) -> bool:
        found:bool = False
        if filename in self.filenames:
            logging.debug("Queue Remove: %s", filename)
            found = True
            self.filenames.remove(filename)
        else:
            logging.error("%s not in queue!", filename)

        return found

    def set_current_subprocess(self, p: subprocess.Popen) -> None:
        if self.current_subprocess is not None:
            logging.error("subprocess not completed")
        else:
            self.current_subprocess = p

    def get_current_subprocess(self) -> subprocess.Popen:
        return self.current_subprocess

    def completed_current_subprocess(self) -> None:
        self.current_subprocess = None

    def check_if_server_is_running(self) -> bool:
        """ Check if server can be reached """
        whisper_server_url:str = self.settings.value("Whisper/WhisperASRwebservice_URL")
        model:str = self.settings.value("Whisper/Engine")

        server_responded:bool = False
        if whisper_server_url != "" and model == "whisper_asr_webservice":

            hostname = urllib.parse.urlparse(whisper_server_url).hostname
            port = urllib.parse.urlparse(whisper_server_url).port

            response:int = -1
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                if port is None:
                    if whisper_server_url.startswith("http:"):
                        response = sock.connect_ex((hostname, 80))
                    else:
                        response = sock.connect_ex((hostname, 443))
                else:
                    response = sock.connect_ex((hostname, int(port)))
            except socket.timeout:
                logging.exception("socket timeout.")
            except socket.error:
                logging.exception("socket error.")
            else:
                if response == 0:
                    server_responded = True
                    logging.info("socket responded, whisper_asr_webservice running")
                else:
                    logging.error("Unknown response.")

            if not server_responded:
                if port is None:
                    msg = f"\nUnable to access Whisper ASR webserver: '{hostname}'."
                else:
                    msg = f"\nUnable to access Whisper ASR webserver: '{hostname}:{port}'."
                self.form_widget.feedback(msg)
                logging.error(msg)

        return server_responded

    def select_files(self) -> None:
        self.clear_queue()

        self.active_folder = self.settings.value("Folder/Active")
        if self.active_folder is None or not os.path.isdir(self.active_folder):
            self.active_folder = os.path.expanduser("~")

        model = self.settings.value("Whisper/Engine")
        acceptable_extensions = self.acceptable_extensions()

        selectable_files = ""
        for extension in acceptable_extensions:
            if extension in [".mp3", ".mp4", ".wav", ".aiff"]:
                if selectable_files == "":
                    selectable_files += f"Audio Files *{extension}"
                else:
                    selectable_files += f" ;; Audio Files *{extension}"
            else:
                if selectable_files == "":
                    selectable_files += f"Video Files *{extension}"
                else:
                    selectable_files += f" ;; Video Files *{extension}"

        files, selected_filter = QFileDialog.getOpenFileNames(self, caption = "Open Audio/Video File(s)",
            directory = self.active_folder,
            filter = selectable_files, initialFilter=self.selected_filter)

        self.selected_filter = selected_filter

        self.add_to_queue(files)

        if self.queue_length() > 0:
            self.files_received()

    def files_received(self) -> None:
        self.active_folder:str = os.path.dirname(self.first_in_queue())

        # save active folder to settings
        if self.settings.contains("Folder/Active"):
            self.settings.setValue("Folder/Active", self.active_folder)
        else:
            self.settings.beginGroup("Folder")
            self.settings.setValue("Active", self.active_folder)
            self.settings.endGroup()

        # enable start button
        self.form_widget.button2.setEnabled(True)
        self.form_widget.button1.setDefault(False)
        self.form_widget.button2.setDefault(True)

        # enable start menu
        self.start_action.setEnabled(True)

        # show files
        self.show_selected_files()

    def show_selected_files(self):
        self.form_widget.delete_feedback()
        msg:str = f"Selected Files: {self.queue_length()}"
        self.form_widget.feedback(msg)

        status_str:str = ""
        for fpath in self.queue():
            _, the_file = app_utils.split_path_file(fpath)
            status_str = f"{status_str}<br>- '{the_file}'"
            #status_str = f"{status_str}<br>- <a href=file:///{fpath}>{the_file}</a>"
        status_str = f"{status_str}<br>"
        self.form_widget.feedback(status_str)

    def do_process_files(self) -> None:
        self.tic = time.perf_counter()

        QCoreApplication.processEvents()

        # disable gui elements during processing
        self.form_widget.set_enabled_gui_elements(False)

        self.STOP = False
        self.overwrite_all_existing_files = False
        self.not_overwrite_all_existing_files = False
        self.insecure_server_ok_all = False
        
        self.set_status(Status.PROCESSING)

        self.process_files()

    def process_files(self) -> None:
        """
        Process each file
        """
        if self.queue_length() > 0 and not self.STOP:
            mime:str = "audio/mpeg"
            output_file_extension:str = ".txt"
            name:str

            # first one
            fpath = self.first_in_queue()
            the_path, the_file = app_utils.split_path_file(fpath)

            language = app_utils.lang_to_code(self.settings.value("Settings/Language"))
            if language == "auto":
                duration:float = app_utils.get_audio_duration(fpath)
                if duration > 0 and duration < 30:
                    self.form_widget.feedback("Automatically detecting the spoken language requires a clip of 30 seconds or longer. Specify the language of the speech rather than using 'Auto Detect'.\n")
                    msg = f"Duration of file '{the_file}' is {duration:.2} seconds.\n"
                    self.form_widget.feedback(msg)
                    logging.info(msg)
                    self.STOP = True

            if not self.STOP:
                logging.debug("the_path: %s", the_path)
                msg = f"Processing file: '{the_file}'"
                logging.info(msg)
                self.form_widget.feedback(msg)

                # output filename
                outputSelected = self.settings.value("Settings/Output")
                if outputSelected == "VTT":
                    output_file_extension = ".vtt"
                elif outputSelected == "SRT":
                    output_file_extension = ".srt"
                elif outputSelected == "JSON":
                    output_file_extension = ".json"
                elif outputSelected == "TSV":
                    output_file_extension = ".tsv"
                elif outputSelected == "LRC":
                    output_file_extension = ".lrc"
                elif outputSelected == "TEXT":
                    output_file_extension = ".txt"

                name = the_file.split(".")[0]
                output_filename = os.path.join(the_path, f"{name}{output_file_extension}")

                if self.settings.value("Whisper/Engine") == "whisper":

                    self.whisper_engine.run(the_path, fpath, output_filename, name, output_file_extension, mime)

                elif self.settings.value("Whisper/Engine") == "mlx-whisper":

                    self.mlx_whisper_engine.run(the_path, fpath, output_filename, name, output_file_extension, mime)

                elif self.settings.value("Whisper/Engine") == "whisper.cpp":

                    self.whispercpp_engine.run(fpath, output_filename, name, output_file_extension)

                elif self.settings.value("Whisper/Engine") == "whisper_asr_webservice":

                    self.whisper_webservice_engine.run(fpath, output_filename, name, output_file_extension, mime)

                elif self.settings.value("Whisper/Engine") == "whisper.api":

                    self.whisper_api_engine.run(fpath, output_filename, name, output_file_extension, mime)

                elif self.settings.value("Whisper/Engine") == "faster-whisper":

                    self.faster_whisper_engine.run(the_path, fpath, output_filename, name, output_file_extension, mime)

        if self.STOP:
            self.finished_processing("", False, cancelled=True)

    def convert_input_file_if_needed(self, fpath:str) -> tuple[bool, bool, bool, str]:
        err:bool = False
        need_conversion:bool = False
        already_exist:bool = False
        target_format:str = 'wav'
        msg:str

        the_path, the_file = app_utils.split_path_file(fpath)
        if "." in the_file:
            the_file = the_file.split(".")[0]
        out_filename = f"{the_file}.{target_format}"
        converted_fpath = os.path.join(the_path, out_filename)

        if self.settings.value("Whisper/Engine") == "whisper":
            # convert to 'wav'
            if app_utils.check_acceptable_file(self, fpath, [".wav", ".mp3", ".mp4", ".m4a", ".aiff", ".mpeg", ".mov", ".avi", ".wmv", ".webm"]):
                if app_utils.check_acceptable_file(self, fpath, [".wav"]):
                    ok_format, err = app_utils.correct_wav_file(fpath)

                    if not err:
                        if not ok_format:
                            need_conversion = True
                            err = self.convert_input_file_format(fpath, target_format)
                    else:
                        need_conversion = True
                        err = self.convert_input_file_format(fpath, target_format)
                else:
                    if os.path.isfile(converted_fpath):
                        if not app_utils.correct_wav_file(converted_fpath):
                            # existing file not in corrected format
                            logging.info("Existing wav file not in correct format")
                            need_conversion = True
                            err = self.convert_input_file_format(fpath, target_format)
                        else:
                            already_exist = True
                            msg = f"Converted file: '{out_filename}' already exists."
                            self.form_widget.feedback(msg)
                            self.add_to_queue(converted_fpath)
                    else:
                        need_conversion = True
                        err = self.convert_input_file_format(fpath, target_format)

                #self.finished_processing(fpath, err, conversion = True)
        elif self.settings.value("Whisper/Engine") == "whisper.cpp":
            # convert to 'wav'
            if app_utils.check_acceptable_file(self, fpath, [".wav", ".mp3", ".mp4", ".m4a", ".aiff", ".mpeg", ".mov", ".avi", ".wmv", ".webm"]):
                if app_utils.check_acceptable_file(self, fpath, [".wav"]):
                    ok_format, err = app_utils.correct_wav_file(fpath)

                    if not err:
                        if not ok_format:
                            need_conversion = True
                            err = self.convert_input_file_format(fpath, target_format)
                    else:
                        need_conversion = True
                        err = self.convert_input_file_format(fpath, target_format)
                else:
                    if os.path.isfile(converted_fpath):
                        if not app_utils.correct_wav_file(converted_fpath):
                            # existing file not in corrected format
                            need_conversion = True
                            logging.info("Existing wav file not in correct format")
                            err = self.convert_input_file_format(fpath, target_format)
                        else:
                            already_exist = True
                            msg = f"Converted file: '{out_filename}' already exists."
                            self.form_widget.feedback(msg)
                            logging.info(msg)
                            self.add_to_queue(converted_fpath)
                    else:
                        need_conversion = True
                        err = self.convert_input_file_format(fpath, target_format)

        return err, already_exist, need_conversion, converted_fpath

    def convert_input_file_format(self, fpath:str, target_format: str='wav') -> bool:
        err:bool = False
        converted_file_exists:bool = False
        final_out_fpath:str = ""

        the_path, the_file = app_utils.split_path_file(fpath)

        if "." in the_file:
            the_file = the_file.split(".")[0]
        out_filename = f"{the_file}.{target_format}"
        out_fpath = os.path.join(the_path, out_filename)

        if os.path.isfile(out_fpath):

            # original file
            the_path, the_file = app_utils.split_path_file(out_fpath)
            full_filename:str = the_file
            if "." in the_file:
                the_file = the_file.split(".")[0]

            final_out_filename = f"{the_file}.{target_format}"
            final_out_fpath = os.path.join(the_path, out_filename)

            # if file already exists add '_converted' to avoid overwriting original file
            if full_filename == final_out_filename:
                final_out_filename = f"{the_file}_converted.{target_format}"
                final_out_fpath = os.path.join(the_path, final_out_filename)

            if os.path.isfile(final_out_fpath):
                if not self.overwrite_all_existing_files and not self.not_overwrite_all_existing_files:

                    response = self.form_widget.ask_to_overwrite(final_out_filename, self.queue_length())

                    if response == QMessageBox.StandardButton.No:
                        converted_file_exists = True
                    if response == QMessageBox.StandardButton.NoToAll:
                        converted_file_exists = True
                        self.not_overwrite_all_existing_files = True
                    if response == QMessageBox.StandardButton.YesToAll:
                        self.overwrite_all_existing_files = True
                
                if self.not_overwrite_all_existing_files:
                    # do not convert
                    converted_file_exists = True

        if not converted_file_exists:
            is_wav = app_utils.check_acceptable_file(self, fpath, ['.wav'])
            ffmpeg_fpath = self.settings.value("FFmpeg/path")

            if ffmpeg_fpath != "" or is_wav:
                wav_info:str = ""
                if target_format == "wav":
                    wav_info = "(16kHz 16-bit mono)"

                msg = f"Converting file '{the_file}' to '{target_format}' format {wav_info}..."
                self.form_widget.feedback(msg)
                logging.info(msg)

                data = {
                    'original_file_fpath': fpath,
                    'ffmpeg_fpath': ffmpeg_fpath,
                    'target_format': target_format,
                    'final_file_fpath': "",
                    'is_wav': is_wav
                }

                self.worker = ConvertWorker(data)
                self.worker.finished.connect(self.handle_finished_convert_worker)
                self.worker.output.connect(self.feedback)
                #self.worker.finished.connect(self.worker.deleteLater)

                self.set_status(Status.CONVERTING)

                self.form_widget.set_show_progress_element(True)
                QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)

                logging.debug("Starting Convert Worker thread")
                self.worker.start()

            else:
                logging.error("FFmpeg not installed. Unable to convert file to correct 'wav' format (16kHz, 16-bit, mono).")
                msg = "<html><br><br>Unable to convert file to correct 'wav' format (16kHz, 16-bit, mono) because <a href='https://ffmpeg.org'>FFmpeg</a> is not installed.\
                       Use, for example, <a href='https://www.audacityteam.org'>Audacity</a> to convert file to 'wav' format (16kHz, 16-bit, mono).</html>"
                self.form_widget.show_info(msg)
                err = True
        else:
            logging.info("Converted file already exists.")
            data = {
                'original_file_fpath': fpath,
                'final_file_fpath': final_out_fpath,
                'err': err
            }
            self.handle_finished_convert_worker(data)
        return err

    def handle_finished_convert_worker(self, data) -> None:
        """ Finished after Completing Convert Worker """
        original_file_fpath = data['original_file_fpath']
        final_file_fpath = data['final_file_fpath']
        err = data['err']

        # set status to processing to continue processing of files in queue
        self.set_status(Status.PROCESSING)

        self.convert_worker_shutdown()
        QApplication.restoreOverrideCursor()

        if not err:
            # add new path to filenames
            self.add_to_queue(final_file_fpath)

        # continue
        self.finished_processing(original_file_fpath, err)

    def convert_worker_shutdown(self) -> None:
        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    self.worker.quit()
                    self.worker.wait()
                self.worker.deleteLater()
                self.worker = None
            except Exception as e:
                msg = f"Error during convert worker shutdown: {e}"
                logging.exception(msg)
        else:
            logging.info("Convert worker is not initialized or already cleaned up.")

    def finished_processing(
            self,
            fpath: str,
            err: bool,
            conversion: bool = False,
            cancelled:bool = False,
            error_msg:str = ""
        ) -> None:
        """ Update interface after complete one or all threads """
        if not err and not cancelled:
            logging.debug("Ending and removing from queue: %s", fpath)

            #if fpath in self.queue():
            self.remove_from_queue(fpath)

            #remaining_threads = len(self.my_threads)
            remaining_files = self.queue_length()

            if not conversion:
                if remaining_files > 0:

                    if remaining_files > 1:
                        status_str = f"\nStill processing {remaining_files} files:\n"
                    else:
                        status_str = f"\nStill processing {remaining_files} file:\n"

                    for n, fpath in enumerate(self.queue()):
                        _, the_file = app_utils.split_path_file(fpath)
                        if n == 0:
                            status_str = f"{status_str}'{the_file}'"
                        else:
                            status_str = f"{status_str}, '{the_file}'"
                    status_str = f"{status_str}\n"

                    self.form_widget.feedback(status_str)
                    msg = f"Still processing {remaining_files} files"
                    logging.info(msg)

            if remaining_files == 0 and not cancelled:
                self.notify_finished(conversion)
            else:
                if not conversion and not cancelled:
                    # process next file
                    if not self.STOP:
                        self.process_files()
                    else:
                        self.finished_processing("", False, cancelled=True)

        elif cancelled:
            self.feedback("\nProcess has been cancelled.")
            logging.info("Process has been cancelled.")

            self.reset_after_err_or_cancelled()
        else:
            if error_msg != "":
                self.feedback(error_msg)
                logging.error(error_msg)
            else:
                msg = f"Error and now reset: {fpath}"
                logging.info(msg)
                self.feedback("\nAn error has occurred.")
            
            self.reset_after_err_or_cancelled(err)

    def reset_after_err_or_cancelled(self, err:bool=False) -> None:
        self.form_widget.set_enabled_gui_elements(True)
        # Buttons
        self.form_widget.button1.setDefault(True)
        self.form_widget.button2.setEnabled(False)
        # Menu
        self.start_action.setEnabled(False)

        # reset text of cancel_button
        self.form_widget.cancel_button.setText("Cancel")
        self.form_widget.cancel_button.setEnabled(True)
        self.form_widget.cancel_button.update()

        # hide processing bar
        self.form_widget.set_show_progress_element(False)

        # stop threads
        self.clear_queue()

        self.set_status(Status.IDLE)

    def notify_finished(self, conversion: bool = False) -> None:
        """ Report how long the process took and send desktop notification """
        logging.debug("notify finished")

        # report how it took
        self.clear_queue()
        toc:float = time.perf_counter()
        duration:float = toc - self.tic

        if not conversion and self.get_status() != "STOPPED":
            time_str = time.strftime('%Hh%Mm%Ss', time.gmtime(duration))
            human_readable_time = app_utils.duration_str(duration)
            if human_readable_time != "":
                msg = f"Finished processing file(s) in {app_utils.duration_str(duration)} ({time_str})"
            else:
                msg = f"Finished processing file(s) in {time_str}."
            
            self.form_widget.feedback(f"<br>{msg}")
            logging.info(msg)

            folder = self.settings.value("Folder/Active")
            self.form_widget.feedback(f"<br>Output file(s) can be found in the folder: '<b>{folder}</b>'<br>")

            # reset cancel button
            self.form_widget.cancel_button.setText("Cancel")
            self.form_widget.cancel_button.setEnabled(True)
            self.form_widget.cancel_button.update()

            # hide processing bar
            self.form_widget.set_show_progress_element(False)

            # enable GUI elements
            self.form_widget.set_enabled_gui_elements(True)
            self.form_widget.button1.setDefault(True)
            self.form_widget.button2.setEnabled(False)
            # menu
            self.start_action.setEnabled(False)

            # only report when whisper has finished
            app_utils.desktop_notification(f"{self.APP_NAME}", "Completed processing file(s).")

            self.set_status(Status.IDLE)

        if self.get_status() == "STOPPED":
            # hide processing bar
            self.form_widget.set_show_progress_element(False)

            # enable GUI elements
            # buttons
            self.form_widget.set_enabled_gui_elements(True)
            self.form_widget.button1.setDefault(True)
            self.form_widget.button2.setEnabled(False)
            # menu
            self.start_action.setEnabled(False)

            self.set_status(Status.IDLE)

    def show_app_info(self, start_up = True) -> None:

        msg:str = "This application provides a simple graphical user interface for different automatic speech recognition (ASR) systems and services based on \
                    <a href='https://openai.com/research/whisper'>OpenAI's Whisper</a> (<a href='https://github.com/ggerganov/whisper.cpp'>whisper.cpp</a>, \
                    <a href='https://github.com/ml-explore/mlx-examples/tree/main/whisper'>mlx-whisper</a>, \
                    <a href='https://github.com/guillaumekln/faster-whisper'>faster-whisper</a>,  \
                    <a href='https://github.com/ahmetoner/whisper-asr-webservice'>whisper ASR webservice</a>, and \
                    the <a href='https://openai.com/blog/introducing-chatgpt-and-whisper-apis'>whisper API</a>).<br><br>\
                    Speech2Text transcribes the speech from audio and video files. The output is a text file or a subtitle file (.vtt or .srt). \
                    When you select OpenAI's Whisper, mlx-whisper, whisper.cpp, \
                    or faster-whisper, the ASR runs locally on your computer. <br><br> \
                    Please note that mlx-whisper, which requires Mac with an M1, M2, or later, and whisper.cpp are much faster than OpenAI's whisper implementation. \
                    Speech2Text can also send the audio to a remote computer running the \
                    whisper ASR webservice or use OpenAI's whisper API, which performs ASR on OpenAI's servers.<br><br>\
                    To achieve the best accuracy, select one of the 'large' models in the Settings (e.g. large-v2 or large-v3-turbo).\
                    "

        ffmpeg_fpath = self.settings.value("FFmpeg/path")

        if not start_up:

            msg = f"<a href='{self.WEBSITE_URL}'>{self.APP_NAME}</a> (v{self.VERSION}) created by {self.AUTHOR}. <br><br>\
                       This application provides a simple graphical user interface for different automatic speech recognition (ASR) systems and services based on  \
                    <a href='https://openai.com/research/whisper'>OpenAI's Whisper</a> (<a href='https://github.com/ggerganov/whisper.cpp'>whisper.cpp</a>, \
                    <a href='https://github.com/ml-explore/mlx-examples/tree/main/whisper'>mlx-whisper</a>, \
                    <a href='https://github.com/guillaumekln/faster-whisper'>faster-whisper</a>, \
                    <a href='https://github.com/ahmetoner/whisper-asr-webservice'>whisper ASR webservice</a>, and \
                    the <a href='https://openai.com/blog/introducing-chatgpt-and-whisper-apis'>whisper API</a>).<br><br>"

            # whisper
            if self.whisper_engine:
                import whisper
                msg = f"{msg}OpenAI's whisper (v{whisper.__version__}).<br>"

            # whisper.cpp
            if self.whispercpp_engine:
                if platform.system() == "Windows":
                    msg = f"{msg}whisper.cpp (v1.7.4, d682e15).<br>"
                else:
                    msg = f"{msg}whisper.cpp (v1.7.4, d682e15).<br>"

            # faster-whisper
            if self.faster_whisper_engine:
                import faster_whisper
                msg = f"{msg}faster-whisper (v{faster_whisper.__version__}).<br>"

            if self.mlx_whisper_engine:
                import mlx_whisper
                import mlx.core as mx
                msg = f"{msg}mlx (v{mx.__version__}).<br>"
                msg = f"{msg}mlx-whisper (v{mlx_whisper.__version__}).<br>"

            # openai api
            if self.whisper_api_engine:
                import openai
                msg = f"{msg}openai api (v{openai.version.VERSION}).<br>"

            msg = f"{msg}torch (v{torch.__version__})"

            if app_utils.cuda_available():
                msg = f"{msg}<br>CUDA available.<br>"

            if ffmpeg_fpath != "":
                msg = f"{msg}<br><br><a href='https://ffmpeg.org'>FFmpeg</a> path: '{ffmpeg_fpath}'."

        if ffmpeg_fpath == "":
            msg = f"{msg}<br><br>Please note that <a href='https://ffmpeg.org'>FFmpeg</a> is not installed on this computer. \
                    FFmpeg is required for OpenAI's whisper and mlx-whisper as well as for converting video/audio files. \
                    Install FFmpeg using <a href='https://brew.sh'>brew</a> on macOS \
                    and <a href='https://scoop.sh'>Scoop</a> or <a href='https://chocolatey.org'>Chocolatey</a> on Windows."

            msg = f"{msg}<br><br>Without FFmpeg, whisper.cpp and can only process 'wav' files. \
                    FFmpeg is not required for faster-whisper, whisper ASR webservice, and the OpenAI's Whisper API."

        self.form_widget.delete_feedback()
        self.form_widget.show_info(msg)

    def feedback(self, msg:str):
        self.form_widget.feedback(msg)

    def open_output_folder(self) -> None:
        folder = self.settings.value("Folder/Active")
        url = QUrl.fromLocalFile(folder)
        QDesktopServices.openUrl(url)

    def output_items(self) -> list[str]:
        engine = self.settings.value("Whisper/Engine")
        if engine == "whisper.cpp":
            # only whisper.cpp has TSV option
            output_items = ["VTT", "SRT", "JSON", "TSV", "LRC", "TEXT"]
        elif engine == "faster-whisper":
            output_items = ["VTT", "SRT", "TEXT"]
        else:
            output_items = ["VTT", "SRT", "JSON", "TEXT"]

        return output_items

    def show_settings_dlg(self) -> None:
        """ show settings window """
        dlg = SettingsDialog(self)
        dlg.exec()

    def update_output_setting(self) -> None:
        self.settings.setValue("Settings/Output", self.form_widget.comboOutput.currentText())

    def update_language_setting(self) -> None:
        self.settings.setValue("Settings/Language", self.form_widget.comboLanguage.currentText())

    def update_task_options(self) -> None:
        engine: str = self.settings.value("Whisper/Engine")
        if engine == 'faster-whisper':
            # faster-whisper has only option to transcribe
            self.form_widget.set_tasks('Transcribe')
        else:
            # transcribe and translate
            self.form_widget.set_tasks('default')
        self.settings.setValue("Settings/Task", self.form_widget.comboTask.currentText())

    def delete_downloaded_models(self) -> None:
        # delete all downloaded model files
        # useful when update involves updated weights
        logging.info("Deleting all model files")

        model_file_fpath = os.path.join(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation),
            "Models_whisper"
        )

        if os.path.isdir(model_file_fpath):
            try:
                shutil.rmtree(model_file_fpath)
            except OSError:
                msg = f"OSError when deleting folder: {model_file_fpath}"
                self.mainWindow.form_widget.feedback(msg)
                logging.exception(msg)
            else:
                os.mkdir(model_file_fpath)
        else:
            os.mkdir(model_file_fpath)

        # remove models Whisper.cpp
        model_file_fpath = os.path.join(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation),
            "Models_whisper_cpp"
        )

        if os.path.isdir(model_file_fpath):
            try:
                shutil.rmtree(model_file_fpath)
            except OSError:
                msg = f"OSError when deleting folder: {model_file_fpath}"
                self.mainWindow.form_widget.feedback(msg)
                logging.exception(msg)
            else:
                os.mkdir(model_file_fpath)
        else:
            os.mkdir(model_file_fpath)

        # remove models faster-whisper
        model_file_fpath = os.path.join(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation),
            "Models_faster-whisper"
        )

        if os.path.isdir(model_file_fpath):
            try:
                shutil.rmtree(model_file_fpath)
            except OSError:
                msg = f"OSError when deleting folder: {model_file_fpath}"
                self.mainWindow.form_widget.feedback(msg)
                logging.exception(msg)
            else:
                os.mkdir(model_file_fpath)
        else:
            os.mkdir(model_file_fpath)

        # remove models MLX Whisper
        model_file_fpath = os.path.join(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation),
            "Models_MLX_Whisper"
        )

        if os.path.isdir(model_file_fpath):
            try:
                shutil.rmtree(model_file_fpath)
            except OSError:
                msg = f"OSError when deleting folder: {model_file_fpath}"
                self.mainWindow.form_widget.feedback(msg)
                logging.exception(msg)
            else:
                os.mkdir(model_file_fpath)
        else:
            os.mkdir(model_file_fpath)
