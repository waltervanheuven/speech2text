"""
Speech2Text - GUI for different ASR implementations
"""

# Copyright © 2023-2025 Walter van Heuven

import sys
import os
import ssl
import logging
import multiprocessing
import platform
import warnings
import certifi
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QStandardPaths, Qt
import utils as app_utils
import mainwindow

__version__: str = "2.3.0"
__version_url__: str = 'https://waltervanheuven.net/s2t/version.txt'
__website_url__: str = "https://waltervanheuven.net/s2t/"
__author__: str = "Walter van Heuven"

_APP_NAME: str = "Speech2Text"

def get_cert_path():
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle
        return os.path.join(sys._MEIPASS, 'certifi', 'cacert.pem')
    else:
        # Running in a normal Python environment
        return certifi.where()

def main():
    # crucial next line to make sure whisperccp_engine is working when building
    # app with PyInstaller
    multiprocessing.freeze_support()
    os.environ['SSL_CERT_FILE'] = get_cert_path()
    ssl._create_default_https_context = ssl.create_default_context

    app: QApplication = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName(_APP_NAME)
    if platform.system() == "Windows":
        # Hide icons in Menu
        app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, True)

    _DEBUG: bool = False
    if len(sys.argv) > 1:
        # from command line add 'True' to set DEBUG
        _DEBUG = app_utils.bool_value(sys.argv[1])

    if not _DEBUG:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)

    try:
        app_data_location:str = QStandardPaths.writableLocation(
                                    QStandardPaths.StandardLocation.AppDataLocation
                                )
        if not os.path.isdir(app_data_location):
            os.mkdir(app_data_location)

        log_path:str = os.path.join(app_data_location, f"{_APP_NAME}.log")

        if _DEBUG:
            logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode="w+")
        else:
            logging.basicConfig(filename=log_path, level=logging.INFO, filemode="w+")

        window:QMainWindow = mainwindow.MainWindow(_APP_NAME, __version__, __version_url__, __website_url__, __author__, _DEBUG)
        window.show()

        if _DEBUG:
            logging.info("Debug ON")

        sys.exit(app.exec())
    except OSError as e:
        print(f"Unable to create log file: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
