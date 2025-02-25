""" Whisper ASR webservice Engine """

# Copyright © 2023-2025 Walter van Heuven

import os
import logging
import requests
import validators
from pydantic import BaseModel, ConfigDict
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget, QHBoxLayout
import utils as app_utils

class WhisperWebserviceData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    audio_file: str = ""
    file_size_str: str = ""
    mime: str = ""
    whisper_server_url: str = ""
    params: dict | None = None
    headers: dict | None = None
    language: str | None = None
    response: requests.Response | None = None
    task: str = "transcribe"
    outputfilename: str = ""
    err: bool = False

class Worker(QThread):
    finished = pyqtSignal(WhisperWebserviceData)
    output = pyqtSignal(str)

    def __init__(self, data:WhisperWebserviceData = None) -> None:
        super().__init__()
        self.setObjectName("WhisperWebservice")
        self.data:WhisperWebserviceData = data

    def run(self) -> None:
        err:bool = False

        err = self.transcribe()
        self.data.err = err

        # emit results
        self.finished.emit(self.data)

    def transcribe(self) -> bool:
        err:bool = False

        with open(self.data.audio_file, 'rb') as b:
            files:dict = {
                'audio_file': b,
                'type' : self.data.mime,
                'task' : self.data.task
            }

            try:
                with requests.Session() as s:
                    response:requests.Response = s.post(
                        self.data.whisper_server_url,
                        params=self.data.params,
                        headers=self.data.headers,
                        files=files
                    )

                self.data.response = response

            except requests.exceptions.ConnectTimeout:
                logging.exception("ConnectTimeout")
                msg = "\nUnable to access server.\nPlease check URL and internet connection."                
                self.output.emit(msg)
                err = True
            except requests.exceptions.RequestException:
                logging.exception("RequestException")
                msg = "\nUnable to access server.\nPlease check URL and internet connection."
                self.output.emit(msg)
                err = True
            except TimeoutError:
                logging.exception("TimeoutError")
                msg = "\nUnable to access server.\nPlease check URL and internet connection."
                self.output.emit(msg)
                err = True
            except Exception as e:
                msg = f"Exception: {e}"
                logging.exception(msg)
                msg = "\nAn error occured when accessing server.\nPlease check URL and internet connection."
                self.output.emit(msg)
                err = True
        return err

class WhisperWebserviceEngine():

    def __init__(self, mainWindow):
        self.mainWindow = mainWindow
        self.settings = self.mainWindow.settings
        self.worker = None

    def run(self, audio_file: str, outputfilename: str, name: str, output_file_extension: str, mime: str) -> None:
        """
        Function called from WhisperWorker
        """
        err:bool = False
        msg:str

        headers = {
            'accept': 'application/json',
        }

        # remove . from output extension
        output = output_file_extension.replace(".", "")

        language = app_utils.lang_to_code(self.settings.value("Settings/Language"))
        if language.lower() == "auto":
            language = None

        task:str = self.settings.value("Settings/Task")

        params = {
            'task': task,
            'language': language,
            'encode': 'true',
            'output': output,
        }

        # file size
        file_size_bytes = os.path.getsize(audio_file)
        file_size_megabytes = file_size_bytes / (1024 * 1024)
        if file_size_megabytes > 1:
            file_size_str = f"{file_size_megabytes:.0f} MB"
        else:
            file_size_kilobytes = file_size_bytes / 1024
            file_size_str = f"{file_size_kilobytes:.0f} KB"

        whisper_server_url:str = self.settings.value("Whisper/WhisperASRwebservice_URL")
        if whisper_server_url == "" or not validators.url(whisper_server_url):
            logging.error("URL webservice '%s' not valid", whisper_server_url)
            self.feedback(f"Unable to access server, the URL '{whisper_server_url}' invalid!")
            err = True
        
        # add '/asr/ if missing
        if not whisper_server_url.lower().endswith('/asr'):
            if whisper_server_url.endswith('/'):
                whisper_server_url = f"{whisper_server_url}asr"
            else:
                whisper_server_url = f"{whisper_server_url}/asr"

        if not err:
            data = WhisperWebserviceData(
                audio_file=audio_file,
                file_size_str=file_size_str,
                mime=mime,
                whisper_server_url=whisper_server_url,
                params=params,
                headers=headers,
                language=language,
                response=None,
                task=task,
                outputfilename=outputfilename,
                err=False
            )

            if whisper_server_url.startswith("http:"):
                if not self.mainWindow.insecure_server_ok_all:
                    msg = f"Please be aware that the connection to\n'{whisper_server_url}' is unencrypted.\n\nContinue?"
                    dlg = CustomMessageBox(self.mainWindow)
                    dlg.setIcon(QMessageBox.Icon.Critical)
                    dlg.setWindowModality(Qt.WindowModality.WindowModal)
                    dlg.setText(msg)
                    if self.mainWindow.queue_length() > 1:
                        dlg.setStandardButtons(
                            QMessageBox.StandardButton.Yes |
                            QMessageBox.StandardButton.YesAll |
                            QMessageBox.StandardButton.No
                        )
                    else:
                        dlg.setStandardButtons(
                            QMessageBox.StandardButton.Yes |
                            QMessageBox.StandardButton.No
                        )
                    dlg.setDefaultButton(QMessageBox.StandardButton.No)

                    button = dlg.exec()

                    if button == QMessageBox.StandardButton.Yes or button == QMessageBox.StandardButton.YesToAll:

                        if button == QMessageBox.StandardButton.YesToAll:
                            self.mainWindow.insecure_server_ok_all = True

                        self.start_worker(data)
                    else:
                        self.mainWindow.finished_processing(audio_file, err=False, cancelled=True)
                else:
                    self.start_worker(data)

            elif whisper_server_url.startswith("https:"):
                self.start_worker(data)
        else:
            data = WhisperWebserviceData(
                audio_file=audio_file,
                err=err
            )
            self.handle_finished(data)

    def start_worker(self, data:WhisperWebserviceData):
        file_size_str = data.file_size_str
        whisper_server_url = data.whisper_server_url

        msg:str = f"Sending file ({file_size_str}) to: '{whisper_server_url}', waiting...<br>(this may take a while, no feedback)"
        self.feedback(msg)
        msg = f"Sending file ({file_size_str}) to: '{whisper_server_url}':"
        logging.info(msg)

        self.worker = Worker(data)
        self.worker.setObjectName("WhisperWebservice")

        # function to run when finished
        self.worker.finished.connect(self.handle_finished)
        #self.worker.finished.connect(self.worker.deleteLater)

        # function to print string
        self.worker.output.connect(self.feedback)

        # show processing bar
        self.mainWindow.form_widget.set_show_progress_element(True)

        # show busycursor
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)

        # start thread
        self.worker.start()

    def handle_finished(self, data:WhisperWebserviceData) -> None:
        err:bool = data.err
        audio_file = data.audio_file

        if not err:
            response = data.response
            outputfilename = data.outputfilename
            audio_file = data.audio_file
            msg:str = ""

            if (response is not None and response.ok):

                text:str = response.text
                    
                if not text.startswith("<!doctype html>") and not text.startswith("<html>"):
                    # save received file
                    with open(outputfilename, 'w', encoding='utf-8') as f:
                        f.write(text)

                    msg = f"\nSaved output to file: '{outputfilename}'"
                    self.feedback(msg)
                else:
                    msg = "\nUnexpected response. Please check URL."
                    logging.error(msg)
                    self.feedback(msg)
                    err = True
            else:
                if response is not None:
                    msg_log = f"Error posting request: {response.status_code}"
                    logging.error(msg_log)
                    msg = f"\nError posting request: {response.status_code}\nPlease check URL"
                    self.feedback(msg)
                    err = True

        self.worker_shutdown()
        QApplication.restoreOverrideCursor()

        self.mainWindow.finished_processing(audio_file, err)

    def worker_shutdown(self) -> None:
        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    self.worker.quit()
                    self.worker.wait()
                self.worker.deleteLater()
                self.worker = None
            except Exception as e:
                msg = f"Error during worker shutdown: {e}"
                logging.exception(msg)
        else:
            logging.info("Worker is not initialized or already cleaned up.")

    def feedback(self, msg):
        self.mainWindow.form_widget.feedback(msg)

class CustomMessageBox(QMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowModality(Qt.WindowModality.WindowModal)

        button_widget = QWidget(self)
        button_layout = QHBoxLayout(button_widget)

        button_list = []
        for button in self.buttons():
            button_list.append(button)
            self.removeButton(button)

        for b in button_list:
            button_layout.addWidget(b)

        layout = self.layout()
        layout.addWidget(button_widget, layout.rowCount(), 0, 1, layout.columnCount())
