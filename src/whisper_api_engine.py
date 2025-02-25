""" Whisper API Engine """

# Copyright © 2023-2025 Walter van Heuven

import os
import logging
import json
from pydantic import BaseModel, ConfigDict
import openai
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication
import utils as app_utils

class WhisperAPIData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: openai.Client | None
    audio_file: str = ""
    task_str: str = ""
    response_format: str = ""
    language: str | None = None
    transcript: dict | None
    outputfilename: str
    err: bool = False
    error_msg: str = ""
    interrupted: bool = False

class Worker(QThread):
    finished = pyqtSignal(WhisperAPIData)
    output = pyqtSignal(str)
    output_progress = pyqtSignal(str)

    def __init__(self, data:WhisperAPIData=None) -> None:
        super().__init__()
        self.setObjectName("WhisperAPI")
        self.data:WhisperAPIData = data

    def run(self) -> None:
        err:bool = False

        err = self.transcribe()

        self.data.err = err

        # emit results
        self.finished.emit(self.data)

    def transcribe(self) -> bool:
        err: bool = False
        msg: str = "An error has occurred."

        try:
            # large-v2 model: whisper-1
            audio_file = open(self.data.audio_file, "rb")

            client = self.data.client
            if self.data.task_str.lower() == "transcribe":
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format=self.data.response_format,
                    language=self.data.language,
                    temperature=0
                )
            else:
                # translate
                transcript = client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format=self.data.response_format,
                    language=self.data.language,
                    temperature=0
                )
            audio_file.close()

        except openai.AuthenticationError:
            msg = "Incorrect API key provided"
            logging.exception(msg)
            err = True
        except openai.APIConnectionError:
            msg = "Failed to connect to OpenAI API"
            logging.exception(msg)
            err = True
        except openai.RateLimitError:
            msg = "OpenAI API request exceeded rate limit"
            logging.exception(msg)
            err = True
        except openai.PermissionDeniedError:
            msg = "No permission to access to the requested resource"
            logging.exception(msg)
            err = True
        else:
            self.data.transcript = transcript

        if err:
            self.output.emit(msg)

        return err

class WhisperAPIEngine():

    def __init__(self, mainWindow) -> None:
        self.mainWindow = mainWindow
        self.settings = self.mainWindow.settings
        self.worker = None

    def run(self, audio_file: str, outputfilename: str, name: str, output_file_extension: str, mime: str) -> None:
        """
        Using OpenAI Whisper API
        https://openai.com/blog/introducing-chatgpt-and-whisper-apis

        requires OpenAI API key

        Function called from WhisperWorker
        """
        key = self.settings.value("Whisper/WhisperOpenAI_API")

        client = openai.OpenAI(
            api_key=key
        )

        err:bool = False
        msg:str

        if key != "":

            task_str:str = self.settings.value("Settings/Task")

            response_format:str = self.settings.value("Settings/Output").lower()
            language = app_utils.lang_to_code(self.mainWindow.form_widget.comboLanguage.currentText())
            if language.lower() == "auto":
                # auto detect language
                language = None

            # check if file does not exceed max length accepted by OpenAI API
            file_size_bytes = os.path.getsize(audio_file)
            file_size_megabytes = file_size_bytes / (1024 * 1024)

            if file_size_megabytes > 25:
                err = True
                msg = f"Filesize ({file_size_megabytes:.0f} MB) exceeds maximum upload size (25 MB) allowed by OpenAI API."
                #self.feedback(msg)
            else:
                if file_size_megabytes > 1:
                    file_size_str = f"{file_size_megabytes:.0f} MB"
                else:
                    file_size_kilobytes = file_size_bytes / 1024
                    file_size_str = f"{file_size_kilobytes:.0f} KB"
                msg = f"Sending file ({file_size_str}) to OpenAI servers, waiting...<br>(this may take a while, no feedback)"
                self.feedback(msg)
                msg = f"Sending file ({file_size_str}) to OpenAI"
                logging.info(msg)

            if not err:
                data = WhisperAPIData(
                    client=client,
                    audio_file=audio_file,
                    task_str=task_str,
                    response_format=response_format,
                    language=language,
                    transcript=None,
                    outputfilename=outputfilename
                )

                self.worker = Worker(data)
                self.worker.setObjectName("WhisperAPI")

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
            else:
                data = WhisperAPIData(
                    client=None,
                    audio_file=audio_file,
                    task_str="",
                    response_format="",
                    language="",
                    transcript=None,
                    outputfilename="",
                    err=err,
                    error_msg=msg
                )
                self.handle_finished(data)
        else:
            err = True
            msg = "Whisper API key missing."
            logging.error(msg)

            dlg = QMessageBox(self.mainWindow)
            dlg.setIcon(QMessageBox.Icon.Critical)
            dlg.setWindowModality(Qt.WindowModality.WindowModal)
            dlg.setText(msg)
            dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
            _ = dlg.exec()

            # create data with key variables
            data = WhisperAPIData(
                client=None,
                audio_file=audio_file,
                task_str="",
                response_format="",
                language="",
                transcript=None,
                outputfilename="",
                err=err,
                error_msg=msg
            )
            self.handle_finished(data)

    def handle_finished(self, data: WhisperAPIData) -> None:
        err = data.err
        audio_file = data.audio_file

        if not err:
            response_format = data.response_format
            outputfilename = data.outputfilename
            transcript = data.transcript

            if response_format != 'json':
                with open(outputfilename, 'w', encoding='utf-8') as f:
                    f.write(transcript)
            else:
                json_dump = json.dumps(transcript, indent=4)
                with open(outputfilename, 'w', encoding='utf-8') as f:
                    f.write(json_dump)

            self.feedback(f"\nSaved output to file: '{outputfilename}'")

        self.worker_shutdown()
        QApplication.restoreOverrideCursor()

        self.mainWindow.finished_processing(audio_file, err, error_msg=data.error_msg)

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

    def feedback(self, msg: str):
        self.mainWindow.form_widget.feedback(msg)
