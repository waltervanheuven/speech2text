""" Whisper Engine """

# Copyright © 2023-2025 Walter van Heuven

import os
import logging
import contextlib
from pydantic import BaseModel, ConfigDict
import whisper
import torch
from PyQt6.QtCore import QStandardPaths
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication
import utils as app_utils
from stream_emitter import StreamEmitter

class WhisperData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_str: str
    device: str
    fp16: bool
    model_dir_fpath: str
    task: str = "transcribe"
    model: whisper.Whisper | None = None
    filename: str = ""
    language: str | None = None
    result: dict | None = None
    outputfilename: str
    output_folder: str = ""
    err: bool = False
    err_msg: str = ""

class Worker(QThread):
    finished = pyqtSignal(WhisperData)
    output = pyqtSignal(str)
    output_progress = pyqtSignal(str)

    def __init__(self, data:WhisperData=None) -> None:
        super().__init__()
        self.setObjectName("WhisperEngine")

        self.stdout = StreamEmitter()
        self.stderr = StreamEmitter()
        self.stdout.message.connect(self.output.emit)
        self.stderr.message.connect(self.output_progress.emit)

        self.data: WhisperData = data

    def run(self) -> None:
        try:
            err: bool = False

            err = self.download_model()
            if not err:
                err = self.execute_task()
            else:
                logging.error("Unable to download model")

            self.data.err = err
        except RuntimeError as e:
            logging.exception("RuntimeErro in Worker")
            self.data.err = True
            self.data.error_msg = str(e)
        finally:
            self.finished.emit(self.data)

    def download_model(self) -> bool:
        err: bool = False
        msg: str

        model_file = f"{self.data.model_str}.pt"
        model_file_fpath = os.path.join(self.data.model_dir_fpath, model_file)

        model = None
        if not os.path.isfile(model_file_fpath):
            logging.info("Whisper model file not found: %s", model_file_fpath)

            try:
                msg = "Downloading model..."
                self.output.emit(msg)
                logging.info(msg)

                with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr), \
                     app_utils.modified_environ(HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_ENABLE_HF_TRANSFER='True', HF_HUB_DISABLE_SYMLINKS_WARNING='1'):
                    model = whisper.load_model(
                                self.data.model_str,
                                self.data.device,
                                self.data.model_dir_fpath
                            )

            except RuntimeError:
                logging.exception("RuntimeError")
                err = True
            except OSError:
                logging.exception("OSError")
                err = True

            if err:
                self.output.emit("An error occurred!")
            else:
                self.output.emit("Download completed.\n")
                self.data.model = model

        if not err:
            if not os.path.isfile(model_file_fpath):
                err = True
                logging.error("Model file does not exist!")
                self.output.emit("An error occurred!")

            elif os.path.getsize(model_file_fpath) == 0:
                err = True
                # something went wrong, remove it
                logging.debug("Something went wrong, os.remove")
                os.remove(model_file_fpath)
                msg = "Model file size is zero, try again."
                self.output.emit(msg)

            if not err:
                # only when using openai-whisper
                with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr), \
                     app_utils.modified_environ(HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_ENABLE_HF_TRANSFER='True', HF_HUB_DISABLE_SYMLINKS_WARNING='1'):
                    self.data.model = whisper.load_model(
                                            self.data.model_str,
                                            self.data.device,
                                            self.data.model_dir_fpath
                                        )

        return err

    def execute_task(self) -> bool:
        err: bool = False

        try:
            task = self.data.task
            msg = f"Starting {task}...\n"
            self.output.emit(msg)
            logging.info(msg)
            result = None
            if self.data.model is not None:
                # see https://github.com/openai/whisper/blob/main/whisper/transcribe.py
                options_dict = dict(
                    task =task.lower(),
                    language = self.data.language,
                    beam_size = 5, # default 5
                    best_of = 5,   # default 5
                    compression_ratio_threshold = 2.4, # default 2.4, lower might help with repetition
                    fp16 = self.data.fp16
                )
                with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
                    result = self.data.model.transcribe(
                                    self.data.filename,
                                    verbose=True,
                                    temperature = (0, 0.2, 0.4, 0.6, 0.8, 1),
                                    no_speech_threshold = 0.6, # default 0.6
                                    condition_on_previous_text = True, # default True
                                    **options_dict
                                )
            else:
                logging.error("Whisper model not available")
                err = True
        except Exception as e :
            logging.exception("Exception: %s", str(e))
            err = True
        else:
            self.data.result = result

        return err

class WhisperEngine():

    def __init__(self, mainWindow) -> None:
        self.mainWindow = mainWindow
        self.settings = self.mainWindow.settings
        self.worker = None

    def run(self, output_folder: str, filename: str, outputfilename: str, name: str, output_file_extension: str, mime: str) -> None:
        """
        Using OpenAI whisper
        https://github.com/openai/whisper

        """
        converted: bool
        err: bool

        # convert file if needed
        #converted, converted_filename = self.mainWindow.convert_input_file_if_needed(filename)
        err, already_exist, converted, fpath = self.mainWindow.convert_input_file_if_needed(filename)

        if not converted and not already_exist and not err:
            # remove extension for outputfilename
            the_folder, new_outputfilename, _ = app_utils.split_path_file(filename)
            new_outputfilename = os.path.join(the_folder, new_outputfilename)

            self.continue_processing(output_folder, filename, outputfilename, name, output_file_extension, mime)

        elif already_exist and not err:
            logging.debug("File %s already exists and already added to queue", fpath)
            self.mainWindow.finished_processing(filename, False)
        elif err:
            logging.error("An error occurred while trying to convert audio file")
            self.mainWindow.finished_processing(filename, err)

    def continue_processing(self, output_folder: str, filename: str, outputfilename:str,  name: str, output_file_extension: str, mime: str) -> None:

        fp16: bool = True
        device: str = "cuda" if app_utils.cuda_available() else "cpu"
        logging.debug("Device: %s", device)
        if device == "cpu":
            fp16 = False

        language = app_utils.lang_to_code(self.settings.value("Settings/Language"))
        if language.lower() == "auto":
            # auto detect language
            language = None

        # model_dir_fpath
        model_dir_fpath = os.path.join(os.path.abspath(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)), "Models_whisper")
        if not os.path.isdir(model_dir_fpath):
            os.mkdir(model_dir_fpath)

        # model_str
        model_str = self.settings.value("Settings/OpenAI_model")
        if model_str == "large":
            model_str = "large-v3"
        if model_str == "turbo":
            model_str = "large-v3-turbo"

        msg = f"Running whisper ({device.upper()}, model: '{model_str}')"
        self.feedback(msg)
        logging.info(msg)

        data = WhisperData(
            model_str=model_str,
            device=device,
            fp16=fp16,
            model_dir_fpath=model_dir_fpath,
            task=self.settings.value("Settings/Task"),
            model=None,
            filename=filename,
            language=language,
            result=None,
            output_folder=output_folder,
            outputfilename=outputfilename
        )

        self.worker = Worker(data)
        self.worker.setObjectName("Whisper")

        # function to run when finished
        self.worker.finished.connect(self.handle_finished)
        #self.worker.finished.connect(self.worker.deleteLater)

        # function to print string
        self.worker.output.connect(lambda x: self.feedback(x, True, False))
        self.worker.output_progress.connect(lambda x: self.feedback(x, False, True))

        # show processing bar
        self.mainWindow.form_widget.set_show_progress_element(True)

        # show busycursor
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)

        # start thread
        self.worker.start()

    def handle_finished(self, data:WhisperData) -> None:
        err = data.err
        filename = data.filename

        if not err:
            result = data.result
            output_folder = data.output_folder
            outputfilename = data.outputfilename

            output_type = self.settings.value("Settings/Output")
            output_type = output_type.lower()
            if output_type == "text":
                output_type = "txt"

            # write output file
            txt_writer = whisper.utils.get_writer(output_type, output_folder)

            if txt_writer is not None:
                txt_writer(result, outputfilename)

                self.feedback(f"\nSaved output to file: '{outputfilename}'")
                logging.info("Output: '%s'", outputfilename)
            else:
                logging.error("Unable access txt_writer")

        self.worker_shutdown()
        QApplication.restoreOverrideCursor()

        if not err:
            self.mainWindow.finished_processing(filename, err)
        else:
            self.mainWindow.finished_processing(filename, err, error_msg=data.err_msg)

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

    def whisper_models(self) -> list[str]:
        return whisper.available_models()

    def feedback(self, msg:str, add_newline:bool = True, check_progress_bar:bool = False) -> None:
        self.mainWindow.form_widget.feedback(msg, add_newline = add_newline, check_progress_bar=check_progress_bar)
