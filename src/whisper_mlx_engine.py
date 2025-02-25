""" Whisper MLX engine """

# Copyright © 2023-2025 Walter van Heuven

import os
import logging
import contextlib
from pathlib import Path
from dataclasses import asdict
import json
from pydantic import BaseModel, ConfigDict
from PyQt6.QtCore import QStandardPaths, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication
import numpy as np
import whisper as whisper_openai
import mlx_whisper
from mlx_whisper.whisper import Whisper
import convert
import utils as app_utils
from stream_emitter import StreamEmitter

class MLXWhisperData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_str: str
    model_dir_fpath: str
    task: str = "transcribe"
    model: Whisper | None = None
    filename: str = ""
    language: str | None = None
    result: str
    outputfilename: str
    output_folder: str
    err: bool = False
    error_msg:str = ""
    interrupted: bool = False

class Worker(QThread):
    finished = pyqtSignal(MLXWhisperData)
    output = pyqtSignal(str)
    output_progress = pyqtSignal(str)
    interrupt = pyqtSignal()

    def __init__(self, data:MLXWhisperData = None) -> None:
        super().__init__()
        self.setObjectName("WhisperEngine")
        self.stdout = StreamEmitter()
        self.stderr = StreamEmitter()
        self.stdout.message.connect(self.output.emit)
        self.stderr.message.connect(self.output_progress.emit)
        self.data = data
        self.interrupted = False
        self.interrupt.connect(self.handle_interrupt)

    def get_id(self) -> int:
        return int(QThread.currentThreadId())

    def run(self) -> None:
        err: bool = False
        already_existed: bool = False
        interrupted: bool = False

        err, already_existed = self.download_model()

        if not err:
            # convert weights to mlx format
            if not already_existed:
                err = self.convert_to_mlx()

            if not err:
                err, interrupted = self.execute_task_mlx()
            else:
                logging.error("Unable to convert weights to mlx")
        else:
            logging.error("Unable to download weights")

        # store err
        self.data.err = err
        self.data.interrupted = interrupted

        # emit results
        self.finished.emit(self.data)

    def model_str(self, model_str) -> str:
        if '/' in model_str:
            return model_str.replace('/', '_')
        else:
            return model_str

    def download_model(self) -> list[bool, bool]:
        err: bool = False
        already_exists: bool = False
        msg: str

        model_str = self.model_str(self.data.model_str)
        model_folder = os.path.join(self.data.model_dir_fpath, model_str)

        if not model_str.startswith("mlx-community"):
            # .pt is original whisper model file ending
            model_file_fpath = os.path.join(self.data.model_dir_fpath, f"{model_str}.pt")
        else:
            # mlx-community models
            model_file_fpath = os.path.join(self.data.model_dir_fpath, model_str)

        if not os.path.exists(model_folder):
            # mlx model folder e.g. 'tiny' indicates that model has already been downloaded and converted
            logging.info("Whisper model folder not found: %s", model_folder)
            model = None
            try:
                #size = round(app_utils.get_hf_file_size("openai/whisper-large-v2", model_file))
                msg = "Downloading model"
                self.output.emit(msg)
                logging.info(msg)

                with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr), \
                    app_utils.modified_environ(HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_ENABLE_HF_TRANSFER='True', HF_HUB_DISABLE_SYMLINKS_WARNING='1'):
                    model:Whisper = convert.load_torch_model(self.data.model_str, self.data.model_dir_fpath)

            except RuntimeError:
                logging.exception("RuntimeError")
                err = True
            except OSError:
                logging.exception("OSError")
                err = True
            finally:
                if not err:
                    self.output.emit("Download completed.")
                    self.data.model = model
        else:
            already_exists = True

        if not err and not already_exists:
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

        return err, already_exists

    # python convert.py --torch-name-or-path '/Users/lpzwjv/Library/Application Support/Speech2Text/Models_whisper/tiny.pt' --mlx-path '/Users/lpzwjv/Library/Application Support/Speech2Text/Models_whisper/tiny'
    def convert_to_mlx(self) -> bool:
        err: bool = False
        model_dir_fpath = self.data.model_dir_fpath
        model_str = self.model_str(self.data.model_str)

        if model_str.startswith("mlx-community"):
            # no need to convert
            return False

        # openai whisper model
        original_model_file = f"{model_str}.pt"
        file_with_original_weights = os.path.join(model_dir_fpath, original_model_file)

        # mlx model
        mlx_model_file = "weights.npz"
        file_with_mlx_weights = os.path.join(model_dir_fpath, model_str, mlx_model_file)

        if os.path.isfile(file_with_original_weights):
            if not os.path.isfile(file_with_mlx_weights):
                logging.info("Converting weights...")

                torch_name_or_path = file_with_original_weights
                mlx_path = os.path.join(model_dir_fpath, model_str)
                quantize = False
                args = ""

                self.output.emit("Loading weights...")
                #dtype = mx.float16
                #model = convert.torch_to_mlx(convert.load_torch_model(torch_name_or_path)) #, dtype)
                model:Whisper = convert.convert(torch_name_or_path) #, dtype)
                config = asdict(model.dims)
                weights = dict(convert.tree_flatten(model.parameters()))

                mlx_path = Path(mlx_path)
                mlx_path.mkdir(parents=True, exist_ok=True)

                # Save weights
                self.output.emit("Saving converted weights...")
                np.savez(str(mlx_path / "weights.npz"), **weights)

                # Save config.json with model_type
                with open(str(mlx_path / "config.json"), "w", encoding="utf-8") as f:
                    config["model_type"] = "whisper"
                    json.dump(config, f, indent=4)

                self.output.emit("Saved converted weights.\n")
                logging.info("Completed and saved converted weights")

            else:
                logging.info("Weights for MLX already exist: %s", file_with_mlx_weights)
        else:
            err = True

        if not err:
            if os.path.isfile(file_with_original_weights):
                os.remove(file_with_original_weights)
            else:
                logging.error("Model file does not exist")
                err = True

        return err

    def execute_task_mlx(self) -> list[bool, bool]:
        err: bool = False
        interrupted: bool = False

        logging.debug("model_dir_fpath: %s", self.data.model_dir_fpath)
        logging.debug("language: %s", self.data.language)
        logging.debug("Task: %s", self.data.task)

        self.output.emit(f"Starting {self.data.task}...\n")
        file_with_mlx_weights = os.path.join(self.data.model_dir_fpath, self.data.model_str)

        try:
            decode_options = dict(
                language = self.data.language,
                task=self.data.task.lower()
            )
            with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
                result = mlx_whisper.transcribe(
                            audio=self.data.filename,
                            path_or_hf_repo=file_with_mlx_weights,
                            verbose=True,
                            temperature = (0, 0.2, 0.4, 0.6, 0.8, 1),
                            compression_ratio_threshold = 2.4,
                            no_speech_threshold = 0.6,
                            condition_on_previous_text = True,
                            **decode_options
                            
                        )
        except RuntimeError as e:
            logging.exception("RuntimeError in Worker")
            self.data.err = True
            self.data.error_message = str(e)
        except KeyboardInterrupt:
            logging.exception("KeyboardInterrupt in Worker")
            interrupted = True
        else:
            self.data.result = result

        return err, interrupted

    def handle_interrupt(self):
        """ not working yet """
        self.interrupted = True
        self.data.interrupted = True

class WhisperMLXEngine():

    def __init__(self, mainWindow) -> None:
        self.mainWindow = mainWindow
        self.settings = self.mainWindow.settings
        self.worker = None
        self.cancelled: bool = False

    def run(self, output_folder: str, filename: str, outputfilename: str, name: str, output_file_extension: str, mime: str) -> None:
        """
        Using Apple's MLX implementation of Whisper
        """
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

        language:str = app_utils.lang_to_code(self.settings.value("Settings/Language"))
        if language.lower() == "auto":
            # auto detect language
            language = None

        model_dir_fpath = os.path.join(os.path.abspath(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)), "Models_MLX_Whisper")
        if not os.path.isdir(model_dir_fpath):
            os.mkdir(model_dir_fpath)

        model_str:str = self.settings.value("Settings/MLX_model")
        if model_str == "large":
            model_str = "large-v3"
        if model_str == "turbo":
            model_str = "large-v3-turbo"
        
        msg = f"Running Whisper MLX (model: '{model_str}')"
        self.feedback(msg)
        logging.info(msg)

        task:str = self.settings.value("Settings/Task")

        data = MLXWhisperData(
            model_str=model_str,
            model_dir_fpath=model_dir_fpath,
            task=task,
            model=None,
            filename=filename,
            language=language,
            result="",
            output_folder=output_folder,
            outputfilename=outputfilename
        )

        self.worker = Worker(data)
        self.worker.setObjectName("WhisperMLX")
        self.worker.setTerminationEnabled(True)

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

    def handle_finished(self, data:MLXWhisperData) -> None:
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
            txt_writer = whisper_openai.utils.get_writer(output_type, output_folder)

            if txt_writer is not None and result is not None:
                txt_writer(result, outputfilename)
                self.feedback(f"\nSaved output to file: '{outputfilename}'")
                logging.info("Output: '%s'", outputfilename)
            else:
                logging.error("Unable access txt_writer")

        self.worker_shutdown()
        QApplication.restoreOverrideCursor()

        self.mainWindow.finished_processing(filename, err, cancelled = data.interrupted)

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
        m = convert.available_models()
        # m.append("mlx-community/whisper-large-v3-turbo")
        return m

    def feedback(self, msg:str, add_newline:bool = True, check_progress_bar:bool = False) -> None:
        self.mainWindow.form_widget.feedback(msg, add_newline = add_newline, check_progress_bar=check_progress_bar)
