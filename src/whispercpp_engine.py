""" whisper.cpp Engine """

# Copyright © 2023-2025 Walter van Heuven

import os
import sys
import logging
import subprocess
import platform
import contextlib
from pathlib import Path
import shutil
import zipfile
from pydantic import BaseModel, PositiveInt
from PyQt6.QtCore import Qt, QStandardPaths, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication
from huggingface_hub import hf_hub_download
import utils as app_utils
from stream_emitter import StreamEmitter

# CoreML and Metal options:
# https://github.com/ggerganov/whisper.cpp/discussions/1722

class WhisperCPPData(BaseModel):
    model_str: str
    model_file: str
    use_coreml: bool
    use_metal: bool = True
    use_cuda: bool
    model_dir_fpath: str
    model_file_fpath: str
    outputfilename: str
    output_file_extension: str
    task_str: str = "transcribe"
    audiofile: str
    file_folder: str
    threads: PositiveInt = 4
    text_output_format: str
    language: str = "auto"
    options_str: str
    err: bool = False
    error_message: str = ""
    interrupted: bool = False

WHISPER_CPP = 'whisper-cli'
WHISPER_CPP_WIN = 'whisper-cli.exe'

class Worker(QThread):
    finished = pyqtSignal(WhisperCPPData)
    output = pyqtSignal(str)
    output_progress = pyqtSignal(str)
    output_nn = pyqtSignal(str, bool)
    process = None

    def __init__(self, data:WhisperCPPData=None) -> None:
        super().__init__()

        self.stdout = StreamEmitter()
        self.stderr = StreamEmitter()
        self.stdout.message.connect(self.output.emit)
        self.stderr.message.connect(self.output_progress.emit)

        self.data:WhisperCPPData = data
        self.setObjectName("WhisperCPP")

    def run(self) -> None:
        interrupted: bool = False

        try:
            err:bool = self.download_model()

            if self.isInterruptionRequested():
                interrupted = True
                self.data.interrupted = interrupted

            if not err and not interrupted:
                err, interrupted = self.start_asr()

            self.data.err = err
        except RuntimeError as e:
            logging.exception("RuntimeError in Worker")
            self.data.err = True
            self.data.error_message = str(e)
        finally:
            self.finished.emit(self.data)

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

        self.requestInterruption()

    def download_model(self) -> bool:
        err: bool = False

        # if model file does not exist download it
        if not os.path.exists(self.data.model_file_fpath):
            err = self.download_whisper_cpp_model()

        if not err: # and self.data.use_coreml:
            # For macOS and CoreML
            model_str = self.data.model_str
            if model_str.endswith('-q5_0'):
                model_str = model_str.replace('-q5_0', '')
            if model_str.endswith('-q8_0'):
                model_str = model_str.replace('-q8_0', '')

            des_name = f"ggml-{model_str}-encoder.mlmodelc"
            des_name_path = os.path.join(self.data.model_dir_fpath, des_name)

            if not os.path.exists(des_name_path):
                err = self.download_mlmodelc(des_name)

        return err

    def download_whisper_cpp_model(self) -> bool:
        err: bool = False
        model_file_fpath: str = ""
        logging.debug(self.data.model_file_fpath)

        try:
            msg:str = "Downloading model..."
            logging.info(msg)
            self.output.emit(msg)

            with contextlib.redirect_stdout(self.stdout), \
                 contextlib.redirect_stderr(self.stderr), \
                 app_utils.modified_environ(HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_ENABLE_HF_TRANSFER='True'):
                model_file_fpath = hf_hub_download(
                                    repo_id="ggerganov/whisper.cpp",
                                    filename=self.data.model_file,
                                    local_dir=self.data.model_dir_fpath
                                )
            logging.info("Download model FINISHED")
        except ConnectionError:
            logging.exception("ConnectionError")
            err = True
        except ValueError:
            logging.exception("ValueError")
            err = True
        except OSError:
            logging.exception("OSError")
            err = True
        else:
            self.output.emit("Download completed.")
            self.data.model_file_fpath = model_file_fpath

        if err:
            self.output.emit("An error occurred!")

        return err

    def download_mlmodelc(self, des_name: str) -> tuple[bool, str]:
        err: bool = False

        #https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base-encoder.mlmodelc.zip
        file: str = f"{des_name}.zip"
        model_file_fpath:str = os.path.join(self.data.model_dir_fpath, file)
        name = os.path.join(self.data.model_dir_fpath, Path(model_file_fpath).stem)

        logging.info("model: '%s' not found!", des_name)

        try:
            msg = "Downloading mlmodelc..."
            logging.info(msg)
            self.output.emit(msg)

            with contextlib.redirect_stdout(self.stdout), \
                 contextlib.redirect_stderr(self.stderr), \
                 app_utils.modified_environ(HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_ENABLE_HF_TRANSFER='True'):
                zipped_model_file_fpath = hf_hub_download(
                                    repo_id="ggerganov/whisper.cpp",
                                    filename=file,
                                    local_dir=self.data.model_dir_fpath
                                )
            logging.info("Download mlmodelc FINISHED")
        except ConnectionError:
            logging.exception("ConnectionError")
            err = True
        except ValueError:
            logging.exception("ValueError")
            err = True
        except OSError:
            logging.exception("OSError")
            err = True
        else:
            self.output.emit("Download completed.")

        if err:
            msg = f"mlmodelc download failed: {file}"
            logging.error(msg)
            self.output.emit("An error occurred!")

        if not err and not os.path.isdir(name):
            if "-v2" in self.data.model_str:
                # issue large-v2 and mlmodelc
                # file in zipped mlmodelc 
                temp_dir = os.path.join(self.data.model_dir_fpath, "tmp")

                try:
                    os.makedirs(temp_dir, exist_ok = True)

                    with zipfile.ZipFile(zipped_model_file_fpath, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                        extracted_files = zip_ref.namelist()

                    if len(extracted_files) > 0:
                        extracted_file = extracted_files[0]

                        shutil.move(
                            os.path.join(temp_dir, extracted_file),
                            os.path.join(self.data.model_dir_fpath, des_name)
                        )
                        shutil.rmtree(temp_dir)

                        if os.path.isfile(zipped_model_file_fpath):
                            os.remove(zipped_model_file_fpath)
                    else:
                        err = True
                        logging.error("Unexpected number of files in zip file.")

                except OSError:
                    err = True
                    logging.exception("shutil error (-)")
            else:
                try:
                    self.output.emit(f"Extracting file: '{Path(model_file_fpath).stem}'...")

                    shutil.unpack_archive(
                        model_file_fpath,
                        extract_dir=self.data.model_dir_fpath,
                        format="zip"
                    )

                    if not os.path.isdir(name):
                        logging.error("Error extracting file: %s", model_file_fpath)
                        err = True

                    if os.path.isfile(model_file_fpath):
                        os.remove(model_file_fpath)

                except OSError as e:
                    err = True
                    msg = f"shutil OSError: {e}"
                    logging.exception(msg)
                    self.output.emit("An error occurred while extracting zip file!")
                else:
                    self.output.emit("Extraction completed.\n")

        return err

    def start_asr(self) -> tuple[bool, bool]:
        err: bool = False
        msg: str = ""
        app_str: str = ""

        self.output.emit(f"Starting {self.data.task_str}... (press Esc to stop)\n")
        msg = f"Starting {self.data.task_str}..."
        logging.info(msg)

        if platform.system() == "Darwin":
            if self.data.use_coreml:
                app_str = os.path.join("coreml", WHISPER_CPP)
            else:
                app_str = os.path.join("metal", WHISPER_CPP)

        elif platform.system() == "Windows":
            if not self.data.use_cuda:
                app_str = WHISPER_CPP_WIN
            elif self.data.use_cuda:
                app_str = os.path.join("cuda", WHISPER_CPP_WIN)
        else:
            logging.error("No binary available for this system")
            err = True

        # find location of whisper.cpp executable
        if getattr(sys, 'frozen', False):

            main_dir = getattr(sys, '_MEIPASS', os.getcwd())
            if platform.system() == "Windows":
                msg = f"Windows main_dir (in app): {main_dir}"
            else:
                msg = f"main_dir (in app): {main_dir}"
            logging.debug(msg)
        else:
            main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            logging.debug("main_dir (in script): %s", main_dir)

        # app location within bin folder
        app_file_path = os.path.join(main_dir, "bin", app_str)

        # main command string
        cmd_list:list[str] = [
            f"{app_file_path}",
            "-t", str(self.data.threads),
            "-of", self.data.outputfilename, self.data.text_output_format,
            "-l", self.data.language,
            "-m", self.data.model_file_fpath
        ]

        if self.data.task_str == 'translate':
            cmd_list.append('--translate')

        logging.debug("file in cmd: %s", self.data.audiofile)
        cmd_list = cmd_list + ["-f", self.data.audiofile]

        if len(self.data.options_str) > 0:
            cmd_list = cmd_list + self.data.options_str.split()

        logging.debug("Subprocess: %s", cmd_list)

        interrupted:bool = False
        try:
            if platform.system() == "Darwin":
                self.process = subprocess.Popen(
                        cmd_list,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        shell=False
                    )
            elif platform.system() == "Windows":
                env, si = app_utils.get_windows_env()
                self.process = subprocess.Popen(
                        cmd_list,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        startupinfo=si,
                        env=env
                    )

            if self.process:
                for line in iter(self.process.stdout.readline, b""):
                    if self.process.poll() is not None:
                        break
                    elif self.isInterruptionRequested():
                        interrupted = True
                        break
                    else:
                        line = line.rstrip()
                        if len(line) > 0:
                            self.output.emit(line)

            if self.process and not interrupted:
                out, err_out = self.process.communicate() # wait untill completed

                if err_out and ("error:" in err_out or "failed to load" in err_out):
                    logging.error("Error in whisper.cpp out:\n%s", err_out)
                    err = True

            if not err and not interrupted:
                self.output.emit(f"\nSaved output to file: '{self.data.outputfilename}{self.data.output_file_extension}'")
                msg = f"Output: '{self.data.outputfilename}{self.data.output_file_extension}'"
                logging.info(msg)

        except subprocess.CalledProcessError as e:
            msg = f"CalledProcessError: {e}"
            logging.exception(msg)
            err = True
        except OSError as e:
            msg = f"OSError: {e}"
            logging.exception(msg)
            err = True
        finally:
            if self.process:
                for fd in [self.process.stdin, self.process.stdout, self.process.stderr]:
                    if fd:
                        fd.close()
                self.stop()
            if not err:
                self.data.interrupted = interrupted

        return err, interrupted

class WhisperCPPEngine():

    def __init__(self, mainWindow):
        self.mainWindow = mainWindow
        self.settings = self.mainWindow.settings
        self.worker: Worker = None

    def run(self, filename: str, output_file_extension: str) -> None:
        """
        Using whisper.cpp
        https://github.com/ggerganov/whisper.cpp

        Function called from WhisperWorker
        """

        task_str = str(self.mainWindow.form_widget.comboTask.currentText()).lower()

        # convert file if needed
        err, already_exist, converted, fpath = self.mainWindow.convert_input_file_if_needed(filename)

        if not converted and not already_exist and not err:
            # remove extension for outputfilename
            the_folder, the_basename, _ = app_utils.split_path_file(filename)
            new_outputfilename = os.path.join(the_folder, the_basename)

            # continue with converting
            self.continue_processing(filename, new_outputfilename, task_str, output_file_extension)

        elif already_exist and not err:
            logging.debug("File %s already exists and already added to queue", fpath)
            self.mainWindow.finished_processing(filename, False)
        elif err:
            logging.error("An error occurred while trying to convert audio file")
            self.mainWindow.finished_processing(filename, err)

    def continue_processing(self, filename: str, outputfilename: str, task_str: str, output_file_extension: str) -> None:
        err: bool = False
        msg: str
        text_output_format: str

        file_folder, the_filename, _ = app_utils.split_path_file(filename)

        threads:int
        try:
            threads = int(self.settings.value("Settings/CPP_threads"))
        except ValueError:
            msg = f"Unable to convert to int: {self.settings.value("Settings/CPP_threads")}"
            logging.error(msg)
            threads = 4

        use_metal:bool = app_utils.bool_value(self.settings.value("Settings/CPP_Metal"))
        use_coreml:bool = app_utils.bool_value(self.settings.value("Settings/CPP_CoreML"))
        use_cuda:bool = app_utils.bool_value(self.settings.value("Settings/CPP_CUDA"))

        # set device
        if use_metal and not use_coreml:
            device = "Metal"
        elif use_coreml and not use_metal:
            device = "CoreML"
        elif use_coreml and use_metal:
            device = "Metal_CoreML"
        elif use_cuda:
            device = "CUDA"
        else:
            device = "CPU"

        # whisper options
        options_str:str = self.settings.value("Settings/CPP_options")

        model_str:str = self.settings.value("Settings/CPP_model")
        # select v3 is model_str == large
        if model_str == "large":
            model_str = "large-v3"

        # check if model file already downloaded, if not download it
        model_file = f"ggml-{model_str}.bin"
        if "-tdrz" in options_str or "--tinydiarize" in options_str:
            model_file = f"ggml-{model_str}-tdrz.bin"

        model_folder_fpath = os.path.join(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation),
            "Models_whisper_cpp"
        )
        if not os.path.isdir(model_folder_fpath):
            os.mkdir(model_folder_fpath)

        model_file_fpath = os.path.join(model_folder_fpath, model_file)

        # check hash256 of model file in case download failed previously
        if os.path.exists(model_file_fpath):
            hash_err = False
            hash_from_file = app_utils.compute_sha(model_file_fpath)
            if model_str == "tiny" and hash_from_file != 'bd577a113a864445d4c299885e0cb97d4ba92b5f':
                hash_err = True
            elif model_str == "tiny.en" and hash_from_file != 'c78c86eb1a8faa21b369bcd33207cc90d64ae9df':
                hash_err = True
            elif model_str == "base" and hash_from_file != '465707469ff3a37a2b9b8d8f89f2f99de7299dac':
                hash_err = True
            elif model_str == "base.en" and hash_from_file != '137c40403d78fd54d454da0f9bd998f78703390c':
                hash_err = True
            elif model_str == "small" and hash_from_file != '55356645c2b361a969dfd0ef2c5a50d530afd8d5':
                hash_err = True
            elif model_str == "small.en" and hash_from_file != 'db8a495a91d927739e50b3fc1cc4c6b8f6c2d022':
                hash_err = True
            elif model_str == "medium" and hash_from_file != 'fd9727b6e1217c2f614f9b698455c4ffd82463b4':
                hash_err = True
            elif model_str == "medium.en" and hash_from_file != '8c30f0e44ce9560643ebd10bbe50cd20eafd3723':
                hash_err = True
            elif model_str == "large-v1" and hash_from_file != 'b1caaf735c4cc1429223d5a74f0f4d0b9b59a299':
                hash_err = True
            elif model_str == "large-v2" and hash_from_file != '0f4c8e34f21cf1a914c59d8b3ce882345ad349d6':
                hash_err = True
            elif model_str == "large-v2-q5_0" and hash_from_file != '00e39f2196344e901b3a2bd5814807a769bd1630':
                hash_err = True
            elif model_str == "large-v3" and hash_from_file != 'ad82bf6a9043ceed055076d0fd39f5f186ff8062':
                hash_err = True
            elif model_str == "large-v3-q5_0" and hash_from_file != 'e6e2ed78495d403bef4b7cff42ef4aaadcfea8de':
                hash_err = True
            elif model_str == "large-v3-turbo" and hash_from_file != '4af2b29d7ec73d781377bfd1758ca957a807e941':
                hash_err = True
            elif model_str == "large-v3-turbo-q5_0" and hash_from_file != 'e050f7970618a659205450ad97eb95a18d69c9ee':
                hash_err = True

            if hash_err:
                msg = f"Model file '{model_file}' (model_str: {model_str}) has incorrect hash: {hash_from_file}.\nDeleting file."
                logging.error(msg)
                self.feedback(msg)

                # delete file
                os.remove(model_file_fpath)

                return True
        
        # continue if no error has occurred
        if options_str == "":
            msg = f"Running whisper.cpp ({device}, model: '{model_str}')"
        else:
            msg = f"Running whisper.cpp ({device} {options_str}, model: '{model_str}')"

        self.feedback(msg)
        logging.info(msg)

        settings_output_format:str = self.settings.value("Settings/Output")

        # default output as text
        text_output_format = '-otxt'
        if settings_output_format == "TSV":
            text_output_format = '-ocsv'
        elif settings_output_format == "VTT":
            text_output_format = '-ovtt'
        elif settings_output_format == "SRT":
            text_output_format = '-osrt'
        elif settings_output_format == "LRC":
            text_output_format = '-olrc'
        elif settings_output_format == "JSON":
            text_output_format = '-oj'

        language:str = app_utils.lang_to_code(self.settings.value("Settings/Language"))

        logging.debug("FILENAME AUDIOFILE: %s", the_filename)

        data = WhisperCPPData(
            model_str=model_str,
            model_file=model_file,
            use_coreml=use_coreml,
            use_metal=use_metal,
            use_cuda=use_cuda,
            model_dir_fpath=model_folder_fpath,
            model_file_fpath=model_file_fpath,
            outputfilename=outputfilename,
            output_file_extension=output_file_extension,
            task_str=task_str,
            audiofile=filename,
            file_folder=file_folder,
            threads=threads,
            text_output_format=text_output_format,
            language=language,
            options_str=options_str
        )

        self.worker = Worker(data)
        self.worker.setObjectName("WhisperCPP")
        self.worker.setTerminationEnabled(True)

        # function to run when finished
        self.worker.finished.connect(self.handle_finished)
        #self.worker.finished.connect(self.worker.deleteLater)

        # function to print string
        self.worker.output.connect(lambda x: self.feedback(x, True, False))
        self.worker.output_progress.connect(lambda x: self.feedback(x, False, True))
        self.worker.output_nn.connect(self.feedback)

        # show processing bar
        self.mainWindow.form_widget.set_show_progress_element(True)

        # show busycursor
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)

        # start thread
        logging.debug("Starting WhisperCPP thread")
        self.worker.start()

    def handle_finished(self, data:WhisperCPPData) -> None:
        err = data.err
        audiofile = data.audiofile

        logging.debug("Handle_finished whisper.cpp")

        self.worker_shutdown()
        QApplication.restoreOverrideCursor()

        if not err:
            self.mainWindow.finished_processing(audiofile, err, cancelled=data.interrupted)
        else:
            self.mainWindow.finished_processing(audiofile, err, error_msg=data.error_message)

    def stop_processing(self):
        if self.worker is not None:
            if self.worker.isRunning():
                self.worker.stop()
                self.worker_shutdown()

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

    def whisper_cpp_models(self) -> list[str]:
        return [
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large",
            "large-v1",
            "large-v2",
            #"large-v2-q5_0",
            "large-v3",
            #"large-v3-q5_0",
            "large-v3-turbo",
            #"large-v3-turbo-q5_0"
        ]

    def feedback(self, msg:str , add_newline: bool = True, check_progress_bar: bool = False) -> None:
        self.mainWindow.form_widget.feedback(msg, add_newline = add_newline, check_progress_bar=check_progress_bar)
