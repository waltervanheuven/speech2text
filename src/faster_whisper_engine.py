""" Faster whisper Engine """

# Copyright © 2023-2025 Walter van Heuven

import os
import logging
import contextlib
from pydantic import BaseModel, ConfigDict
import faster_whisper
import faster_whisper.transcribe
import pysubs2
from PyQt6.QtCore import Qt, QStandardPaths, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication
import utils as app_utils
from stream_emitter import StreamEmitter

class FasterWhisperData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_dir_fpath: str
    model_str: str
    model_dir: str
    device: str
    filename: str
    task: str
    language: str | None = None
    ssafile: pysubs2.SSAFile | None
    outputfilename: str
    result: list[str] | None
    transcription_info: faster_whisper.transcribe.TranscriptionInfo | None
    err: bool = False
    interrupted: bool = False

def format_timestamp(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

class Worker(QThread):
    finished = pyqtSignal(FasterWhisperData)
    output = pyqtSignal(str)
    output_progress = pyqtSignal(str)

    def __init__(self, data:FasterWhisperData=None) -> None:
        super().__init__()
        self.setObjectName("FasterWhisper")

        self.stdout = StreamEmitter()
        self.stderr = StreamEmitter()
        self.stdout.message.connect(self.output.emit)
        self.stderr.message.connect(self.output_progress.emit)

        self.data:FasterWhisperData = data

    def run(self) -> None:

        err:bool = self.download_model()

        if not err:
            # err, interrupted
            err, _ = self.transcribe()

        self.data.err = err

        # emit results
        self.finished.emit(self.data)

    def download_model(self) -> bool:
        err: bool = False
        model_dir: str = ""

        # download model if model file does not exist
        selected_model_path = os.path.join(self.data.model_dir_fpath, self.data.model_str)
        if not os.path.isdir(selected_model_path):
            self.output.emit(f"Downloading model: {self.data.model_str}...")

            try:
                with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr), \
                     app_utils.modified_environ(HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_ENABLE_HF_TRANSFER='True', HF_HUB_DISABLE_SYMLINKS_WARNING='1'):

                    model_dir = faster_whisper.download_model(
                                    self.data.model_str,
                                    output_dir=selected_model_path
                                )
            except ValueError as e:
                msg = f"ValueError: {e}"
                logging.exception(msg)
                err = True
            except Exception as e:
                msg = f"Exception: {e}"
                logging.exception(msg)
                err = True
            else:
                self.output.emit("Download completed.\n")
        else:
            model_dir = selected_model_path

        if not err:
            self.data.model_dir = model_dir

        return err

    def transcribe(self) -> tuple[bool, bool]:
        err: bool = False
        interrupted: bool = False
        try:
            self.output.emit("Starting transcription... (press Esc to stop)\n")

            compute_type = "float16"
            if self.data.device == "cuda":
                compute_type = "float16"
            elif self.data.device == "cpu":
                compute_type = "int8"

            subs = pysubs2.SSAFile()

            with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr), \
                 app_utils.modified_environ(HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_ENABLE_HF_TRANSFER='True', HF_HUB_DISABLE_SYMLINKS_WARNING='1'):
                model = faster_whisper.WhisperModel(
                        model_size_or_path=self.data.model_dir,
                        device=self.data.device,
                        compute_type=compute_type,
                        cpu_threads=4,
                        num_workers=1
                    )
                # generator, so iterate
                segments, transcription_info = model.transcribe(
                                self.data.filename,
                                task=self.data.task.lower(),
                                beam_size=5, # default
                                best_of=5,   # default
                                patience=1,  # default
                                language=self.data.language,
                                condition_on_previous_text=True # default
                            )
            result = []
            for segment in segments:
                if self.isInterruptionRequested():
                    interrupted = True
                    break

                out_str = f"[{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}] {segment.text}"
                event = pysubs2.SSAEvent(
                    start=segment.start * 1000.0,
                    end=segment.end * 1000.0,
                    text=segment.text
                )
                subs.append(event)

                self.output.emit(out_str)
                result.append(out_str)

            self.data.interrupted = interrupted
            if not interrupted:
                self.data.ssafile = subs
                self.data.result = result
                self.data.transcription_info = transcription_info

        except OSError:
            logging.exception("OSError during transcription")
            self.output.emit("An error occurred!")
            err = True
        except Exception as e:
            msg = f"Exception: {e}"
            logging.exception(msg)
            err = True
        finally:
            self.output.emit("")

        return err, interrupted

class FasterWhisperEngine():

    def __init__(self, mainWindow):
        self.mainWindow = mainWindow
        self.settings = self.mainWindow.settings
        self.worker = None

    def run(self, filename: str, outputfilename: str) -> None:
        """
        Use faster-whisper
        https://github.com/guillaumekln/faster-whisper
        """
        # language
        language = app_utils.lang_to_code(self.settings.value("Settings/Language"))
        if language == "auto":
            language = None

        # output
        output_type = self.settings.value("Settings/Output")
        output_type = output_type.lower()
        if output_type == "text":
            output_type = "txt"

        # task
        # note that faster-whisper can only transcribe audio
        task = self.settings.value("Settings/Task")

        # model_str
        model_str = self.settings.value("Settings/FW_model")

        # model_dir_fpath
        model_dir_fpath = os.path.join(
            os.path.abspath(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)),
            "Models_faster-whisper"
        )
        if not os.path.isdir(model_dir_fpath):
            os.mkdir(model_dir_fpath)

        if app_utils.cuda_available():
            device = "cuda"
            msg = f"Running faster-whisper ({device.upper()} float16, model: '{model_str}')"
        else:
            device = "cpu"
            msg = f"Running faster-whisper ({device.upper()} int8, model: '{model_str}')"

        self.feedback(msg)
        logging.info(msg)

        logging.debug("Running faster-whisper task: %s", task)
        logging.debug("filename %s", filename)
        logging.debug("outputfilename %s", outputfilename)

        data = FasterWhisperData(
            model_dir_fpath=model_dir_fpath,
            device=device,
            model_str=model_str,
            model_dir="",
            filename=filename,
            task=task,
            language=language,
            ssafile=None,
            outputfilename=outputfilename,
            result=None,
            transcription_info=None
        )

        self.worker = Worker(data)
        self.worker.setObjectName("FasterWhisper")

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

    def handle_finished(self, data) -> None:
        err:bool = data.err
        filename:str = data.filename
        cancelled:bool = data.interrupted

        if not cancelled:
            outputfilename = data.outputfilename

            output_type = self.settings.value("Settings/Output")
            output_type = output_type.lower()
            if output_type == "text":
                output_type = "txt"

            if not err:
                subs = data.ssafile
                if output_type != "txt":
                    subs.save(outputfilename)
                else:
                    result = data.result
                    with open(outputfilename, 'w', encoding='utf-8') as f:
                        for line in result:
                            f.write(f"{line}\n")

                self.feedback(f"\nSaved output to file: '{outputfilename}'")

        self.worker_shutdown()
        QApplication.restoreOverrideCursor()

        self.mainWindow.finished_processing(filename, err, cancelled=cancelled)

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

    def faster_whisper_models(self) -> list[str]:
        # not function lib to return list of models
        return [
            "tiny.en", "tiny",
            "base.en", "base",
            "small.en", "small",
            "medium.en", "medium",
            "large-v1", "large-v2", "large-v3",
            "large",
            "distil-small.en", 
            "distil-medium.en", 
            "distil-large-v2", "distil-large-v3",
            #"Systran/faster-whisper-large-v3"
        ]

    def feedback(self, msg, add_newline = True, check_progress_bar = False) -> None:
        self.mainWindow.form_widget.feedback(msg, add_newline = add_newline, check_progress_bar=check_progress_bar)

