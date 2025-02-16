""" Module provides utility functions """

# Copyright © 2023-2025 Walter van Heuven

import contextlib
import os
import subprocess
import time
import shutil
import platform
import hashlib
import logging
import wave
import psutil
import puremagic
from iso639 import Lang
from torch import cuda
from PyQt6.QtGui import QDesktopServices, QFont, QFontMetrics
from PyQt6.QtCore import QUrl
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from mainwindow import MainWindow

def cuda_available() -> bool:
    return cuda.is_available()

# https://github.com/laurent-laporte-pro/stackoverflow-q2059482
@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        _ = [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        _ = [env.pop(k) for k in remove_after]

# based on https://www.debugpointer.com/python/create-sha256-hash-of-a-file-in-python
def compute_sha(file_name: str) -> None:
    """ compute sha256 hash of file """
    hash_sha = hashlib.sha1()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha.update(chunk)
    return hash_sha.hexdigest()

def split_path_file(complete_file_path: str) -> tuple[str, str]:
    return os.path.dirname(complete_file_path), os.path.basename(complete_file_path)

def is_program_installed(program_name: str) -> bool:
    return shutil.which(program_name)

def bool_value(s: str) -> bool:
    if isinstance(s, str):
        return s.lower() == "true"
    else:
        return False

def bool_to_string(b: bool) -> str:
    return "True" if b else "False"

def check_acceptable_file(
        main_window: MainWindow,
        fpath: str,
        acceptable_extensions: list[str]
    ) -> bool:
    """ check whether file is acceptable """
    ok:bool = False
    msg: str
    if os.path.isfile(fpath):
        try:
            extension = puremagic.from_file(fpath)
        except puremagic.main.PureError:
            msg = "Puremagic unable to identify file."
            logging.exception(msg)
        except ValueError:
            msg = "ValueError puremagic"
            logging.exception(msg)
            main_window.form_widget.feedback(msg)
        except OSError:
            msg = "OSError puremagic"
            logging.exception(msg)
            main_window.form_widget.feedback(msg)
        else:
            if extension in acceptable_extensions:
                ok = True
    return ok

def correct_wav_file(fpath: str) -> tuple[bool, bool]:
    """ check whether audio file is 16kHz """
    err:bool = False
    ok_format:bool = False

    sample_rate:int = 0
    sample_width:int = 0

    # if wav file check appropriate format
    logging.debug("Check sample rate of: %s", fpath)

    try:
        with wave.open(fpath, 'rb') as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
    except wave.Error:
        logging.exception("Wave error")
        err = True
    else:
        logging.debug("wav file, sample_rate: %d Hz, sample_width: %d", sample_rate, sample_width)

    if sample_rate == 16000 and sample_width == 2:
        ok_format = True
        logging.debug("Format wav ok")

    return ok_format, err

def get_audio_duration(file_path: str) -> float:
    extension: str = ""
    if os.path.isfile(file_path):
        try:
            extension = puremagic.from_file(file_path)
        except puremagic.main.PureError:
            msg = "Puremagic unable to identify file."
            logging.exception(msg)
        except ValueError:
            msg = "ValueError puremagic"
            logging.exception(msg)
        except OSError:
            msg = "OSError puremagic"
            logging.exception(msg)

    try:
        if extension in ['.wav']:
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
            return duration
        else:
            return 0
    except Exception as e:
        msg = f"get_wav_duration exception: {e}"
        logging.exception(msg)
        return 0

def check_speech_detected(outputfilename: str) -> bool:
    try:
        with open(outputfilename, 'r', encoding="utf-8") as file:
            if file.readline().strip() == "":
                return False
            else:
                return True
    except Exception as e:
        msg = f"check_speech_detected exception: {e}"
        logging.exception(msg)
        return False

def duration_str(duration: float) -> str:
    result = time.gmtime(duration)
    time_str = ""
    if result.tm_sec < 60 and result.tm_min == 0 and result.tm_hour == 0:
        if result.tm_sec == 1:
            time_str = "a second"
        else:
            time_str = f"{result.tm_sec} seconds"
    elif result.tm_min < 60 and result.tm_hour == 0:
        if result.tm_min == 1 and result.tm_sec == 0:
            time_str = "a minute"
        elif result.tm_min == 1 and result.tm_sec > 0:
            time_str = f"one minute and {result.tm_sec} seconds"
        else:
            time_str = f"{result.tm_min} minutes and {result.tm_sec} seconds"
    elif result.tm_hour < 24:
        if result.tm_hour == 1 and result.tm_min == 0:
            time_str = "an hour"
        else:
            m = result.tm_min
            if result.tm_sec < 30:
                if result.tm_hour == 1:
                    time_str = f"One hour and {m} minutes"
                else:
                    time_str = f"{result.tm_hour} hours and {m} minutes"
            else:
                m = m + 1
                if result.tm_hour == 1:
                    time_str = f"One hour and {m} minutes"
                else:
                    time_str = f"{result.tm_hour} hours and {m} minutes"

    return time_str

def lang_to_code(lang_str: str) -> str:
    """ convert to iso639 """
    if lang_str == "Auto Detect":
        return "auto"
    else:
        return Lang(lang_str).pt1

def str_width(widget, s: str) -> int:
    the_font: QFont = widget.font()
    fm: QFontMetrics = QFontMetrics(the_font)
    return fm.horizontalAdvance(s)

def str_height(widget, s: str) -> int:
    the_font: QFont = widget.font()
    fm: QFontMetrics = QFontMetrics(the_font)
    return fm.xHeight

def max_str_in_list(the_list: list) -> str:
    return max(the_list, key=len)

def handle_links(url: str) -> None:
    """ open link in default web browser """    
    if not url.scheme() or url.toString().startswith('file:'):
        url = QUrl.fromLocalFile(url.toString())
    QDesktopServices.openUrl(url)

def desktop_notification(title: str, msg: str) -> None:
    """ send notification that transcription has ended """
    if platform.system() == "Darwin":
        try:
            # unfortunately it is not possible to use a custom icon
            cmd = ["sh", "-c", "osascript -e 'display notification \"%s\" with title \"%s\"'" % (msg, title)]
            subprocess.run(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True, shell=False, text=True
            )
        except subprocess.CalledProcessError:
            logging.exception("CalledProcessError during Desktop Notification")
    else:
        try:
            import plyer
            plyer.notification.notify(
                title=title,
                message=msg,
                timeout=2
            )
        except ImportError:
            logging.exception("Plyer ImportError")
        except NotImplementedError:
            logging.exception("Plyer NotImplementedError")

def app_already_running(app_name: str) -> bool:
    processes: list[str] = []
    for process in psutil.process_iter(['pid', 'name']):
        processes.append(process.info['name'])

    return processes.count(app_name) > 1

def kill(proc_pid: int):
    """ kill process """
    try:
        if psutil.pid_exists(proc_pid):
            process = psutil.Process(proc_pid)
            if process is not None:
                for p in process.children(recursive=True):
                    p.kill()
                process.kill()
    except ProcessLookupError:
        logging.exception("ProcessLookupError")
    except psutil.NoSuchProcess:
        logging.exception("NoSuchProcess")
    except OSError:
        logging.exception("OSError")

def get_windows_env():
    """ get windows ENV and disable how terminal window 
    
    returns os._Environ, subprocess.STARTUPINFO
    """
    # https://github.com/pyinstaller/pyinstaller/wiki/Recipe-subprocess
    if hasattr(subprocess, 'STARTUPINFO'):
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        env = os.environ
    else:
        si = None
        env = None

    return env, si

def file_size_in_mb(size: int) -> int:
    return int(size / (1024 * 1000))

def get_hf_file_size(repo_id: str, filename:str) -> int:
    """ return model file in megabytes """
    size:int = 0
    try:
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        meta = get_hf_file_metadata(url)
    except EnvironmentError:
        logging.exception("EnvironmentError")
    else:
        size = int(meta.size / (1024 * 1000))

    return size
