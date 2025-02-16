""" Module defines the ConvertWorker class """

# Copyright © 2023-2025 Walter van Heuven

import os
import subprocess
import logging
import time
import platform
import numpy as np
import soundfile
from PyQt6.QtCore import QThread, pyqtSignal
import utils as app_utils

class ConvertWorker(QThread):
    """ class to convert files to .wav """
    finished = pyqtSignal(dict)
    output = pyqtSignal(str)

    def __init__(self, data:dict=None) -> None:
        super().__init__()
        self.setObjectName("ConvertWorker")

        self.data = data
        self.original_file_fpath = data['original_file_fpath']
        self.ffmpeg_fpath = data['ffmpeg_fpath']
        self.target_format = data['target_format']
        self.final_file_fpath = data['final_file_fpath']
        self.is_wav = data['is_wav']
        self.exit_flag = False
        self.err = False

    def run(self) -> None:
        the_path:str
        the_file:str
        err: bool = False

        # original file
        the_path, the_file = app_utils.split_path_file(self.original_file_fpath)
        full_filename:str = the_file
        if "." in the_file:
            the_file = the_file.split(".")[0]

        out_filename:str = f"{the_file}.{self.target_format}"
        out_fpath:str = os.path.join(the_path, out_filename)

        # if file already exists add '_converted' to avoid overwriting original file
        if full_filename == out_filename:
            out_filename:str = f"{the_file}_converted.{self.target_format}"
            out_fpath:str = os.path.join(the_path, out_filename)

        msg = f"ConvertWorker - script path: {os.getcwd()}"
        logging.debug(msg)
        msg = f"ConvertWorker - fpath: {out_fpath}"
        logging.debug(msg)

        if self.ffmpeg_fpath != "":
            err = self.convert_using_ffmpeg(out_fpath)
        elif self.is_wav:
            err = self.convert_wav(out_fpath)

        self.data['err'] = err

        # emit results
        self.finished.emit(self.data)

    def convert_using_ffmpeg(self, out_fpath) -> bool:
        msg:str = ""
        err:bool = False
        p:subprocess.Popen = None
        try:
            tic:float = time.perf_counter()

            if self.target_format == 'mp3':

                logging.debug("Converting to mp3")
                # convert to 128kps and mono mp3 file
                cmd_list: list[str] = [
                    self.ffmpeg_fpath,
                    "-nostdin",
                    "-i", self.original_file_fpath,
                    "-f", "mp3",
                    "-ab", "128000", "-ac", "1", "-vn",
                    out_fpath
                ]

                if platform.system() == "Windows":
                    env, si = app_utils.get_windows_env()
                    p = subprocess.Popen(
                            cmd_list,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True, shell=False, startupinfo=si, env=env
                        )
                else:
                    p = subprocess.Popen(
                            cmd_list,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True, shell=False
                        )

                #out, err_out = p.communicate() # wait untill completed
                _, _ = p.communicate() # wait untill completed

            elif self.target_format == 'wav':

                logging.debug("Converting to wav")
                # for whisper and whisper.cpp
                # -ar 16000 -ac 1 -c:a pcm_s16le

                cmd_list: list[str] = [
                    self.ffmpeg_fpath,
                    "-nostdin", "-i",
                    self.original_file_fpath,
                    "-f", "wav", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                    out_fpath
                ]

                # stereo required for speaker detection
                #cmd_list: list[str] = [
                # self.ffmpeg_fpath, "-nostdin", "-i", 
                # self.original_file_fpath, "-f", "wav", "-ar", "16000", "-c:a", "pcm_s16le", 
                # out_fpath
                #]

                if platform.system() == "Windows":
                    env, si = app_utils.get_windows_env()
                    p = subprocess.Popen(
                            cmd_list,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True, shell=False, startupinfo=si, env=env
                        )
                else:
                    p = subprocess.Popen(
                            cmd_list,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True, shell=False
                        )

                #out, err_out = p.communicate() # wait untill completed
                _, _ = p.communicate() # wait untill completed

        except subprocess.CalledProcessError:
            logging.exception("ffmpeg CalledProcessError")
            err = True
        except subprocess.TimeoutExpired:
            logging.exception("ffmpeg TimeoutExpired")
            err = True
        except OSError:
            logging.exception("ffmpeg OSError")
            err = True
        finally:
            if p:
                for fd in [p.stdin, p.stdout, p.stderr]:
                    if fd:
                        fd.close()

            if self.exit_flag:
                pass
            else:
                if err:
                    self.output.emit("An error occurred!")
                else:
                    toc:float = time.perf_counter()
                    time_str:str = time.strftime('%Hh%Mm%Ss', time.gmtime(toc - tic))

                    self.data['final_file_fpath'] = out_fpath

                    logging.info("Completed conversion: new file: %s", out_fpath)
                    msg = f"Completed '{self.target_format}' conversion ({time_str})."
                    self.output.emit(msg)

        return err

    def convert_wav(self, out_fpath) -> bool:
        err:bool = False

        new_samplerate = 16000
        bits = 'PCM_16' # 16 bits
        try:
            data, samplerate = soundfile.read(self.original_file_fpath)
            
            if len(data.shape) > 1:
                # convert to mono
                mono_data = np.mean(data, axis=1)
            else:
                mono_data = data

            soundfile.write(out_fpath, mono_data, new_samplerate, subtype=bits)
        except soundfile.SoundFileError:
            err = True
            logging.error("SoundFileError")
            self.output.emit("SoundFileError")
        except OSError:
            err = True
            logging.error("OSError while converting wav")
            self.output.emit("OSError while converting wav")
        finally:
            self.data['final_file_fpath'] = out_fpath

            logging.info("Completed file conversion to: %s", out_fpath)
            msg = f"Completed '{self.target_format}' conversion."
            self.output.emit(msg)

        return err
