""" Module defines the ConvertWorker class """

# Copyright © 2023-2025 Walter van Heuven

import os
import subprocess
import logging
import time
import platform
import av
from PyQt6.QtCore import QThread, pyqtSignal
import utils as app_utils

class ConvertWorker(QThread):
    """ class to convert files to .wav """
    finished = pyqtSignal(dict)
    output = pyqtSignal(str)

    def __init__(self, data:dict=None) -> None:
        super().__init__()
        self.setObjectName("ConvertWorker")
        self.data:dict = data
        self.exit_flag:bool = False
        self.err:bool = False

    def run(self) -> None:
        err: bool = False

        # original file
        the_path, the_file_name, _ = app_utils.split_path_file(self.data['original_file_fpath'])

        out_filename:str = f"{the_file_name}{self.data['target_format']}"
        out_fpath:str = os.path.join(the_path, out_filename)

        # if file already exists add '_converted' to avoid overwriting original file
        if the_file_name.lower() == out_filename.lower():
            out_filename:str = f"{the_file_name}_converted{self.data['target_format']}"
            out_fpath:str = os.path.join(the_path, out_filename)

        msg = f"ConvertWorker - script path: {os.getcwd()}"
        logging.debug(msg)
        msg = f"ConvertWorker - fpath: {out_fpath}"
        logging.debug(msg)

        #if self.data['ffmpeg_fpath'] != "":
        #    self.data['err'] = self.convert_using_ffmpeg(out_fpath)
        #else:
        self.data['err'] = self.convert_to_wav(out_fpath)

        # emit results
        self.finished.emit(self.data)

    def convert_using_ffmpeg(self, out_fpath) -> bool:
        """ Use FFmpeg to convert file to wav """
        msg:str = ""
        err:bool = False
        p:subprocess.Popen = None
        try:
            tic:float = time.perf_counter()

            if self.data['target_format'].lower() == '.mp3':

                logging.debug("Converting to mp3")
                # convert to 128kps and mono mp3 file
                cmd_list: list[str] = [
                    self.data['ffmpeg_fpath'],
                    "-nostdin",
                    "-i", self.data['original_file_fpath'],
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

            elif self.data['target_format'].lower() == '.wav':

                logging.debug("Converting to wav")
                # for whisper and whisper.cpp
                # -ar 16000 -ac 1 -c:a pcm_s16le

                cmd_list: list[str] = [
                    self.data['ffmpeg_fpath'],
                    "-nostdin", "-i",
                    self.data['original_file_fpath'],
                    "-f", "wav", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
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
                    msg = f"Completed '{self.data['target_format']}' conversion ({time_str})."
                    self.output.emit(msg)

        return err

    def convert_to_wav(self, out_fpath) -> bool:
        """ Use av to convert audio/video file to wav format required for whisper. """
        err: bool = False
        new_samplerate = 16000
        target_format = 's16'
        target_layout = 'mono'

        try:
            input_container = av.open(self.data['original_file_fpath'])
            input_audio_stream = next((s for s in input_container.streams if s.type == 'audio'), None)
            if input_audio_stream is None:
                raise ValueError("No audio stream found in file")

            output_container = av.open(out_fpath, mode='w')
            output_stream = output_container.add_stream('pcm_s16le', rate=new_samplerate, layout=target_layout)

            resampler = av.audio.resampler.AudioResampler(
                format=target_format,
                layout=target_layout,
                rate=new_samplerate
            )

            for packet in input_container.demux(input_audio_stream):
                for frame in packet.decode():
                    resampled_frames = resampler.resample(frame)
                    if not isinstance(resampled_frames, list):
                        resampled_frames = [resampled_frames]

                    for resampled_frame in resampled_frames:
                        for out_packet in output_stream.encode(resampled_frame):
                            output_container.mux(out_packet)
            
            for out_packet in output_stream.encode():
                output_container.mux(out_packet)

            output_container.close()
            input_container.close()
            
        except Exception as exc:
            err = True
            logging.exception("Error during conversion: %s", exc)
            self.output.emit("Error during conversion.")
        finally:
            self.data['final_file_fpath'] = out_fpath
            logging.info("Completed file conversion to: %s", out_fpath)
            msg = f"Completed '{self.data['target_format']}' conversion."
            self.output.emit(msg)

        return err
