""" Module defines the StreamEmitter class """

# Copyright © 2023-2025 Walter van Heuven

import re
from PyQt6.QtCore import QThread, pyqtSignal

class StreamEmitter(QThread):
    message = pyqtSignal(str)

    ANSI_ESCAPE_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message: str, encoding='utf-8') -> None:
        if isinstance(message, bytes):
            message = message.decode(encoding)
        self._process_line(message)

    def _process_line(self, line: str):
        line = line.rstrip("\n")
        line = self.ANSI_ESCAPE_PATTERN.sub('', line)
        if line:
            self.message.emit(line)
    
    def flush(self) -> None:
        """ required for stdout and stderr """
        pass
