""" Module defines the FromWidget contained in the MainWindow """

# Copyright © 2023-2025 Walter van Heuven

import platform
import re
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import QMessageBox, QWidget, QPushButton, QLabel, \
                            QComboBox, QProgressBar, QTextBrowser, QSpacerItem, QSizePolicy, \
                            QHBoxLayout, QGridLayout
from PyQt6.QtCore import Qt, QCoreApplication
import utils as app_utils

class FormWidget(QWidget):
    """ Form Widget inside main Window """

    def __init__(self, parent):
        super(FormWidget, self).__init__(parent)
        self.layout = QGridLayout(self)
        self.mainWindow = parent
        self.overwriteDlg = None

        text: str
        w1 = QWidget()

        # Main button row
        hlayout0 = QHBoxLayout()

        text = "Select Audio/Video File(s)"
        self.button1 = QPushButton(text)
        self.button1.setDefault(True)
        self.button1.setFixedWidth(app_utils.str_width(self.button1, text) + self.mainWindow._TEXT_SPACE)
        self.button1.clicked.connect(self.mainWindow.select_files)
        self.button1.setToolTip("Select audio/video file(s).\nTo process all files in a folder, just drop the folder into the main window")
        #self.layout.addWidget(self.button1)
        hlayout0.addWidget(self.button1, Qt.AlignmentFlag.AlignLeft)

        # spacer item to allow resizing window
        hlayout0.addItem(QSpacerItem(275, 25, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum))

        text = "Start Speech Recognition"
        self.button2 = QPushButton(text)
        self.button2.setFixedWidth(app_utils.str_width(self.button2, text) + self.mainWindow._TEXT_SPACE)
        self.button2.clicked.connect(self.mainWindow.do_process_files)
        self.button2.setToolTip("Start speech recgonition process. Press Escape key to stop")
        self.button2.setDisabled(True)
        hlayout0.addWidget(self.button2, Qt.AlignmentFlag.AlignRight)

        w1.setLayout(hlayout0)
        #w1.setFixedHeight(50)

        self.layout.addWidget(w1, 0, 0, 1, 2) # Qt.AlignmentFlag.AlignLeft

        # Options buttons row
        hlayout1 = QHBoxLayout()

        text = "Output:"
        labelOutput1 = QLabel(text)
        labelOutput1.setFixedWidth(app_utils.str_width(labelOutput1, text))
        hlayout1.addWidget(labelOutput1, Qt.AlignmentFlag.AlignRight)

        self.comboOutput = QComboBox()
        output_items = self.mainWindow.output_items()
        self.comboOutput.addItems(output_items)
        self.comboOutput.setFixedWidth(100)
        self.comboOutput.setToolTip("Set the format of the output file")
        self.comboOutput.currentTextChanged.connect(self.mainWindow.update_output_setting)
        hlayout1.addWidget(self.comboOutput, Qt.AlignmentFlag.AlignLeft)

        text = "Language:"
        labelOutput2 = QLabel(text)
        labelOutput2.setFixedWidth(app_utils.str_width(labelOutput2, text))
        hlayout1.addWidget(labelOutput2, Qt.AlignmentFlag.AlignLeft)

        self.comboLanguage = QComboBox()
        self.comboLanguage.addItems(self.languages())
        self.comboLanguage.setToolTip("Select language or select 'Auto Detect' to automatically detect the source language")
        self.comboLanguage.currentTextChanged.connect(self.mainWindow.update_language_setting)
        self.comboLanguage.setFixedWidth(140)
        hlayout1.addWidget(self.comboLanguage, Qt.AlignmentFlag.AlignLeft)

        text = "Task:"
        labelOutput3 = QLabel(text)
        labelOutput3.setFixedWidth(app_utils.str_width(labelOutput3, text))
        hlayout1.addWidget(labelOutput3, Qt.AlignmentFlag.AlignRight)

        self.comboTask = QComboBox()
        self.set_tasks("default")
        self.comboTask.setFixedWidth(150)
        hlayout1.addWidget(self.comboTask, Qt.AlignmentFlag.AlignLeft)

        # spacer item to allow resizing window
        hlayout1.addItem(QSpacerItem(20, 25, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        w2 = QWidget()
        w2.setLayout(hlayout1)
        w2.setFixedHeight(50)
        self.layout.addWidget(w2, 1, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)

        # Progress section that is shown when whisper is processing audio file
        self.progress_widget = QWidget()
        hlayout_progress = QHBoxLayout()

        # Progress
        self.pbar = QProgressBar()
        self.pbar.setRange(0,0)
        self.pbar.setFixedHeight(15)
        self.pbar.setFixedWidth(400)
        #self.layout.addWidget(self.pbar, 2, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        self.pbar.hide()

        hlayout_progress.addWidget(self.pbar)

        text = "Cancel"
        self.cancel_button = QPushButton(text)
        self.cancel_button.setToolTip("Click to cancel processing file(s) or press Escape key")
        self.cancel_button.setFixedWidth(app_utils.str_width(self.cancel_button, text) + self.mainWindow._TEXT_SPACE)
        self.cancel_button.clicked.connect(self.mainWindow.stop_processing)
        self.cancel_button.setEnabled(True)
        self.cancel_button.hide()

        hlayout_progress.addWidget(self.cancel_button)
        self.progress_widget.setLayout(hlayout_progress)
        self.layout.addWidget(self.progress_widget, 2, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)

        # hide it
        self.progress_widget.hide()

        # Feedback text browser area
        self.terminal = QTextBrowser()

        if platform.system() == "Windows":
            self.terminal.setFont(QFont("Menlo", 10))
        else:
            self.terminal.setFont(QFont("Menlo", 12))

        
        self.terminal.setOpenExternalLinks(True)
        #self.terminal.anchorClicked.connect(handle_links)
        self.terminal.setAcceptRichText(False)
        self.terminal.setUndoRedoEnabled(False)
        self.terminal.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | 
            Qt.TextInteractionFlag.LinksAccessibleByMouse | 
            Qt.TextInteractionFlag.TextEditorInteraction
        )
        self.terminal.setReadOnly(True)
        self.terminal.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        #self.terminal.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        #self.terminal.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.terminal.setMinimumHeight(350)
        self.terminal.setMinimumWidth(700)
        self.layout.addWidget(self.terminal, 3, 0, 1, 2)

        # Version
        self.info = QLabel(f"<a href='{self.mainWindow.WEBSITE_URL}'>{self.mainWindow.APP_NAME}</a> v{self.mainWindow.VERSION}")
        self.info.setOpenExternalLinks(True)
        self.layout.addWidget(self.info, 4, 0, 1, 1)

        hlayout2 = QHBoxLayout()

        text = "More info"
        self.button3_info = QPushButton(text)
        self.button3_info.setToolTip(f"More info about {self.mainWindow.APP_NAME}")
        self.button3_info.setFixedWidth(app_utils.str_width(self.button3_info, text) + self.mainWindow._TEXT_SPACE)
        self.button3_info.clicked.connect(lambda: self.mainWindow.show_app_info(False))
        self.layout.addWidget(self.button3_info, 4, 1, 1, 1)
        hlayout2.addWidget(self.button3_info, Qt.AlignmentFlag.AlignLeft)

        hlayout2.addItem(QSpacerItem(130, 25, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum))

        text = "Settings"
        self.buttonWhisperURL = QPushButton(text)
        self.buttonWhisperURL.setToolTip("Change Settings")
        self.buttonWhisperURL.setFixedWidth(app_utils.str_width(self.buttonWhisperURL, text) + self.mainWindow._TEXT_SPACE)
        self.buttonWhisperURL.clicked.connect(self.mainWindow.show_settings_dlg)
        hlayout2.addWidget(self.buttonWhisperURL, Qt.AlignmentFlag.AlignRight)

        text = "Output Folder"
        self.buttonOutput = QPushButton(text)
        self.buttonOutput.setToolTip("Open Output Folder")
        self.buttonOutput.setFixedWidth(app_utils.str_width(self.buttonOutput, text) + self.mainWindow._TEXT_SPACE)
        self.buttonOutput.clicked.connect(self.mainWindow.open_output_folder)
        hlayout2.addWidget(self.buttonOutput, Qt.AlignmentFlag.AlignRight)

        w3 = QWidget()
        w3.setLayout(hlayout2)
        self.layout.addWidget(w3, 4, 1, 1, 1)

        self.setLayout(self.layout)

    def languages(self) -> list[str]:
        # all languages supported by OpenAI's Whisper
        languages = ["Auto Detect", "Afrikaans","Albanian","Amharic","Arabic","Armenian","Assamese","Azerbaijani","Bashkir","Basque",\
                     "Belarusian","Bengali","Bosnian","Breton","Bulgarian","Burmese","Castilian","Catalan","Chinese", \
                     "Croatian","Czech","Danish","Dutch","English","Estonian","Faroese","Finnish","Flemish","French", \
                     "Galician","Georgian","German","Greek","Gujarati","Haitian","Haitian Creole","Hausa","Hawaiian", \
                     "Hebrew","Hindi","Hungarian","Icelandic","Indonesian","Italian","Japanese","Javanese","Kannada", \
                     "Kazakh","Khmer","Korean","Lao","Latin","Latvian","Letzeburgesch","Lingala","Lithuanian","Luxembourgish", \
                     "Macedonian","Malagasy","Malay","Malayalam","Maltese","Maori","Marathi","Moldavian","Moldovan","Mongolian", \
                     "Myanmar","Nepali","Norwegian","Nynorsk","Occitan","Panjabi","Pashto","Persian","Polish","Portuguese", \
                     "Punjabi","Pushto","Romanian","Russian","Sanskrit","Serbian","Shona","Sindhi","Sinhala","Sinhalese","Slovak", \
                     "Slovenian","Somali","Spanish","Sundanese","Swahili","Swedish","Tagalog","Tajik","Tamil","Tatar","Telugu",\
                     "Thai","Tibetan","Turkish","Turkmen","Ukrainian","Urdu","Uzbek","Valencian","Vietnamese","Welsh","Yiddish","Yoruba"]

        return languages

    def set_content(self) -> None:
        settings = self.mainWindow.settings

        self.comboOutput.setCurrentText(settings.value("Settings/Output"))
        self.comboLanguage.setCurrentText(settings.value("Settings/Language"))
        #self.comboTask.setCurrentText(settings.value("Settings/Task"))

    def set_enabled_gui_elements(self, status: bool) -> None:
        # disable/enable gui elements during processing
        self.button1.setEnabled(status)
        self.button2.setEnabled(status)
        self.comboOutput.setEnabled(status)
        self.comboLanguage.setEnabled(status)
        self.comboTask.setEnabled(status)
        self.buttonWhisperURL.setEnabled(status)
        self.button3_info.setEnabled(status)

        # menu
        self.mainWindow.menu_open_action.setEnabled(status)
        self.mainWindow.start_action.setEnabled(status)

        # update
        QCoreApplication.processEvents()

    def set_tasks(self, set_to_task):
        default_tasks = ["Transcribe", "Translate"]
        if set_to_task.lower() == 'default':
            if self.comboTask.count() != 2:
                if self.comboTask.receivers(self.comboTask.currentTextChanged) > 0:
                    self.comboTask.currentTextChanged.disconnect()
                self.comboTask.clear()
                self.comboTask.addItems(default_tasks)
                self.comboTask.currentTextChanged.connect(self.mainWindow.update_task_options)
                self.comboTask.setToolTip("Select task:\n1) Transcribe into language selected (if different from language spoken, \nASR with multilingual model will translate into that language) or \n2) Translate into English")
        else:
            if self.comboTask.count() == 2:
                if self.comboTask.receivers(self.comboTask.currentTextChanged) > 0:
                    self.comboTask.currentTextChanged.disconnect()
                self.comboTask.clear()
                self.comboTask.addItems([set_to_task])
                self.comboTask.currentTextChanged.connect(self.mainWindow.update_task_options)
                self.comboTask.setToolTip("Transcribe into source language")

    def set_show_progress_element(self, show: bool) -> None:
        # progress bar and stop button visible during processing
        if show:
            self.progress_widget.show()
            self.pbar.show()
            self.cancel_button.show()
        else:
            self.progress_widget.hide()
            self.pbar.hide()
            self.cancel_button.hide()

    def ask_to_overwrite(self, out_filename: str, n_files: int) -> None:
        self.overwriteDlg = CustomMessageBox(self)

        self.overwriteDlg.setIcon(QMessageBox.Icon.Warning)
        self.overwriteDlg.setWindowModality(Qt.WindowModality.WindowModal)
        self.overwriteDlg.setWindowTitle("File(s) Already Exist")
        self.overwriteDlg.setText(f"Overwrite existing file: '{out_filename}'?")

        if n_files > 1:
            self.overwriteDlg.setStandardButtons(
                QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.NoToAll |
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.YesToAll
            )
        else:
            self.overwriteDlg.setStandardButtons(
                QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Yes
            )
        
        self.overwriteDlg.setDefaultButton(QMessageBox.StandardButton.No)

        return self.overwriteDlg.exec()

    def show_info(self, msg: str) -> None:
        self.terminal.insertHtml(msg)

    def remove_last_line(self):
        cursor = self.terminal.textCursor()

        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.select(QTextCursor.SelectionType.LineUnderCursor)
        cursor.removeSelectedText()

        # QTextCursor.MoveOperation.PreviousBlock
        #cursor.movePosition(QTextCursor.MoveOperation.End)
        #cursor.movePosition(
        #    QTextCursor.MoveOperation.StartOfLine,
        #    QTextCursor.MoveMode.KeepAnchor, 1
        #)
        #cursor.removeSelectedText()

        #cursor.movePosition(QTextCursor.MoveOperation.End)
        #cursor.movePosition(
        #    QTextCursor.MoveOperation.StartOfLine,
        #    QTextCursor.MoveMode.KeepAnchor, 1
        #)
        #cursor.removeSelectedText()

        self.terminal.setTextCursor(cursor)

    def feedback(self, msg: str, add_newline: bool=True, check_progress_bar: bool=False) -> None:
        if add_newline:
            self.terminal.append(msg) # newline added
            self.terminal.moveCursor(QTextCursor.MoveOperation.End)
            self.terminal.ensureCursorVisible()
        elif check_progress_bar:
            # get last line
            cursor = self.terminal.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.movePosition(
                QTextCursor.MoveOperation.PreviousBlock,
                QTextCursor.MoveMode.KeepAnchor, 1
            )
            # check if this is a progress bar
            if re.search(r'(\d+%|\d+/\d+|\d+\.\d+/\d+\.\d+).*?\[.+?\]', cursor.selectedText()):
                cursor.movePosition(QTextCursor.MoveOperation.End)
                cursor.movePosition(
                    QTextCursor.MoveOperation.Up,
                    QTextCursor.MoveMode.KeepAnchor, 1
                )
                cursor.removeSelectedText()
                self.terminal.setTextCursor(cursor)

                self.terminal.append(msg.strip())
            else:
                self.terminal.append(msg.strip()) # newline added
                
            self.terminal.moveCursor(QTextCursor.MoveOperation.End)
            self.terminal.ensureCursorVisible()

        else:
            # no newline
            self.terminal.moveCursor(QTextCursor.MoveOperation.End)
            self.terminal.insertPlainText(msg)
            self.terminal.moveCursor(QTextCursor.MoveOperation.End)
            self.terminal.ensureCursorVisible()

        # move scroll bar to see feedback
        horScrollBar = self.terminal.horizontalScrollBar()
        verScrollBar = self.terminal.verticalScrollBar()
        verScrollBar.setValue(verScrollBar.maximum())
        horScrollBar.setValue(0)

    def delete_feedback(self: QWidget) -> None:
        self.terminal.setText("")

    def write(self, s) -> None:
        """ function used to capture stdio """
        if isinstance(s, str):
            s = s.strip()
            if len(s) > 0:
                self.feedback(s)
        elif isinstance(s, bytes):
            s = s.decode("utf-8")
            s = s.strip()
            if len(s) > 0:
                self.feedback(s)

    def resizeMe(self) -> None:
        self.resize(self.minimumSizeHint())

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
