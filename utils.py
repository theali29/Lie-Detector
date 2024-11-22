import tkinter as tk
from tkinter import filedialog

def get_video_file():
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.wmv")])
    return filename
