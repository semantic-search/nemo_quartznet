from __future__ import absolute_import, division, print_function
from fastapi import FastAPI, File, UploadFile, Form, Response, status
import subprocess
import os
app = FastAPI()
@app.post("/uploadfile/")
def create_upload_file(file: UploadFile = File(...)):
    file_name = file.filename
    with open(file_name, 'wb') as f:
        f.write(file.file.read())
    subprocess.call(["python", "stt.py", "--audio", file_name])
    with open(file_name+".txt") as tran:
        transcription = tran.read()
    os.remove(file_name)
    os.remove(file_name+".json")
    os.remove(file_name+".txt")
    return transcription
