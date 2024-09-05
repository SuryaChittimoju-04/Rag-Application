from flask import Flask, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from fastapi import FastAPI, Request, Form, Response

from rag import get_response

app = Flask(__name__)

# Define the folder where PDF files will be stored
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

# Route to accept PDF file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return 'File uploaded successfully'
    else:
        return 'Invalid file format'
    
@app.post("/get_response")
async def get_response_from_model(query: str = Form(...)):
    print(query)
    return Response(get_response(query))

if __name__ == '__main__':
    app.run(debug=True)
