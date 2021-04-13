import os, json, librosa
from flask import Flask, render_template, request, redirect
from inference import inference
import soundfile as sf


# app = Flask(__name__)
app = Flask(__name__, static_folder="/Users/zombie/Google Drive/Research/Dereverbify")
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    with open('config.json') as f:
        data=f.read()
    stftParams=json.loads(data)['stftParams']
    data=0


    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        file, _ =librosa.load(file, sr=16000)
        sf.write("original.wav", file, 16000)
        genAudio, genSpec = inference(None, file, stftParams, checkpoint="/Users/zombie/Downloads/colabBatchSize8_5000.pt")
        return render_template('result.html', genAudio=genAudio, genSpec=genSpec)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
