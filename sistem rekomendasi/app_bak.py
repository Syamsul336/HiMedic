from flask import Flask, render_template, request


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')
deteksi = 0
@app.route("/predict", methods=["POST"])
def predict():
    global deteksi
    deteksi += 1
    return render_template('index.html', deteksi=deteksi)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)