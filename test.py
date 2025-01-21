from flask import Flask

app = Flask(__name__)

@app.route('/about')
def home():
    return "Hello, Flask! I am running from Taher's laptop"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
