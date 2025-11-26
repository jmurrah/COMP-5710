from flask import Flask
import sys

WELCOME_MESSAGE = "Welcome to a Simple Flask API!"
SQA_MESSAGE = "Welcome to the SQA course!"
SSP_MESSAGE = "Secure Software Process"
VANITY_MESSAGE = "Jacob Murrah"
MYPYTHON_MESSAGE = sys.version
CSSE_MESSAGE = "Department of Computer Science and Software Engineering"

app = Flask(__name__)


def as_h1(text):
    return f"<h1>{text}</h1>"


@app.route("/", methods=["GET"])
def home():
    return as_h1(WELCOME_MESSAGE)


@app.route("/sqa", methods=["GET"])
def greetSQA():
    return as_h1(SQA_MESSAGE)


@app.route("/ssp", methods=["GET"])
def greetSSP():
    return as_h1(SSP_MESSAGE)


@app.route("/vanity", methods=["GET"])
def greetVanity():
    return as_h1(VANITY_MESSAGE)


@app.route("/mypython", methods=["GET"])
def greetMyPython():
    return as_h1(MYPYTHON_MESSAGE)


@app.route("/csse", methods=["GET"])
def greetCSSE():
    return as_h1(CSSE_MESSAGE)


if __name__ == "__main__":
    app.run(debug=True)
