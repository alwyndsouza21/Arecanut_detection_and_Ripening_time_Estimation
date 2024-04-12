from flask import Flask, render_template

# Creating an app instance
app = Flask(__name__)

# Creating a route
@app.route("/")
def hello():
    return render_template("start_page.html")

if __name__ == '__main__':
    # Running the app with specified host and port
    app.run(debug=True, host='0.0.0.0', port=8080)
