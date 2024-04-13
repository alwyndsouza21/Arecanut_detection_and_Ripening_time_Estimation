from flask import Flask, render_template,request

# Creating an app instance
app = Flask(__name__)

# Creating a route
@app.route("/")
def hello():
    return render_template("start_page.html")

@app.route("/detect",methods=['GET', 'POST'])
def detect():
  if request.method == 'POST':
  # Handle image detection here
    pass
  return render_template("detection_page.html")
  
@app.route("/estimate",methods=["GET","POST"])
def estimate():
  if request.method=="POST":
    #handle image estimation here
    pass
  return render_template("estimation_page.html")


if __name__ == '__main__':
    # Running the app with specified host and port
    app.run(debug=True, host='0.0.0.0', port=8080)
