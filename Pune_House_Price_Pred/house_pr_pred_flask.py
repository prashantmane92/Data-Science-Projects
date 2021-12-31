from flask import Flask,request,render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("hpp_lr_pipe.pkl",'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/pred",methods=["post"])
def prediction():
    data = request.form
    total_sqft=float(data["total_sqft"])
    availability=float(data["availability"])
    bath=float(data["bath"])
    balcony=float(data["balcony"])
    Bedrooms=float(data["Bedrooms"])
    area_type=data["area_type"]
    site_location=data["site_location"]
    sample = [[total_sqft,availability,bath,balcony,Bedrooms,area_type,site_location]]
    ans = model.predict(sample)
    return render_template("index.html", result = ans[0])

if __name__ == "__main__":
    app.run(debug=True)