from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# 로그인 페이지
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form["id"]
        password = request.form["password"]
        if user_id == "firefighter" and password == "1234":
            return redirect(url_for("select_member"))
        else:
            return render_template("login.html", error="Invalid ID or Password")
    return render_template("login.html")

# 대원 선택 페이지
@app.route("/select_member", methods=["GET", "POST"])
def select_member():
    members = ["대원 1", "대원 2", "대원 3", "대원 4"]
    if request.method == "POST":
        new_names = request.form.getlist("member_name")
        return render_template("select_member.html", members=new_names)
    return render_template("select_member.html", members=members)

if __name__ == "__main__":
    app.run(debug=True)
