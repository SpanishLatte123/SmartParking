from flask import Flask, jsonify, render_template
import pandas as pd

app = Flask(__name__)

# Define the Excel file path
excel_file_path = "outputs/parking_data.xlsx"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/parking_status", methods=["GET"])
def parking_status():
    try:
        # Load only the latest row from the Excel file
        df = pd.read_excel(excel_file_path, sheet_name="Status_Log")
        latest_row = df.iloc[-1].to_dict()  # Get the last logged row
        timestamp = latest_row.pop("Timestamp")  # Extract timestamp separately
        data = {"timestamp": timestamp, "spaces": latest_row}
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Bind to 0.0.0.0 for network access
    app.run(host="0.0.0.0", port=5000, debug=True)
