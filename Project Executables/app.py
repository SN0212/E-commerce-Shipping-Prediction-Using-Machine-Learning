from flask import Flask, render_template, request
import pandas as pd
import joblib 
from sklearn.preprocessing import StandardScaler

# Initialize Flask application
app = Flask(__name__)

# Load your trained model 
model = joblib.load('rf_acc_68.pkl')  
norm = joblib.load('normalizer.pkl')  

# Define label mapping function
def map_labels(data):
    label_map = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            label_map[column] = {label: idx for idx, label in enumerate(data[column].unique())}
            data[column] = data[column].map(label_map[column])
    return data, label_map

# Function to process input data for prediction
def process_input(data):
    new_data = pd.DataFrame(data, index=[0])  
    new_data, _ = map_labels(new_data)
    
    # Normalize new data
    new_data = norm.transform(new_data)
    
    return new_data

# Route to handle home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction request
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        warehouse_block = request.form['Warehouse_block']
        mode_of_shipment = request.form['Mode_of_Shipment']
        customer_care_calls = int(request.form['Customer_care_calls'])
        customer_rating = int(request.form['Customer_rating'])
        cost_of_the_product = float(request.form['Cost_of_the_Product'])
        prior_purchases = int(request.form['Prior_purchases'])
        product_importance = request.form['Product_importance']
        gender = request.form['Gender']
        discount_offered = float(request.form['Discount_offered'])
        weight_in_gms = float(request.form['Weight_in_gms'])
        
        # Create a dictionary of input data
        input_data = {
            'Warehouse_block': warehouse_block,
            'Mode_of_Shipment': mode_of_shipment,
            'Customer_care_calls': customer_care_calls,
            'Customer_rating': customer_rating,
            'Cost_of_the_Product': cost_of_the_product,
            'Prior_purchases': prior_purchases,
            'Product_importance': product_importance,
            'Gender': gender,
            'Discount_offered': discount_offered,
            'Weight_in_gms': weight_in_gms
        }
        
        # Process input data for prediction
        processed_data = process_input(pd.DataFrame(input_data, index=[0]))
        
        # Predict using the trained model
        prediction = model.predict(processed_data)
        prediction_text = 'Order will reach on time' if prediction[0] == 1 else 'Order will not reach on time'
        
        # Render prediction result template with prediction text
        return render_template('index.html', prediction_text=prediction_text)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
