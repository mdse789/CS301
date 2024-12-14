from dash import Dash, dcc, html, Input, Output
import base64
import io
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

app = Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1("Milestone 4"),
        dcc.Upload(
            id="upload-data",
            children=html.Div([ html.A("Upload File")]),
            multiple=False,
        ),
       
        dcc.Store(id="processed-data"),  
        html.Div(
            children=[
                html.Label("Select Target: ", className = "label-target"),
                dcc.Dropdown(id="target-variable", placeholder="Select Target: "),
            ],
            className = "target_div",
        ),
      
        
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                dcc.RadioItems(id="categorical-variable"),
                            ],
                            className = "label-cat"
                        ),
                        dcc.Graph(id="chart1"),
                    ],
                    className= "cater"
                ),
                html.Div(
                    children=[
                        dcc.Graph(id="chart2"),
                    ],
                    className = "corrers"
                ),
            ],
            className = "charts",
        ),


        html.Div(
            children=[
                html.Div(
                    children=[
                       
                             dcc.Checklist(id="train-variable"),
                            ],
                 ),
                html.Button('Train', id='train-model', n_clicks=0),
            ],
            className = "trains"
        ),

        
        html.Div(id='model-eval', children=""),
    
        
        
                html.Div(
                    children=[  
                        dcc.Input(id="input-pred", type="text", placeholder='Input Values for Prediction - comma separated'),
                        html.Button('Predict', id='pred-model', n_clicks=0),
                        html.Div(id='pred-out', children="Prediction: None"),
                    ],
                    className="preders"    
                ),
            
    ]
    
)

def preprocess_data(df):
    #droppping null values
    #need to  do encoding?
    return df.dropna()

@app.callback(
    [ 
     Output("processed-data", "data"),
     Output("target-variable", "options"),
     Output("categorical-variable", "options"),
     Output("train-variable", "options")],
    Input("upload-data", "contents"),

)
def upload_and_process(contents):
    if contents is None:
        #in case of no file
        return "No file uploaded", [], [], []
   
    try:
        
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string).decode("utf-8")
        df = pd.read_csv(io.StringIO(decoded))

        processed_df = preprocess_data(df)

        #separating numerical and categorical 
        numerical_cols = processed_df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = processed_df.select_dtypes(exclude=["number"]).columns.tolist()
        all_cols =  processed_df.columns.tolist()

        
        pross_df = base64.b64encode(io.StringIO(processed_df.to_csv(index=False)).getvalue().encode()).decode()

        # for dropdown menu
        target_options = [{"label": col, "value": col} for col in numerical_cols]
        categorical_options = [{"label": col, "value": col} for col in categorical_cols]
        trainopt = [{"label": col, "value": col} for col in all_cols]

        return  pross_df , target_options, categorical_options, trainopt

    except Exception as e:
        return f"Error processing file: {str(e)}", [], []

# For the barchart
@app.callback(
    [Output("chart1", "figure"), 
     Output("chart2", "figure")],
    [Input("processed-data", "data"), 
     Input("target-variable", "value"), 
     Input("categorical-variable", "value")],
)

def update_charts(data, target_var, cat_var):
    if data is None or target_var is None:
        return {}, {}

    decoded_data = base64.b64decode(data).decode("utf-8") 
    df = pd.read_csv(io.StringIO(decoded_data))

    

    if cat_var:
        grouped_df = df.groupby(cat_var)[target_var].mean().reset_index()
        chart1 = go.Figure(data=[go.Bar(
            x=grouped_df[cat_var],
            y=grouped_df[target_var],
            text=grouped_df[target_var],
            textposition='inside',  
            marker=dict(color="rgba(112, 186, 226, 0.3)") 
        )])
        chart1.update_layout(
            title=f"Average {target_var} by {cat_var}",
            plot_bgcolor="white",   
            title_x=0.5, 
         )
        chart1.update_traces(textposition='inside')

    else:
        chart1 = {}


    num_only = df.select_dtypes(include=["number"]).columns.tolist()
    correlation = df[num_only].corr()[target_var].abs().drop(target_var).reset_index()
    correlation.columns = ["Numerical Variable", "Correlation Strength (Absolute Value)"]
    chart2 = go.Figure(data=[go.Bar(
    
        x=correlation["Numerical Variable"],
        y=correlation["Correlation Strength (Absolute Value)"],
        text=correlation["Correlation Strength (Absolute Value)"],
        textposition='inside', 
    )])

    chart2.update_layout(
        title=f"Correlation Strength of Numerical Variables with {target_var}",
        paper_bgcolor="white", 
        title_x=0.5 
    )

    return chart1, chart2


@app.callback(
    [Output('model-eval', 'children')],
    [Input('train-variable', 'value'),
     Input('train-model', 'n_clicks'),
     Input("target-variable", "value"),
     Input('processed-data', 'data')],
)


def train_model(train_variables, n_clicks, targvar, data):
    if n_clicks > 0:
        if not train_variables:
            return ["No train variables selected"]

        
        decoded_data = base64.b64decode(data).decode("utf-8")  
        processed_df = pd.read_csv(io.StringIO(decoded_data))
        
        X = processed_df[train_variables]
        y = processed_df[targvar]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

        estimator_1 = DecisionTreeRegressor()
        pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ('model', BaggingRegressor(estimator=estimator_1, n_estimators=10, random_state=42))
        ])

    
        pipe.fit(X_train, y_train)

        predictions = pipe.predict(X_test)

        r2_bagg = r2_score(y_test, predictions)
        
        return [f"R2 Score: {r2_bagg:.5f}"]
    #if train_variables is None:
            #return ["No train variables selected"]
    return[""]

#@app.callback(
 #   Output('selected-target-variable', 'children'),
  #  [Input("target-variable", "value")],
#)

#def display_selected_target(targvar):
 #   if targvar:
  #      return f"Selected Target: {targvar}"
   # return "Selected Target: None"

@app.callback(
    Output('pred-out', 'children'),
    [Input('pred-model', 'n_clicks'),
     Input('input-pred', 'value'),
     Input('processed-data', 'data'), 
     Input("train-variable", "value"), 
     Input("target-variable", "value"),
     Input("model-eval", "children")],
)

def predictions (n_clicks, input_pred, data, train_variables, targvar, model_eval):
    if n_clicks > 0:
        if not model_eval:
            return "Model is not trained yet."
        
        if train_variables is None:
            return ["No train variables selected"]
    
        if input_pred is None or input_pred.strip() == "":
            return "No values for prediction provided"

        wrongval = False

        pred_values = []    
        for i in input_pred.split(","):
            i = i.strip()
            if i.replace('.', "", 1).isdigit():
                pred_values.append(float(i))
            elif i.isalpha():
                pred_values.append(i)
            else:
                wrongval = True
                break
    
        if wrongval:
            return "Enter letters or numbers only"
        
        if len(pred_values) != len(train_variables):
            return("Number of selected Features and prediction input do not match")
    
        decoded_data = base64.b64decode(data).decode("utf-8")  
        processed_df = pd.read_csv(io.StringIO(decoded_data))
        X = processed_df[train_variables]
        y = processed_df[targvar]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
        estimator_1 = DecisionTreeRegressor()
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
            ('model', BaggingRegressor(estimator=estimator_1, n_estimators=10, random_state=42))
            ])

        
        pipe.fit(X_train, y_train)

        preds = pipe.predict([pred_values])
        return f"Predicted {targvar}: {preds[0]:.5f}"

    #return "Preds: nono"


if __name__ == "__main__":
    app.run(debug=False)



