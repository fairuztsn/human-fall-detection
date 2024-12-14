from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import random
import json

csv_file = "./data/data_train.csv"
data = pd.read_csv(csv_file)

if 'revalidated_manually' not in data.columns:
    data['revalidated_manually'] = 0

data_to_validate = data[data['path'].str.contains('\\fall\\', regex=False)].copy()
data_to_validate = data_to_validate[data_to_validate['label'] == 0]

app = FastAPI()
app.mount("/images", StaticFiles(directory="."), name="images")

class LabelInput(BaseModel):
    index: int
    label: int

@app.get("/", response_class=HTMLResponse)
async def read_item():
    remaining_data = data_to_validate[data_to_validate["revalidated_manually"] == False]
    
    if remaining_data.empty:
        print(len(remaining_data))
        return HTMLResponse("<h1>Semua gambar sudah divalidasi!</h1>")

    current_index = remaining_data.index[random.randint(0, len(remaining_data)-1)]
    current_image_path = remaining_data.loc[current_index, "path"]
    current_image_label = remaining_data.loc[current_index, "label"]
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Revalidasi Data</title>
    </head>
    <body>
        <h1>Revalidasi Data</h1>
        <p>{len(remaining_data)} images left.</p>
        <p>{current_image_path.split('\\')[-2][2:]}, src: <a href="{current_image_path}" target="_blank">{current_image_path}</a></p>
        <p>Detected as {"Fall" if current_image_label else "Non-Fall"}</p>
        <img src="/images/{current_image_path}" alt="Image" style="max-width: 600px; max-height: 400px;"/>
        <br/>
        <button onclick="submitLabel({current_index}, 1)">Fall</button>
        <button onclick="submitLabel({current_index}, 0)">Non Fall</button>
        <button onclick="submitLabel({current_index}, -1)">Skip</button>
        <script>
            async function submitLabel(index, label) {{
                const response = await fetch('/submit', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{ index: index, label: label }})
                }});
                const result = await response.json();
                if (result.success) {{
                    window.location.reload();
                }} else {{
                    alert("Gagal menyimpan data.");
                }}
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# @app.post("/save")
# async def save_data():
#     try:
#         data.to_csv("../data/data_train.csv", index=False)

#         return JSONResponse({"success": True})
#     except Exception as e:
#         return JSONResponse({"success": False, "error": str(e)})

@app.post("/submit")
async def submit_label(label_input: LabelInput):
    try:
        index = label_input.index
        label = label_input.label

        if label in [0,1]:
            data_to_validate.at[index, "label"] = label

        data_to_validate.at[index, "revalidated_manually"] = int(label in [0, 1])

        data.update(data_to_validate)
        data.to_csv(csv_file, index=False)

        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})