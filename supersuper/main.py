from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from .engine.srcnn import EngineSRCNN



app = FastAPI(
    title="SuperSuperSR",
    description="API for Super Resolution Models",
    version="1.0.0"
)


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home():

    return """
    <html>
        <body>
            <h1>SuperSuperSR</h1>
            <p>Server running successfully</p>
        </body>
    </html>
    """


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class SRCNNRequest(BaseModel):

    input_image: str
    output_image: str


# --------------------------------------------------
# ENGINE
# --------------------------------------------------

srcnn_engine = EngineSRCNN()


# --------------------------------------------------
# API
# --------------------------------------------------

@app.post(
    "/api/v1/srcnn/",
    summary="Run SRCNN Super Resolution",
    description="Upscales image using SRCNN model",
    tags=["Convolution Based SR"]
)
async def srcnn(data: SRCNNRequest):

    result = await srcnn_engine.execute(
        input_image=data.input_image,
        output_image=data.output_image
    )

    return result