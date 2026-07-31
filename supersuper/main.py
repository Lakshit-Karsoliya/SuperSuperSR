
from pathlib import Path
import shutil
import tempfile
import uuid

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from engine.srcnn import EngineSRCNN

app = FastAPI()

srcnn_engine = EngineSRCNN()


def cleanup(*files):
    for file in files:
        path = Path(file)
        if path.exists():
            path.unlink()


@app.post("/api/v1/srcnn/")
def srcnn(file: UploadFile = File(...)):
    temp_dir = Path(tempfile.gettempdir()) / "supersupersr"
    temp_dir.mkdir(exist_ok=True)

    uid = uuid.uuid4().hex

    input_path = temp_dir / f"{uid}_{file.filename}"
    output_path = temp_dir / f"{uid}_output.png"

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = srcnn_engine.execute(
        input_image=str(input_path),
        output_image=str(output_path)
    )

    if not result:
        cleanup(str(input_path))
        return {"status": "failed"}

    return FileResponse(
        path=str(output_path),
        media_type="image/png",
        filename="super_resolution.png",
    )