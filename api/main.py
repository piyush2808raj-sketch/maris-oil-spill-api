from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse

from inference.pipeline import run_pipeline


app = FastAPI(
    title="MARIS Oil Spill Detection API",
    description="AI-powered Sentinel-1 SAR oil spill detection and segmentation API.",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "project": "MARIS",
        "description": "Maritime Oil Spill Detection and Segmentation API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported."
        )

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="maris_"
        )
    )

    input_path = (
        temp_dir /
        file.filename
    )

    try:
        with open(
            input_path,
            "wb"
        ) as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        result = run_pipeline(
            input_path
        )

        # Convert local filesystem paths
        # into API-accessible URLs.
        job_id = result["job_id"]

        files = result["files"]

        for key, path in files.items():
            if path is not None:
                filename = Path(path).name

                files[key] = str(
                    request.url_for(
                     "get_result_file",
                    job_id=job_id,
                    filename=filename
                )
)

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


@app.get("/result/{job_id}/{filename}")
def get_result_file(
    job_id: str,
    filename: str
):
    allowed_files = {
        "predicted_mask.png",
        "overlay.png",
        "result.json"
    }

    if filename not in allowed_files:
        raise HTTPException(
            status_code=400,
            detail="Invalid output file."
        )

    base_dir = (
        Path(__file__).resolve().parent.parent
    )

    file_path = (
        base_dir /
        "inference" /
        "output" /
        job_id /
        filename
    )

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="File not found."
        )

    return FileResponse(
        file_path
    )
