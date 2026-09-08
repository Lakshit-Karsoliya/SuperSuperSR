<div align="center"><img width=300px src='assets/logo.png'/></div>


**SuperSuperSR** is a FastAPI-based image super-resolution API that provides a simple and extensible interface for running deep learning super-resolution models. 

---

## Currently Supported Models

* SRCNN 
* FSRCNN 

> More super-resolution models will be added in future releases.

## Results 
**original left** 
**upscaled right**

<div align="center">SRCNN</div>
<div align="center"><img width=300px src='assets/srcnn_results.png'/></div>
<div align="center">FSRCNN</div>
<div align="center"><img width=300px src='assets/fsrcnn_results.png'/></div>

---

## ▶️ Running the API

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the from the project root:

```bash
uvicorn main:app --reload
```

---


