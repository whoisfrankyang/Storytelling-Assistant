from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from ragcot import RAGSystem
from pydantic import BaseModel
import PyPDF2
import io


app = FastAPI()
rag = RAGSystem()

# Serve static files (like script.js)
app.mount("/static", StaticFiles(directory="static"), name="static")

  
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve HTML templates
templates = Jinja2Templates(directory="templates")

class InputText(BaseModel):
    input_data: str

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/run")
async def run_storytelling(input: InputText, mode: str = "general"):
    text = input.input_data

    # Call RAG system
    general_version = rag.generate_storytelling_output(
        user_abstract=text,
        mode=mode,
        k=3
    )

    return {"result": general_version}

@app.post("/process")
async def process_file(file: UploadFile = File(...), mode: str = Form("general")):
    try:
        content = await file.read()
        if file.filename.lower().endswith('.pdf'):
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        else:
            # Assume text file
            text = content.decode('utf-8')
        
        result = rag.generate_storytelling_output(
            user_abstract=text,
            mode=mode,
            k=3
        )
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}