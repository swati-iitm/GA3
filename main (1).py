from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import json

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load student data from the specified JSON file
file_path = "q-vercel-python_2.json"

try:
    with open(file_path, "r") as file:
        students = json.load(file)  # Load data into `students`
except FileNotFoundError:
    students = []  # Default to an empty list if the file is missing

# Convert list of students to a dictionary for quick lookup
students_dict = {entry["name"]: entry["marks"] for entry in students}

@app.get("/")
async def get_students(name: Optional[List[str]] = Query(default=[])):
    """Fetch student marks based on optional name filtering while maintaining order."""
    if name:
        # Preserve the order in which names are passed
        filtered_marks = [students_dict.get(n, None) for n in name]
        return {"marks": filtered_marks}
    
    return {"marks": students}  # Return all data if no filter is applied

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
