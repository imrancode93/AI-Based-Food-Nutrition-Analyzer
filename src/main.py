from fastapi import FastAPI
from src.ai_model import get_nutrition_info

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "AI-Based Food Nutrition Analyzer is running!"}

@app.get("/analyze/{food_item}")
async def analyze_food(food_item: str):
    result = get_nutrition_info(food_item)
    return {"food": food_item, "nutrition_info": result}
