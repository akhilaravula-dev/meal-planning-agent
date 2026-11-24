import asyncio
import logging
import time
from typing import AsyncGenerator

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from google.genai.errors import APIError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model fallback: if one Gemini model is out of quota (429), overloaded (503)
# or unavailable (404), automatically try the next one in the list.
# ---------------------------------------------------------------------------

# Lite models first: they have the highest free-tier limits, so they fail least.
FALLBACK_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-4-hydrogen",
]

FALLBACK_STATUS_CODES = {404, 429, 500, 502, 503, 504}
ROUNDS = 2                 # go through the whole list this many times
WAIT_BETWEEN_ROUNDS = 8    # seconds; 503 "busy" errors are usually brief
QUOTA_COOLDOWN = 15 * 60   # skip a model for 15 min after a 429 / 404

_cooldown_until: dict[str, float] = {}


class FallbackGemini(Gemini):
    """Gemini model that falls back to other models when one fails."""

    fallback_models: list[str] = FALLBACK_MODELS

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        errors: list[APIError] = []
        for round_no in range(ROUNDS):
            if round_no:
                logger.warning("All models failed; waiting %ss and retrying.", WAIT_BETWEEN_ROUNDS)
                await asyncio.sleep(WAIT_BETWEEN_ROUNDS)

            now = time.time()
            models = [m for m in self.fallback_models if _cooldown_until.get(m, 0) <= now]
            if not models:  # everything is cooling down: try them all anyway
                models = list(self.fallback_models)

            for model_name in models:
                request = llm_request.model_copy(deep=True)
                request.model = model_name
                started = False
                try:
                    async for response in super().generate_content_async(request, stream):
                        started = True
                        yield response
                    return
                except APIError as e:
                    # Only switch models if nothing has been sent to the user yet.
                    if started or e.code not in FALLBACK_STATUS_CODES:
                        raise
                    logger.warning("Model %s failed (%s); trying next model.", model_name, e.code)
                    if e.code in (404, 429):  # out of quota / retired: don't retry soon
                        _cooldown_until[model_name] = time.time() + QUOTA_COOLDOWN
                    errors.append(e)

        # Show the most useful error: "busy" or "quota" rather than "not found".
        errors.sort(key=lambda e: {503: 0, 504: 0, 500: 0, 502: 0, 429: 1}.get(e.code, 2))
        raise errors[0]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def calculate_calories(protein_g: float, carbs_g: float, fat_g: float) -> dict:
    """Calculate estimated calories from protein, carbs and fat in grams.

    Args:
        protein_g: Grams of protein.
        carbs_g: Grams of carbohydrates.
        fat_g: Grams of fat.

    Returns:
        A dict with the estimated calories.
    """
    calories = protein_g * 4 + carbs_g * 4 + fat_g * 9
    return {"estimated_calories": round(calories, 1)}


ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}

GOAL_ADJUSTMENTS = {
    "lose": -500,
    "maintain": 0,
    "gain": 300,
}


def calculate_daily_targets(
    weight_kg: float,
    height_cm: float,
    age: int,
    sex: str,
    activity_level: str,
    goal: str,
) -> dict:
    """Estimate daily calorie and macro targets (Mifflin-St Jeor equation).

    Use this when the user has not given a calorie target but has shared
    their body stats.

    Args:
        weight_kg: Body weight in kilograms.
        height_cm: Height in centimetres.
        age: Age in years.
        sex: "male" or "female".
        activity_level: One of sedentary, light, moderate, active, very_active.
        goal: One of lose, maintain, gain.

    Returns:
        A dict with BMR, TDEE, calorie target and protein/carb/fat targets in grams.
    """
    sex = sex.strip().lower()
    activity_level = activity_level.strip().lower().replace(" ", "_")
    goal = goal.strip().lower()

    if activity_level not in ACTIVITY_MULTIPLIERS:
        return {"error": f"activity_level must be one of {list(ACTIVITY_MULTIPLIERS)}"}
    if goal not in GOAL_ADJUSTMENTS:
        return {"error": f"goal must be one of {list(GOAL_ADJUSTMENTS)}"}
    if sex not in ("male", "female"):
        return {"error": "sex must be 'male' or 'female'"}

    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if sex == "male" else -161)
    tdee = bmr * ACTIVITY_MULTIPLIERS[activity_level]
    target = max(1200, tdee + GOAL_ADJUSTMENTS[goal])

    protein_g = weight_kg * (2.0 if goal == "lose" else 1.6)
    fat_g = target * 0.25 / 9
    carbs_g = max(0, (target - protein_g * 4 - fat_g * 9) / 4)

    return {
        "bmr": round(bmr),
        "tdee": round(tdee),
        "calorie_target": round(target),
        "protein_g": round(protein_g),
        "carbs_g": round(carbs_g),
        "fat_g": round(fat_g),
    }


ALLERGEN_KEYWORDS = {
    "peanut": ["peanut", "groundnut"],
    "tree nut": ["almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut", "macadamia"],
    "dairy": ["milk", "cheese", "butter", "yogurt", "yoghurt", "cream", "ghee", "paneer", "whey"],
    "egg": ["egg", "mayonnaise"],
    "gluten": ["wheat", "flour", "bread", "pasta", "barley", "rye", "couscous", "seitan", "roti"],
    "soy": ["soy", "tofu", "tempeh", "edamame", "miso"],
    "fish": ["salmon", "tuna", "cod", "tilapia", "fish", "anchovy", "sardine"],
    "shellfish": ["shrimp", "prawn", "crab", "lobster", "mussel", "clam", "oyster", "scallop"],
    "sesame": ["sesame", "tahini"],
}


def check_allergens(ingredients: list[str], allergies: list[str]) -> dict:
    """Check a list of ingredients against the user's allergies.

    Always call this on every meal's ingredients before presenting it.

    Args:
        ingredients: Ingredient names, e.g. ["oats", "almond milk"].
        allergies: User allergies, e.g. ["tree nut", "dairy"].

    Returns:
        A dict with "safe" (bool) and any "conflicts" found.
    """
    conflicts = []
    for allergy in allergies:
        key = allergy.strip().lower()
        keywords = ALLERGEN_KEYWORDS.get(key, [key])
        for ingredient in ingredients:
            name = ingredient.lower()
            # "almond milk" / "coconut milk" are dairy-free
            if key == "dairy" and any(p in name for p in ("almond", "oat", "soy", "coconut", "rice")):
                continue
            if any(k in name for k in keywords):
                conflicts.append({"ingredient": ingredient, "allergy": allergy})
    return {"safe": not conflicts, "conflicts": conflicts}


def sum_daily_totals(meals: list[dict]) -> dict:
    """Add up nutrition for a full day of meals.

    Args:
        meals: List of meals, each with protein_g, carbs_g and fat_g.

    Returns:
        Daily totals for protein, carbs, fat and calories.
    """
    protein = sum(float(m.get("protein_g", 0)) for m in meals)
    carbs = sum(float(m.get("carbs_g", 0)) for m in meals)
    fat = sum(float(m.get("fat_g", 0)) for m in meals)
    return {
        "protein_g": round(protein, 1),
        "carbs_g": round(carbs, 1),
        "fat_g": round(fat, 1),
        "calories": round(protein * 4 + carbs * 4 + fat * 9, 1),
    }


def build_grocery_list(items: list[dict]) -> dict:
    """Combine ingredients from all meals into one grocery list.

    Args:
        items: List of {"name": str, "quantity": float, "unit": str}.

    Returns:
        A merged grocery list with quantities summed per ingredient and unit.
    """
    merged: dict[tuple[str, str], float] = {}
    for item in items:
        key = (str(item.get("name", "")).strip().lower(), str(item.get("unit", "")).strip().lower())
        merged[key] = merged.get(key, 0) + float(item.get("quantity", 0))
    return {
        "grocery_list": [
            {"name": name, "quantity": round(qty, 2), "unit": unit}
            for (name, unit), qty in sorted(merged.items())
        ]
    }


def analyze_meal_plan(
    days: list[dict],
    allergies: list[str],
    calorie_target: float,
    protein_target_g: float,
) -> dict:
    """Check a whole multi-day meal plan in ONE call.

    Calculates every meal's calories, checks every ingredient for allergens,
    totals each day against the targets and builds the grocery list.

    Args:
        days: List of days. Each day is
            {"day": "Monday", "meals": [
                {"name": str, "protein_g": float, "carbs_g": float, "fat_g": float,
                 "ingredients": [{"name": str, "quantity": float, "unit": str}]}
            ]}
        allergies: User allergies, e.g. ["peanut"]. Use [] if none.
        calorie_target: Daily calorie target.
        protein_target_g: Daily protein target in grams.

    Returns:
        Per-day totals and status, all allergen conflicts, and a grocery list.
    """
    day_results = []
    all_conflicts = []
    all_items = []

    for day in days:
        meals_out = []
        for meal in day.get("meals", []):
            ingredients = meal.get("ingredients", [])
            names = [str(i.get("name", "")) for i in ingredients]
            all_items.extend(ingredients)

            check = check_allergens(names, allergies)
            for c in check["conflicts"]:
                all_conflicts.append({"day": day.get("day"), "meal": meal.get("name"), **c})

            kcal = calculate_calories(
                float(meal.get("protein_g", 0)),
                float(meal.get("carbs_g", 0)),
                float(meal.get("fat_g", 0)),
            )["estimated_calories"]
            meals_out.append({"name": meal.get("name"), "calories": kcal})

        totals = sum_daily_totals(day.get("meals", []))
        diff_pct = (totals["calories"] - calorie_target) / calorie_target * 100 if calorie_target else 0
        day_results.append({
            "day": day.get("day"),
            "meals": meals_out,
            "totals": totals,
            "calorie_diff_pct": round(diff_pct, 1),
            "within_target": abs(diff_pct) <= 5,
            "protein_met": totals["protein_g"] >= protein_target_g * 0.95,
        })

    return {
        "all_safe": not all_conflicts,
        "allergen_conflicts": all_conflicts,
        "days": day_results,
        "grocery_list": build_grocery_list(all_items)["grocery_list"],
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="meal_planning_agent",
    # Tries each model in FALLBACK_MODELS until one works.
    model=FallbackGemini(
        model=FALLBACK_MODELS[0],
        # Retry each model only briefly, then move on to the next one.
        retry_options=types.HttpRetryOptions(
            attempts=2,
            initial_delay=2,
            exp_base=2,
            http_status_codes=[429, 500, 502, 503, 504],
        ),
    ),
    description="An AI agent that creates personalized meal plans.",
    instruction="""
    You are an AI Meal Planning Agent.

    1. Gather what you need from the user:
       - Daily calorie and protein target. If they don't know it, ask for
         weight, height, age, sex, activity level and goal, then call
         `calculate_daily_targets`.
       - Dietary preferences (vegetarian, vegan, halal, keto, cuisine...)
       - Food allergies
       - Number of meals per day and number of days to plan
       Ask only for what's missing. Don't ask again for things the user already said.

       Default to 7 days if the user doesn't say.

    2. Draft the WHOLE plan (all days, all meals) in your head first. Each
       meal needs a name, ingredients with quantity and unit, and protein,
       carbs and fat in grams. Vary meals across the week; leftovers and
       repeated breakfasts are fine to keep it practical.

    3. Call `analyze_meal_plan` ONCE with the entire plan. Do not call it
       per meal or per day; that makes the agent slow.

    4. Read the result. If `all_safe` is false, or a day is not
       `within_target` or not `protein_met`, adjust those meals and call
       `analyze_meal_plan` once more. Don't loop more than twice.

    5. Present the final plan: one section per day with a table of meals
       (name, ingredients, protein, carbs, fat, calories) and the day's
       totals vs target. End with the grocery list from the tool result.

    Use simple, affordable, easy-to-find ingredients unless the user asks
    otherwise.
    Label all nutrition values as estimates, and say this isn't medical
    advice. Suggest a doctor or dietitian for medical conditions.
    """,
    # Only expose the batch tool (plus targets) so the model can't fall back
    # to dozens of per-meal calls. The small helpers are used inside it.
    tools=[
        calculate_daily_targets,
        analyze_meal_plan,
    ],
)
