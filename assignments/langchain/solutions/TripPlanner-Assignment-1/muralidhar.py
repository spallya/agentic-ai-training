#!/usr/bin/env python3
"""
Trip Planner – Fixed Chain (Groq API)

Steps:
1. Research 3 top tourist attractions for a given city.
2. Summarise them into bullet points.
3. Generate a short 1day itinerary.

Author: <Muralidhar Nayani>
"""

import os
import json
import sys
import textwrap
from typing import List, Optional

import requests

# ----------------------------------------------------------------------
# Configuration ---------------------------------------------------------
# ----------------------------------------------------------------------
# Put your Groq API key in an environment variable or replace the placeholder.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/chat/completions"
# The model you want to use – adjust if you have a different one.
GROQ_MODEL = "llama3-8b-8192-8"


# ----------------------------------------------------------------------
# Helper functions ------------------------------------------------------
# ----------------------------------------------------------------------
def _call_groq(messages: List[dict]) -> Optional[str]:
    """
    Low‑level wrapper around the Groq chat‑completion endpoint.
    Returns the content of the first choice, or None on error.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.7,          # a little creativity, but still deterministic
        "max_tokens": 1024,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:                     # pragma: no cover
        print(f"[ERROR] Groq request failed: {exc}", file=sys.stderr)
        return None


# ----------------------------------------------------------------------
# Chain steps -----------------------------------------------------------
# ----------------------------------------------------------------------
def research_top_attractions(city: str) -> Optional[str]:
    """
    Step 1 – ask Groq for the three best tourist attractions in *city*.
    The prompt is deliberately short; Groq will return a plain list.
    """
    prompt = (
        f"List the three most popular tourist attractions in {city}. "
        "Give each attraction on a separate line, and include a short (1‑2 sentence) description."
    )
    return _call_groq([{"role": "user", "content": prompt}])


def summarise_attractions(raw_text: str) -> str:
    """
    Step 2 – turn the raw answer into a clean bullet‑point list.
    We simply split on line‑breaks and prepend a dash.
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    bullet_points = "\n".join(f"- {ln}" for ln in lines)
    return bullet_points


def generate_itinerary(raw_text: str) -> str:
    """
    Step 3 – build a 1‑day itinerary from the three attractions.
    The itinerary is numbered and adds a tiny suggestion for the order.
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    itinerary = ["1‑Day Itinerary:"]
    for idx, line in enumerate(lines, start=1):
        # Extract just the attraction name (ignore the description after a period)
        name = line.split(".")[0]
        itinerary.append(f"{idx}. Visit {name}")
    return "\n".join(itinerary)


# ----------------------------------------------------------------------
# Main orchestration ----------------------------------------------------
# ----------------------------------------------------------------------
def plan_trip(city: str) -> None:
    """
    Runs the three‑step fixed chain and prints the results.
    """
    print(f"\n=== Trip Planner for {city} ===\n")

    # 1 Research
    raw_attractions = research_top_attractions(city)
    if not raw_attractions:
        print("[ERROR] Could not retrieve attractions.", file=sys.stderr)
        return

    print("Raw attractions from Groq:")
    print(textwrap.indent(raw_attractions, "    "))
    print()

    # 2 Summarise
    summary = summarise_attractions(raw_attractions)
    print("Summary (bullet points):")
    print(summary)
    print()

    # 3 Itinerary
    itinerary = generate_itinerary(raw_attractions)
    print("1‑Day Itinerary:")
    print(itinerary)
    print()


# ----------------------------------------------------------------------
# Entry point -----------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        destination = " ".join(sys.argv[1:])
    else:
        destination = input("Enter a destination city: ").strip()

    if not destination:
        print("[ERROR] No city supplied.", file=sys.stderr)
        sys.exit(1)

    # Simple sanity check – the API key must be set.
    if GROQ_API_KEY == "ACTUAL_GROW_API_KEY":
        print("[ERROR] Please set your Groq API key (environment variable GROQ_API_KEY).", file=sys.stderr)
        sys.exit(1)

    plan_trip(destination)