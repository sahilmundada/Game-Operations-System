import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoints():
    print("Testing GET /leaderboard...")
    try:
        r = requests.get(f"{BASE_URL}/leaderboard")
        print(f"Status: {r.status_code}")
        print(f"Response: {r.json() if r.status_code == 200 else r.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTesting GET /flagged-players...")
    try:
        r = requests.get(f"{BASE_URL}/flagged-players")
        print(f"Status: {r.status_code}")
        print(f"Response: {len(r.json().get('flagged_players', []))} flagged players")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTesting GET /players/PC001...")
    try:
        r = requests.get(f"{BASE_URL}/players/PC001")
        print(f"Status: {r.status_code}")
        print(f"Response: {r.json() if r.status_code == 200 else r.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTesting POST /submit-score...")
    payload = {
        "player_id": "PC006",
        "match_id": "M006",
        "region": "India",
        "device": "Android",
        "ping": 50,
        "score": 2000,
        "kills": 10,
        "deaths": 5,
        "match_duration_seconds": 300
    }
    try:
        r = requests.post(f"{BASE_URL}/submit-score", json=payload)
        print(f"Status: {r.status_code}")
        print(f"Response: {r.json() if r.status_code == 200 or r.status_code == 201 else r.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTesting POST /run-analysis...")
    try:
        r = requests.post(f"{BASE_URL}/run-analysis")
        print(f"Status: {r.status_code}")
        print(f"Response: {r.json() if r.status_code == 200 else r.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTesting GET /matchmaking...")
    try:
        r = requests.get(f"{BASE_URL}/matchmaking")
        print(f"Status: {r.status_code}")
        print(f"Response: {len(r.json())} match groups")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTesting GET /stats...")
    try:
        r = requests.get(f"{BASE_URL}/stats")
        print(f"Status: {r.status_code}")
        print(f"Response: {r.json() if r.status_code == 200 else r.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_endpoints()
