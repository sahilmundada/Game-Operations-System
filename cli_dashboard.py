import os
import sys
import json
import requests

BASE_URL = "http://localhost:8000"

# ANSI Colors
GREEN = "\033[92m"
TEAL = "\033[96m"
YELLOW = "\033[93m"
ORANGE = "\033[38;5;208m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
UNDERLINE = "\033[4m"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    print(f"\n{BOLD}{CYAN}================================================================================{RESET}")
    print(f"   {BOLD}{title.upper()}{RESET}")
    print(f"{BOLD}{CYAN}================================================================================{RESET}\n")

def print_table(headers, rows):
    if not rows:
        print("No records found.")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            clean_val = str(val)
            for color in [GREEN, TEAL, YELLOW, ORANGE, RED, RESET, BOLD, CYAN, UNDERLINE]:
                clean_val = clean_val.replace(color, "")
            widths[i] = max(widths[i], len(clean_val))
            
    header_str = " | ".join(f"{headers[i]:<{widths[i]}}" for i in range(len(headers)))
    print(f"{BOLD}{header_str}{RESET}")
    print("-" * (sum(widths) + 3 * (len(headers) - 1)))
    
    for row in rows:
        formatted_cols = []
        for i, val in enumerate(row):
            val_str = str(val)
            clean_val = val_str
            for color in [GREEN, TEAL, YELLOW, ORANGE, RED, RESET, BOLD, CYAN, UNDERLINE]:
                clean_val = clean_val.replace(color, "")
            diff = len(val_str) - len(clean_val)
            formatted_cols.append(f"{val_str:<{widths[i] + diff}}")
        print(" | ".join(formatted_cols))

def get_zone_color(zone):
    z = zone.lower()
    if z == "clean":
        return GREEN
    elif z == "watch":
        return TEAL
    elif z == "review":
        return YELLOW
    elif z == "restricted":
        return ORANGE
    elif z == "flagged":
        return RED
    return RESET

def show_global_leaderboard():
    print_header("Global Leaderboard")
    limit_str = input("Enter limit (default 20): ").strip()
    limit = int(limit_str) if limit_str.isdigit() else 20
    
    try:
        r = requests.get(f"{BASE_URL}/leaderboard?limit={limit}")
        if r.status_code != 200:
            print(f"Error fetching leaderboard: {r.text}")
            return
        data = r.json()
        
        headers = ["Rank", "Player ID", "Region", "Score", "Kills", "Deaths", "Skill Tier", "Match Group", "Conf Score"]
        rows = []
        for p in data.get("leaderboard", []):
            conf = p.get("confidence_score", 0.0)
            if conf <= 20: zone = "Clean"
            elif conf <= 40: zone = "Watch"
            else: zone = "Review"
            color = get_zone_color(zone)
            rows.append([
                p.get("global_rank"),
                p.get("player_id"),
                p.get("region"),
                p.get("score"),
                p.get("kills"),
                p.get("deaths"),
                p.get("skill_tier"),
                p.get("match_group_id") or "N/A",
                f"{color}{conf:.1f}{RESET}"
            ])
        print_table(headers, rows)
        print(f"\nTotal players in system: {data.get('total_players')}")
        print(f"Cheaters excluded from leaderboard: {data.get('flagged_excluded')}")
    except Exception as e:
        print(f"Error connecting to server: {e}")

def show_region_leaderboard():
    print_header("Region-wise Leaderboard")
    print("Available regions: India, SEA, Europe, NA, LatAm, Middle_East")
    region = input("Enter region name (e.g. India): ").strip()
    if not region:
        print("Region name is required.")
        return
        
    limit_str = input("Enter limit (default 20): ").strip()
    limit = int(limit_str) if limit_str.isdigit() else 20
    
    try:
        r = requests.get(f"{BASE_URL}/leaderboard?region={region}&limit={limit}")
        if r.status_code != 200:
            print(f"Error fetching leaderboard: {r.text}")
            return
        data = r.json()
        
        headers = ["Reg Rank", "Global Rank", "Player ID", "Score", "Kills", "Deaths", "Skill Tier", "Match Group", "Conf Score"]
        rows = []
        for p in data.get("leaderboard", []):
            conf = p.get("confidence_score", 0.0)
            if conf <= 20: zone = "Clean"
            elif conf <= 40: zone = "Watch"
            else: zone = "Review"
            color = get_zone_color(zone)
            rows.append([
                p.get("region_rank"),
                p.get("global_rank"),
                p.get("player_id"),
                p.get("score"),
                p.get("kills"),
                p.get("deaths"),
                p.get("skill_tier"),
                p.get("match_group_id") or "N/A",
                f"{color}{conf:.1f}{RESET}"
            ])
        print_table(headers, rows)
    except Exception as e:
        print(f"Error connecting to server: {e}")

def show_matchmaking():
    print_header("Matchmaking Groups")
    try:
        r = requests.get(f"{BASE_URL}/matchmaking")
        if r.status_code != 200:
            print(f"Error fetching matchmaking: {r.text}")
            return
        data = r.json()
        
        headers = ["Group ID", "Region", "Players Count", "Avg Ping", "Skill Tiers Included"]
        rows = []
        for g in data:
            rows.append([
                g.get("group_id"),
                g.get("region"),
                g.get("player_count"),
                f"{g.get('avg_ping'):.1f} ms",
                ", ".join(g.get("skill_tiers", []))
            ])
        print_table(headers, rows)
        
        print(f"\nTotal active match groups: {len(data)}")
        
        opt = input("\nDo you want to inspect a specific group ID? (y/n): ").strip().lower()
        if opt == 'y':
            g_id = input("Enter Group ID: ").strip()
            for g in data:
                if g.get("group_id") == g_id:
                    print(f"\n{BOLD}Players in Group {g_id}:{RESET}")
                    print(", ".join(g.get("players", [])))
                    return
            print("Group ID not found.")
    except Exception as e:
        print(f"Error connecting to server: {e}")

def show_suspicious_players():
    print_header("Suspicious Players (Review, Restricted, Flagged)")
    try:
        r = requests.get(f"{BASE_URL}/flagged-players")
        if r.status_code != 200:
            print(f"Error fetching flagged players: {r.text}")
            return
        data = r.json()
        
        headers = ["Player ID", "Flag Reason", "Score", "Kills", "KDR", "SPM", "Submitted At"]
        rows = []
        for p in data.get("flagged_players", []):
            rows.append([
                p.get("player_id"),
                p.get("flag_reason"),
                p.get("score"),
                p.get("kills"),
                f"{p.get('kdr'):.2f}" if p.get('kdr') else "0.0",
                f"{p.get('score_per_minute'):.1f}" if p.get('score_per_minute') else "0.0",
                p.get("submitted_at")
            ])
        print_table(headers, rows)
        print(f"\nTotal suspicious players: {len(data.get('flagged_players', []))}")
    except Exception as e:
        print(f"Error connecting to server: {e}")

def submit_player_score():
    print_header("Submit Player Score (Real-time Detection)")
    
    p_id = input("Enter Player ID (e.g. PC007): ").strip()
    m_id = input("Enter Match ID (e.g. M102): ").strip()
    
    print("Regions: India, SEA, Europe, NA, LatAm, Middle_East")
    region = input("Enter Region: ").strip()
    
    print("Devices: Android, iOS, Console, PC")
    device = input("Enter Device: ").strip()
    
    try:
        ping = int(input("Enter Ping (ms): ").strip())
        score = int(input("Enter Score: ").strip())
        kills = int(input("Enter Kills: ").strip())
        deaths = int(input("Enter Deaths: ").strip())
        duration = int(input("Enter Match Duration (seconds): ").strip())
    except ValueError:
        print("Invalid numeric values provided.")
        return
        
    payload = {
        "player_id": p_id,
        "match_id": m_id,
        "region": region,
        "device": device,
        "ping": ping,
        "score": score,
        "kills": kills,
        "deaths": deaths,
        "match_duration_seconds": duration
    }
    
    try:
        r = requests.post(f"{BASE_URL}/submit-score", json=payload)
        if r.status_code not in [200, 201]:
            print(f"Error submitting score: {r.status_code} - {r.text}")
            return
            
        res = r.json()
        print(f"\n{BOLD}{GREEN}Submission processed successfully!{RESET}")
        print("-" * 50)
        print(f"Status        : {res.get('status')}")
        print(f"Player ID     : {res.get('player_id')}")
        
        conf = res.get('confidence_score', 0.0)
        zone = res.get('confidence_zone')
        color = get_zone_color(zone)
        
        print(f"Confidence    : {color}{conf:.1f} / 100  [{res.get('status_label')}]{RESET}")
        print(f"Action        : {res.get('action')}")
        print(f"Cheat Types   : {', '.join(res.get('cheat_types_hit', [])) or 'None'}")
        print(f"Confirmed     : {', '.join(res.get('confirmed_cheats', [])) or 'None'}")
        
        print("\nScore Breakdown:")
        for k, v in res.get("score_breakdown", {}).items():
            print(f"  - {k:<25}: {v}")
            
        print("\nComputed Features:")
        for k, v in res.get("features", {}).items():
            print(f"  - {k:<25}: {v:.2f}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error connecting to server: {e}")

def main():
    while True:
        clear_screen()
        print(f"{BOLD}{GREEN}================================================================================{RESET}")
        print(f"{BOLD}{GREEN}                 GAME OPERATIONS INTELLIGENCE SYSTEM (V2)                      {RESET}")
        print(f"{BOLD}{GREEN}================================================================================{RESET}")
        print(f"1. {BOLD}View Global Leaderboard{RESET}")
        print(f"2. {BOLD}View Region-wise Leaderboard{RESET}")
        print(f"3. {BOLD}View Matchmaking Groups{RESET}")
        print(f"4. {BOLD}View Suspicious Players (Review, Restricted, Flagged){RESET}")
        print(f"5. {BOLD}Submit Player Score (Real-time Prediction){RESET}")
        print(f"6. {BOLD}Exit{RESET}")
        print(f"{GREEN}--------------------------------------------------------------------------------{RESET}")
        
        choice = input("Select an option (1-6): ").strip()
        if choice == '1':
            show_global_leaderboard()
        elif choice == '2':
            show_region_leaderboard()
        elif choice == '3':
            show_matchmaking()
        elif choice == '4':
            show_suspicious_players()
        elif choice == '5':
            submit_player_score()
        elif choice == '6':
            print("\nExiting. Thank you!")
            break
        else:
            print("\nInvalid choice. Press Enter to try again.")
            
        input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    main()
