from agents.app import run_agentic

if __name__ == "__main__":
    prompt = "Write a function that recommends a hobby based on user profile."
    res = run_agentic(prompt=prompt, target_lang="Python", max_iter=2)

    print("\n=== FINAL CODE ===\n")
    print(res["code"])

    print("\n=== LAST REVIEW ===\n")
    print(res["review_feedback"])
