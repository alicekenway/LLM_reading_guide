(ROOT/"demo_results.json").write_text(json.dumps(results,indent=2)+"\n")
print("Saved numerical checks to demo_results.json.")
print("Default demonstrations completed without weights or external network access.")
