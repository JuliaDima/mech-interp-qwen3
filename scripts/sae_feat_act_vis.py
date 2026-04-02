from pathlib import Path

artifact_path = "plots/SAE_feature_activations/feature_investigation.md"
base_img_path = "/home/eid23/mechinterp-qwen-3B-Instruct/mechinterp-qwen3/runs/addition/2026-03-23_1427/operand_plots"

files = sorted([f.name for f in Path(base_img_path).glob("*.png")])
Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)

with open(artifact_path, "w") as f:
    f.write("# SAE Feature Activation Visualizer\n\n")

    f.write("<div style='display: flex; flex-wrap: wrap; gap: 10px;'>\n")
    for file in files:
        # Using relative paths for seamless IDE Markdown Previewer support
        rel_path = f"{base_img_path}/{file}"
        f.write("  <div style='text-align: center;'>\n")
        f.write(f"    <img src='{rel_path}' alt='{file.split('.')[0]}' width='400'/>\n")
        f.write(f"    <p><b>{file.split('.')[0]}</b></p>\n")
        f.write("  </div>\n")
    f.write("</div>\n")

print(f"Artifact created at {artifact_path}")
