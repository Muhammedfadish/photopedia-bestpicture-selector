import os
import shutil
from nima_score.scorer import NIMAScorer

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "best_images"
SCORE_THRESHOLD = 7.0

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

scorer = NIMAScorer()

print("\n📸 Image Scoring Results:\n")

for file in os.listdir(INPUT_FOLDER):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(INPUT_FOLDER, file)
        try:
            final_score, aesthetic, sharpness, faces = scorer.score_image(file_path)

            print(f"{file} -> Score: {final_score} | Aesthetic: {aesthetic} | Blur: {round(sharpness)} | Faces: {faces}")

            if final_score >= SCORE_THRESHOLD:
                output_path = os.path.join(OUTPUT_FOLDER, file)
                if not os.path.exists(output_path):
                    shutil.copy2(file_path, output_path)

        except Exception as e:
            print(f"❌ Error with {file}: {e}")
