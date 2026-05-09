"""
setup_assets.py — Tai cac file model va data tu Google Drive ve local.

Cach dung:
    pip install gdown
    python setup_assets.py

Yeu cau: Ban (owner) phai upload cac file len Drive va dien FILE_IDS bên duoi.
"""

import os
import subprocess
import sys

# ─── HUONG DAN CHO OWNER ─────────────────────────────────────────────────────
# 1. Upload tung file len Google Drive
# 2. Chuot phai → "Get link" → "Anyone with the link" → Copy link
# 3. Lay FILE ID tu link:
#    https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing
#    Dan FILE_ID vao bang bên duoi
# ─────────────────────────────────────────────────────────────────────────────

FILE_IDS = {
    # models/
    "models/combined_features.npy":  "https://drive.google.com/file/d/19SGlR2p_25I0SuJnRX7tARXsKJ9zf17C/view?usp=sharing",
    "models/cnn_features.npy":       "https://drive.google.com/file/d/18tTN-7GFHKhyeA4z7-Uf202qtjpvZcg3/view?usp=sharing",
    "models/tfidf_matrix.npy":       "https://drive.google.com/file/d/1v2oULLZJGT854FMrHx-9atndX9CS6qu7/view?usp=sharing",
    "models/cluster_labels.npy":     "https://drive.google.com/file/d/1OFGcJTlJmX4WdpwZR7pLKrZCyVcAMNBY/view?usp=sharing",
    "models/movie_ids.npy":          "https://drive.google.com/file/d/1Ldw5g6QdwwgCJHXqPKo-y5Y8gIxPG8cT/view?usp=sharing",
    "models/kmeans.pkl":             "https://drive.google.com/file/d/1xK-BDtPcCS1MH9uPKzyOO-e3OILTgDW4/view?usp=sharing",
    "models/nb_model.pkl":           "https://drive.google.com/file/d/1JUZ7GvMmhd72UCSv05ptQormPucFODqv/view?usp=sharing",
    "models/mlb_encoder.pkl":        "https://drive.google.com/file/d/1URiWX3qa4SQrjwdP_rqgdsoTYPZVB6jg/view?usp=sharing",
    "models/tfidf_vectorizer.pkl":   "https://drive.google.com/file/d/1KJVuTGGpSkTMaFaP-aEVt3_40AKNPqrx/view?usp=sharing",
    "models/scalers.pkl":            "https://drive.google.com/file/d/1Sn9VIizQTXSWuf043LgvNBPtO3A1J3Uf/view?usp=sharing",
    # data/
    "data/processed/movies_valid.csv": "https://drive.google.com/file/d/12DyqSiqYwtUWadlBSgiASt49hyY-Ibv_/view?usp=sharing",
}


def install_gdown():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])


def download_file(dest_path, file_id):
    if file_id == "PASTE_FILE_ID_HERE":
        print(f"  [SKIP] {dest_path} — chua co file ID, xem huong dan o tren")
        return False
    if os.path.exists(dest_path):
        print(f"  [OK]   {dest_path} — da ton tai, bo qua")
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Dang tai: {dest_path} ...")
    try:
        gdown.download(url, dest_path, quiet=False)
        print(f"  [DONE] {dest_path}")
        return True
    except Exception as e:
        print(f"  [FAIL] {dest_path} — {e}")
        return False


def main():
    print("=" * 55)
    print("SETUP ASSETS — Movie Recommendation System")
    print("=" * 55)

    # Kiem tra gdown
    try:
        import gdown
    except ImportError:
        print("Cai gdown...")
        install_gdown()
        import gdown

    # Chay tu thu muc goc du an
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    success, skipped, failed = 0, 0, 0
    for dest, fid in FILE_IDS.items():
        if fid == "PASTE_FILE_ID_HERE":
            skipped += 1
        elif download_file(dest, fid):
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 55)
    print(f"Ket qua: {success} tai thanh cong | {skipped} chua co ID | {failed} loi")

    if skipped > 0:
        print("\nCon file chua duoc cau hinh — xem HUONG DAN OWNER bên duoi.")

    if failed == 0 and skipped == 0:
        print("\nTat ca files da san sang! Chay server:")
        print("  cd backend && uvicorn app:app --reload --port 8000")
        print("  cd frontend && npm install && npm run dev")
    print("=" * 55)


if __name__ == "__main__":
    main()
