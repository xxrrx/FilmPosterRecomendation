"""
test_api.py — Automated test cho tat ca endpoints
Chay: python test_api.py  (sau khi uvicorn da chay)
"""

import requests
import sys

BASE = "http://127.0.0.1:8000"


def check(name, r, expected_status=200):
    if r.status_code == expected_status:
        print(f"  [PASS] {name}")
        return True
    else:
        print(f"  [FAIL] {name} — HTTP {r.status_code}: {r.text[:200]}")
        return False


def run_tests():
    print("=" * 55)
    print("CHAY AUTOMATED TEST MODULE 4")
    print("=" * 55)
    results = []

    # Test 1: Root
    r = requests.get(f"{BASE}/")
    results.append(check("GET /", r))

    # Test 2: Danh sach phim
    r = requests.get(f"{BASE}/api/movies?page=1&limit=5")
    ok = check("GET /api/movies?limit=5", r)
    results.append(ok)
    if ok:
        data = r.json()
        assert "movies" in data, "Thieu key 'movies'"
        assert len(data["movies"]) == 5, f"Can 5 phim, nhan {len(data['movies'])}"
        assert "total" in data
        print(f"         Total phim: {data['total']}")

    # Test 3: Tim kiem phim
    r = requests.get(f"{BASE}/api/movies/search?q=avatar")
    ok = check("GET /api/movies/search?q=avatar", r)
    results.append(ok)
    if ok:
        data = r.json()
        print(f"         Tim thay: {data['total']} phim")

    # Test 4: Chi tiet phim — lay movie_id tu danh sach
    r_list = requests.get(f"{BASE}/api/movies?limit=1")
    if r_list.status_code == 200 and r_list.json()["movies"]:
        first_id = r_list.json()["movies"][0]["movie_id"]
        r = requests.get(f"{BASE}/api/movies/{first_id}")
        ok = check(f"GET /api/movies/{first_id}", r)
        results.append(ok)
        if ok:
            data = r.json()
            assert "title" in data
            assert "cluster_id" in data
            assert "predicted_genres" in data
            print(f"         Phim: {data['title']} | Cluster: {data['cluster_id']}")

        # Test 5: Goi y phim
        r = requests.get(f"{BASE}/api/movies/{first_id}/recommend?alpha=0.5&top_k=10")
        ok = check(f"GET /api/movies/{first_id}/recommend", r)
        results.append(ok)
        if ok:
            data = r.json()
            assert "recommendations" in data
            print(f"         So goi y: {len(data['recommendations'])}")
            if data["recommendations"]:
                top = data["recommendations"][0]
                print(f"         Top 1: {top['title']} (sim={top['similarity']})")

        # Test 5b: Goi y voi alpha khac
        r = requests.get(f"{BASE}/api/movies/{first_id}/recommend?alpha=0.8")
        results.append(check(f"GET /api/movies/{first_id}/recommend?alpha=0.8", r))

    # Test 6: Association rules
    r = requests.get(f"{BASE}/api/genres/rules")
    ok = check("GET /api/genres/rules", r)
    results.append(ok)
    if ok:
        data = r.json()
        assert "rules" in data
        print(f"         So luat: {data['total']}")
        if data["rules"]:
            rule = data["rules"][0]
            print(f"         Top rule: {rule['antecedent']} -> {rule['consequent']} (conf={rule['confidence']})")

    # Test 7: Clusters
    r = requests.get(f"{BASE}/api/clusters")
    ok = check("GET /api/clusters", r)
    results.append(ok)
    if ok:
        data = r.json()
        assert "clusters" in data
        print(f"         So cum: {data['total_clusters']}")

    # Test 8: Movies trong cluster
    r_clusters = requests.get(f"{BASE}/api/clusters")
    if r_clusters.status_code == 200 and r_clusters.json()["clusters"]:
        cid = r_clusters.json()["clusters"][0]["cluster_id"]
        r = requests.get(f"{BASE}/api/clusters/{cid}/movies?limit=5")
        ok = check(f"GET /api/clusters/{cid}/movies", r)
        results.append(ok)
        if ok:
            data = r.json()
            print(f"         Cum {cid}: {data['movie_count']} phim")

    # Tong ket
    print("=" * 55)
    passed = sum(results)
    total  = len(results)
    print(f"KET QUA: {passed}/{total} tests PASS")
    if passed == total:
        print("Tat ca tests PASS! Module 4 hoan thanh.")
    else:
        print("Mot so test FAIL. Kiem tra server va thu lai.")
    print("=" * 55)
    return passed == total


if __name__ == "__main__":
    try:
        run_tests()
    except requests.exceptions.ConnectionError:
        print("[ERROR] Khong ket duoc server. Hay chay truoc:")
        print("  cd backend && uvicorn app:app --reload --port 8000")
        sys.exit(1)
