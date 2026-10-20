import json, sys, time, urllib.parse, urllib.request
query = sys.argv[1]; out = sys.argv[2]
base = "https://api.openalex.org/works"
params = {"filter": "title_and_abstract.search:" + query + ",from_publication_date:2020-01-01",
          "per-page": "200", "cursor": "*",
          "select": "id,doi,title,publication_year,abstract_inverted_index,primary_location,type"}
rows = []
while True:
    url = base + "?" + urllib.parse.urlencode(params, safe=":,()\"*")
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    if not rows: print("count:", data["meta"]["count"])
    for w in data["results"]:
        inv = w.get("abstract_inverted_index") or {}
        pos = {}
        for word, idx in inv.items():
            for i in idx: pos[i] = word
        abstract = " ".join(pos[i] for i in sorted(pos))
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        rows.append({"id": w["id"], "doi": w.get("doi"), "title": w.get("title"), "year": w.get("publication_year"),
                     "venue": src, "type": w.get("type"), "abstract": abstract})
    nxt = data["meta"].get("next_cursor")
    if not nxt or not data["results"]: break
    params["cursor"] = nxt
    time.sleep(0.3)
json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print("saved", len(rows))
