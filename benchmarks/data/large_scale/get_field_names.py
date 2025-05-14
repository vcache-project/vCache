#!/usr/bin/env python

import ijson

json_file = "/data/acuadron/VectorQ/benchmarks/data/large_scale/sem_benchmark_search_queries.json"

with open(json_file, 'rb') as f:
    for first_item in ijson.items(f, 'item'):
        print("Keys in the first entry:")
        for key in sorted(first_item.keys()):
            print(f"- {key}")
        break
