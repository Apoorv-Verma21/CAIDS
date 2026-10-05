import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


ATTACKS = ["Fuzzers", "Analysis", "Backdoor", "DoS", "Exploits",
           "Generic", "Reconnaissance", "Shellcode", "Worms"]


def random_ip(rng, private=True):
    if private:
        return "10.0.{}.{}".format(rng.integers(0, 32), rng.integers(1, 255))
    return "{}.{}.{}.{}".format(
        rng.integers(11, 200), rng.integers(0, 255), rng.integers(0, 255), rng.integers(1, 255)
    )


def build(n_flows, seed, window_seconds):
    rng = np.random.default_rng(seed)
    base_ms = 1_600_000_000_000

    servers = [random_ip(rng) for _ in range(8)]
    clients = [random_ip(rng) for _ in range(60)]
    attackers = [random_ip(rng, private=False) for _ in range(6)]

    rows = []
    span_ms = n_flows * 40
    for i in range(n_flows):
        t = base_ms + int(i * span_ms / n_flows) + int(rng.integers(0, 50))
        roll = rng.random()

        if roll < 0.90:
            src = clients[rng.integers(0, len(clients))]
            dst = servers[rng.integers(0, len(servers))]
            dport = int(rng.choice([80, 443, 53, 22, 3306]))
            label, attack = 0, "Benign"
            in_bytes = int(rng.lognormal(6.5, 1.1))
            in_pkts = max(1, int(in_bytes / 600))
        else:
            src = attackers[rng.integers(0, len(attackers))]
            attack = ATTACKS[int(rng.integers(0, len(ATTACKS)))]
            label = 1
            if attack in ("Reconnaissance", "Fuzzers"):
                dst = clients[rng.integers(0, len(clients))]
                dport = int(rng.integers(1, 65535))
                in_bytes = int(rng.lognormal(4.0, 0.8))
            elif attack == "DoS":
                dst = servers[0]
                dport = 80
                in_bytes = int(rng.lognormal(8.5, 0.6))
            else:
                dst = servers[rng.integers(0, len(servers))]
                dport = int(rng.choice([80, 443, 445, 3389]))
                in_bytes = int(rng.lognormal(7.0, 1.0))
            in_pkts = max(1, int(in_bytes / 500))

        out_bytes = int(in_bytes * rng.uniform(0.2, 1.4))
        out_pkts = max(1, int(out_bytes / 600))
        duration = int(rng.integers(1, 20000))

        rows.append({
            "IPV4_SRC_ADDR": src,
            "L4_SRC_PORT": int(rng.integers(1024, 65535)),
            "IPV4_DST_ADDR": dst,
            "L4_DST_PORT": dport,
            "PROTOCOL": int(rng.choice([6, 6, 6, 17, 1])),
            "L7_PROTO": float(rng.integers(0, 200)),
            "IN_BYTES": in_bytes,
            "IN_PKTS": in_pkts,
            "OUT_BYTES": out_bytes,
            "OUT_PKTS": out_pkts,
            "TCP_FLAGS": int(rng.integers(0, 255)),
            "CLIENT_TCP_FLAGS": int(rng.integers(0, 255)),
            "SERVER_TCP_FLAGS": int(rng.integers(0, 255)),
            "FLOW_DURATION_MILLISECONDS": duration,
            "DURATION_IN": duration,
            "DURATION_OUT": duration,
            "MIN_TTL": int(rng.integers(0, 255)),
            "MAX_TTL": int(rng.integers(0, 255)),
            "LONGEST_FLOW_PKT": int(rng.integers(40, 1514)),
            "SHORTEST_FLOW_PKT": int(rng.integers(40, 200)),
            "MIN_IP_PKT_LEN": int(rng.integers(40, 200)),
            "MAX_IP_PKT_LEN": int(rng.integers(40, 1514)),
            "SRC_TO_DST_SECOND_BYTES": float(in_bytes),
            "DST_TO_SRC_SECOND_BYTES": float(out_bytes),
            "RETRANSMITTED_IN_PKTS": int(rng.integers(0, 5)),
            "RETRANSMITTED_OUT_PKTS": int(rng.integers(0, 5)),
            "SRC_TO_DST_AVG_THROUGHPUT": int(rng.integers(0, 100000)),
            "DST_TO_SRC_AVG_THROUGHPUT": int(rng.integers(0, 100000)),
            "NUM_PKTS_UP_TO_128_BYTES": int(rng.integers(0, 50)),
            "NUM_PKTS_128_TO_256_BYTES": int(rng.integers(0, 50)),
            "NUM_PKTS_256_TO_512_BYTES": int(rng.integers(0, 50)),
            "NUM_PKTS_512_TO_1024_BYTES": int(rng.integers(0, 50)),
            "NUM_PKTS_1024_TO_1514_BYTES": int(rng.integers(0, 50)),
            "TCP_WIN_MAX_IN": int(rng.integers(0, 65535)),
            "TCP_WIN_MAX_OUT": int(rng.integers(0, 65535)),
            "ICMP_TYPE": int(rng.integers(0, 255)),
            "DNS_QUERY_ID": int(rng.integers(0, 65535)),
            "DNS_QUERY_TYPE": int(rng.integers(0, 50)),
            "FTP_COMMAND_RET_CODE": int(rng.integers(0, 500)),
            "FLOW_START_MILLISECONDS": t,
            "FLOW_END_MILLISECONDS": t + duration,
            "SRC_TO_DST_IAT_MIN": float(rng.integers(0, 100)),
            "SRC_TO_DST_IAT_MAX": float(rng.integers(100, 5000)),
            "SRC_TO_DST_IAT_AVG": float(rng.integers(0, 1000)),
            "SRC_TO_DST_IAT_STDDEV": float(rng.integers(0, 500)),
            "DST_TO_SRC_IAT_MIN": float(rng.integers(0, 100)),
            "DST_TO_SRC_IAT_MAX": float(rng.integers(100, 5000)),
            "DST_TO_SRC_IAT_AVG": float(rng.integers(0, 1000)),
            "DST_TO_SRC_IAT_STDDEV": float(rng.integers(0, 500)),
            "Label": label,
            "Attack": attack,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic NetFlow-like data for testing")
    parser.add_argument("--out", default="data/sample_flows.csv")
    parser.add_argument("--rows", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-seconds", type=int, default=60)
    args = parser.parse_args()

    df = build(args.rows, args.seed, args.window_seconds)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)

    print("wrote {} rows to {}".format(len(df), args.out))
    print("attack rate: {:.2f}%".format(100 * df["Label"].mean()))
    print(df["Attack"].value_counts().to_string())


if __name__ == "__main__":
    main()
