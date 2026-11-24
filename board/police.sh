#!/bin/sh
# Limit the board's upload to one TCP peer with an egress policer (the kernel has no shaping qdisc).
# Only packets of 256 bytes or more to <dst_ip>:<dst_port> are policed, so pure ACKs of downloads pass.
#   sh police.sh set <iface> <dst_ip> <dst_port> <kbit>
#   sh police.sh clear <iface>
#   sh police.sh show <iface>
set -e
cmd=$1
dev=$2
case "$cmd" in
  set)
    dst=$3; port=$4; rate=$5
    # bucket of 0.25 s at the rate, at least 4 KB, so TCP can reach the rate without constant drops
    burst=$((rate * 1000 / 8 / 4))
    [ "$burst" -lt 4096 ] && burst=4096
    tc qdisc add dev "$dev" clsact 2>/dev/null || true
    tc filter del dev "$dev" egress 2>/dev/null || true
    # prio 1: small packets (IP total length < 256) to the peer pass unpoliced
    tc filter add dev "$dev" egress prio 1 protocol ip u32 \
      match ip dst "$dst"/32 match ip dport "$port" 0xffff match u16 0x0000 0xff00 at 2 action ok
    # prio 2: everything else to the peer is policed
    tc filter add dev "$dev" egress prio 2 protocol ip u32 \
      match ip dst "$dst"/32 match ip dport "$port" 0xffff \
      action police rate "${rate}kbit" burst "$burst" mtu 2k drop
    ;;
  clear)
    tc filter del dev "$dev" egress 2>/dev/null || true
    tc qdisc del dev "$dev" clsact 2>/dev/null || true
    ;;
  show)
    tc -s filter show dev "$dev" egress
    ;;
esac
