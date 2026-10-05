#!/usr/bin/env bash
# Static IP for the direct Ethernet link (Linux).
#   sudo ./scripts/setup_static_ip_linux.sh server eth0     -> 192.168.1.1/24
#   sudo ./scripts/setup_static_ip_linux.sh client enp3s0   -> 192.168.1.2/24
#   sudo ./scripts/setup_static_ip_linux.sh reset eth0      -> back to DHCP
# Also opens TCP 5050 on the worker if ufw is active.
set -e
ROLE="${1:?role: server|client|reset}"; IFACE="${2:?interface name, see: ip -br link}"
case "$ROLE" in server) IP=192.168.1.1;; client) IP=192.168.1.2;; reset) IP="";; *) echo "role must be server|client|reset"; exit 1;; esac

if command -v nmcli >/dev/null 2>&1 && nmcli -t -f DEVICE,STATE dev | grep -q "^$IFACE:"; then
  nmcli con delete offload-link >/dev/null 2>&1 || true
  if [ "$ROLE" = reset ]; then
    nmcli con add type ethernet ifname "$IFACE" con-name offload-link ipv4.method auto >/dev/null
  else
    nmcli con add type ethernet ifname "$IFACE" con-name offload-link ipv4.method manual \
        ipv4.addresses "$IP/24" ipv6.method ignore >/dev/null
  fi
  nmcli con up offload-link
else
  ip addr flush dev "$IFACE"
  [ -n "$IP" ] && ip addr add "$IP/24" dev "$IFACE"
  ip link set "$IFACE" up
fi
if [ "$ROLE" = server ] && command -v ufw >/dev/null 2>&1 && ufw status | grep -q active; then
  ufw allow 5050/tcp
fi
echo "Done."; ip -br addr show "$IFACE"
[ -n "$IP" ] && echo "Test from the other machine:  ping ${IP}"
