#!/usr/bin/env bash
set -euo pipefail

if (( EUID != 0 )); then
    exec sudo bash "$0" "$@"
fi

mkdir -p /etc/nginx/conf.d
cat > /etc/nginx/conf.d/port-forward.conf <<'EOF'
server {
    listen 0.0.0.0:5050;
    client_max_body_size 0;

    client_header_timeout 12h;
    client_body_timeout 12h;
    keepalive_timeout 12h;
    send_timeout 12h;

    location / {
        proxy_pass http://172.17.213.144:5050;
        proxy_connect_timeout 12h;
        proxy_send_timeout 12h;
        proxy_read_timeout 12h;
    }
}

server {
    listen 0.0.0.0:29000;
    client_max_body_size 0;

    client_header_timeout 12h;
    client_body_timeout 12h;
    keepalive_timeout 12h;
    send_timeout 12h;

    location / {
        proxy_pass http://172.17.213.144:29000;
        proxy_connect_timeout 12h;
        proxy_send_timeout 12h;
        proxy_read_timeout 12h;
    }
}
EOF

nginx -t
if nginx -s reload; then
    echo 'nginx 已重载：5050/29000 端口转发配置已生效'
else
    exec nginx -g 'daemon off;'
fi
