#!/bin/bash
cd /opt/mem0

# 保存本地修改的文件
cp server/main.py server/main.py.bak
cp server/.env server/.env.bak
cp server/docker-compose.yaml server/docker-compose.yaml.bak
cp server/requirements.txt server/requirements.txt.bak

# 拉取最新代码
git stash
git pull origin main 2>/dev/null || git pull origin master 2>/dev/null

# 恢复本地修改
cp server/main.py.bak server/main.py
cp server/.env.bak server/.env
cp server/docker-compose.yaml.bak server/docker-compose.yaml
cp server/requirements.txt.bak server/requirements.txt

# 重新构建并启动（只有代码库有更新时才重建）
cd server
docker compose up -d --build 2>&1 | tee /var/log/mem0-update.log
